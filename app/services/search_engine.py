from typing import Union, List, Optional, Dict, Any
from PIL import Image, UnidentifiedImageError
import torch
import clip
import numpy as np
import io
import json
import logging
from app.models.product import Product, SearchResult, SearchQuery
from app.services.s3_handler import S3Handler
from botocore.exceptions import ClientError
import gc
import requests
from io import BytesIO
import os
from pinecone import Pinecone
from typing import List, Dict, Any
from loguru import logger
import aiohttp
from pydantic import HttpUrl, ValidationError
from app.utils.encoders import safe_json_encode

logger = logging.getLogger(__name__)

class ProductSearchEngine:
    """
    Singleton class handling product search using CLIP embeddings and Pinecone.
    Manages model loading, feature extraction, and similarity search.
    """
    
    _instance = None
    
    def __new__(cls):
        """
        Singleton pattern implementation to ensure only one instance exists.
        
        Returns:
            ProductSearchEngine: Single instance of the search engine
        """
        if cls._instance is None:
            cls._instance = super(ProductSearchEngine, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        """
        Initialize search engine with default settings.
        Only runs once due to singleton pattern.
        Sets up device configuration, model placeholders, and S3 paths.
        """
        if not self.initialized:
            try:
                # Add environment validation before initialization
                self.validate_env_vars()
                
                self.device = "cpu"  # Force CPU to save memory
                torch.set_num_threads(1)  # Limit threads
                self.model = None
                self.preprocess = None
                self.model_loaded = False
                self.index_loaded = False
                self.feature_dim = 512
                self.embeddings_file = "embeddings/product_embeddings.pkl"
                logger.info("ProductSearchEngine initialized")
                self.initialized = True
                self.products = {}
                
                # Initialize Pinecone with proper scheme
                pc = Pinecone(
                    api_key=os.getenv('PINECONE_API_KEY'),
                    environment=os.getenv('PINECONE_ENVIRONMENT')
                )
                
                # Get the index with the proper name
                self.index = pc.Index(os.getenv('PINECONE_INDEX_NAME'))
                
                # Initialize CLIP model
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
                self.model_loaded = True
                
                logger.info("Search engine initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize search engine: {str(e)}")
                self.model_loaded = False
                raise

    def validate_env_vars(self):
        """Validate required environment variables"""
        required_vars = ['PINECONE_API_KEY', 'PINECONE_ENVIRONMENT', 'PINECONE_INDEX_NAME']
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    async def ensure_model_loaded(self):
        """Load CLIP model if not already loaded"""
        if not self.model_loaded:
            try:
                logger.info("Loading CLIP model...")
                # Set to eval mode immediately to save memory
                self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=True)
                self.model.eval()
                
                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Force garbage collection
                gc.collect()
                
                self.model_loaded = True
                logger.info("CLIP model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load CLIP model: {str(e)}")
                raise

    async def _get_image_embedding(self, image_url: str) -> Optional[np.ndarray]:
        """Get CLIP embedding for an image URL"""
        try:
            logger.info(f"Starting image embedding generation for URL: {image_url}")
            
            # Ensure model is loaded
            await self.ensure_model_loaded()
            logger.info("Model ready for embedding generation")
            
            # Download image
            async with aiohttp.ClientSession() as session:
                try:
                    logger.info("Fetching image from URL")
                    async with session.get(image_url) as response:
                        if response.status != 200:
                            logger.error(f"Failed to fetch image. Status: {response.status}, URL: {image_url}")
                            raise ValueError(f"Failed to fetch image: HTTP {response.status}")
                        
                        image_data = await response.read()
                        logger.info(f"Image downloaded: {len(image_data)} bytes")
                        
                        # Validate image data
                        if len(image_data) == 0:
                            raise ValueError("Empty image data received")
                        
                        # Open and validate image
                        image = Image.open(io.BytesIO(image_data))
                        if image.mode != 'RGB':
                            logger.info(f"Converting image from {image.mode} to RGB")
                            image = image.convert('RGB')
                        
                        logger.info(f"Image opened successfully: {image.size}")
                        
                        # Preprocess image
                        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                        logger.info("Image preprocessed successfully")
                        
                        # Generate embedding
                        with torch.no_grad():
                            logger.info("Generating embedding")
                            embedding = self.model.encode_image(image_tensor)
                            embedding = embedding.cpu().numpy()
                            # Normalize
                            embedding = embedding / np.linalg.norm(embedding)
                            logger.info("Embedding generated and normalized")
                        
                        # Clear memory
                        del image_data
                        del image
                        del image_tensor
                        gc.collect()
                        
                        logger.info("Image embedding generation completed successfully")
                        return embedding
                        
                except aiohttp.ClientError as e:
                    logger.error(f"Network error fetching image: {str(e)}")
                    raise
                except PIL.UnidentifiedImageError as e:
                    logger.error(f"Invalid image format: {str(e)}")
                    raise ValueError("Invalid image format")
                except Exception as e:
                    logger.error(f"Error processing image: {str(e)}")
                    raise
                    
        except Exception as e:
            logger.error(f"Error generating embedding for {image_url}: {str(e)}")
            return None

    async def _extract_features(self, image_tensor):
        """Extract features from preprocessed image tensor"""
        try:
            with torch.no_grad():
                features = self.model.encode_image(image_tensor)
                features = features.cpu().numpy()
                # Clear memory
                del image_tensor
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
                
                # Normalize features
                features = features / np.linalg.norm(features, axis=1, keepdims=True)
                return features
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise

    async def build_index(self, products: List[Product]):
        try:
            logger.info(f"Starting to build index with {len(products)} products")
            
            # Ensure model is loaded
            await self.ensure_model_loaded()
            
            # Store products in memory for metadata lookup
            self.products = {p.id: p for p in products}
            
            # Process and upsert products in batches
            batch_size = 100
            for i in range(0, len(products), batch_size):
                batch = products[i:i + batch_size]
                vectors = []
                
                for product in batch:
                    embedding = await self.get_embedding_from_metadata(product)
                    vectors.append((
                        product.id,
                        embedding.tolist(),
                        {"metadata": product.metadata}
                    ))
                
                # Upsert to Pinecone
                self.index.upsert(vectors=vectors)
                
            logger.info("Index built successfully")
            return {"status": "success", "message": f"Built index with {len(products)} products"}
            
        except Exception as e:
            logger.error(f"Error building index: {str(e)}")
            raise

    async def _cleanup_memory(self):
        """Clean up GPU/CPU memory"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.debug("Memory cleanup completed")
        except Exception as e:
            logger.warning(f"Memory cleanup warning: {str(e)}")

    def validate_search_input(self, query: SearchQuery) -> None:
        """
        Validate search query inputs
        
        Args:
            query: SearchQuery object containing search parameters
            
        Raises:
            ValueError: If validation fails
        """
        # Check if at least one search method is provided
        if not any([query.text, query.image_url, getattr(query, 'image_file', None)]):
            raise ValueError("Must provide either text, image URL, or image file")
        
        # Check if multiple search methods are provided
        search_methods = sum([
            bool(query.text),
            bool(query.image_url),
            bool(getattr(query, 'image_file', None))
        ])
        if search_methods > 1:
            raise ValueError("Please provide only one search method")
        
        # Validate number of results
        if query.num_results < 1:
            raise ValueError("num_results must be greater than 0")
        
        logger.debug("Search input validation passed")

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Perform similarity search based on text, image URL, or uploaded image
        
        Args:
            query: SearchQuery object containing search parameters
            
        Returns:
            List[SearchResult]: List of search results
            
        Raises:
            ValueError: If input validation fails or embedding generation fails
        """
        try:
            logger.info("Starting search operation")
            
            # Validate inputs
            self.validate_search_input(query)
            
            # Ensure model is loaded
            await self.ensure_model_loaded()
            
            # Get embedding based on input type
            embedding = None
            
            if query.text:
                logger.info(f"Processing text search: {query.text[:50]}...")
                embedding = await self._get_text_embedding(query.text)
                
            elif query.image_url:
                logger.info(f"Processing image URL search: {query.image_url}")
                embedding = await self._get_image_embedding(query.image_url)
                
            elif hasattr(query, 'image_file') and query.image_file:
                logger.info("Processing uploaded image file")
                try:
                    # Read and process uploaded file
                    image = Image.open(io.BytesIO(query.image_file))
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Generate embedding
                    image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        embedding = self.model.encode_image(image_tensor)
                        embedding = embedding.cpu().numpy()
                        embedding = embedding / np.linalg.norm(embedding)
                    
                    # Cleanup
                    del image_tensor
                    await self._cleanup_memory()
                    
                except UnidentifiedImageError:
                    raise ValueError("Invalid image file format")
                except Exception as e:
                    raise ValueError(f"Error processing image file: {str(e)}")

            if embedding is None:
                raise ValueError("Failed to generate embedding")

            logger.info("Querying Pinecone index")
            results = self.index.query(
                vector=embedding.tolist(),
                top_k=query.num_results,
                include_metadata=True
            )

            # Process and validate results
            search_results = []
            for match in results.matches:
                try:
                    result = SearchResult(
                        id=str(match.id),
                        score=float(match.score),
                        metadata={
                            **match.metadata,
                            'image_url': match.metadata.get('image_url')
                        }
                    )
                    search_results.append(result)
                    logger.debug(f"Processed result: ID={match.id}, Score={match.score}")
                except Exception as e:
                    logger.error(f"Error processing result {match.id}: {str(e)}")
                    continue

            # Final cleanup
            await self._cleanup_memory()
            
            logger.info(f"Search completed. Found {len(search_results)} results")
            return search_results

        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            logger.exception("Full traceback:")
            raise
        finally:
            # Ensure memory is cleaned up even if an error occurs
            await self._cleanup_memory()

    async def cleanup(self):
        """Optional: Delete the index when needed"""
        try:
            if hasattr(self, 'index'):
                self.index.delete(delete_all=True)
            logger.info("Index cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error cleaning up index: {str(e)}")
            raise

    async def get_embedding_from_metadata(self, product: Product) -> np.ndarray:
        """
        Get CLIP embedding for a product using both image and metadata.
        Combines image and text embeddings for a richer representation.
        """
        try:
            # Get image embedding
            image_features = await self._get_image_embedding(product.image_url)
            
            # Get text embedding from metadata
            metadata_text = f"{product.metadata.get('name', '')} {product.metadata.get('description', '')} {product.metadata.get('category', '')}"
            with torch.no_grad():
                text = clip.tokenize([metadata_text]).to(self.device)
                text_features = self.model.encode_text(text)
                text_features = text_features.cpu().numpy()
                text_features = text_features / np.linalg.norm(text_features)
            
            # Combine embeddings (using average)
            combined_features = (image_features.squeeze() + text_features.squeeze()) / 2
            
            # Normalize the combined embedding
            combined_features = combined_features / np.linalg.norm(combined_features)
            
            return combined_features
                
        except Exception as e:
            logger.error(f"Error getting embedding for product {product.id}: {str(e)}")
            raise

    async def add_to_index(self, products: List[Dict[str, Any]]):
        try:
            for product in products:
                # Get image embedding
                image_embedding = await self._get_image_embedding(product['image_url'])
                
                if image_embedding is not None:
                    # Ensure embedding is properly formatted
                    vector_values = image_embedding.squeeze().tolist()
                    
                    # Prepare vector data with image_url in metadata
                    vector_data = [{
                        'id': product['id'],
                        'values': vector_values,
                        'metadata': {
                            **product['metadata'],
                            'image_url': product['image_url']  # Ensure image_url is in metadata
                        }
                    }]
                    
                    # Upsert to Pinecone
                    self.index.upsert(vectors=vector_data)
                    logger.info(f"Added product {product['id']} to index with image URL")
                else:
                    logger.warning(f"Skipping product {product['id']} - could not generate embedding")
                    
        except Exception as e:
            logger.error(f"Error adding products to index: {str(e)}")
            raise

    async def shutdown(self):
        """Graceful shutdown of the search engine"""
        try:
            if self.model_loaded:
                del self.model
                del self.preprocess
                self.model_loaded = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            gc.collect()
            logger.info("Search engine shut down successfully")
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            raise

    async def _get_text_embedding(self, text: str) -> np.ndarray:
        """Get CLIP embedding for text"""
        try:
            await self.ensure_model_loaded()
            with torch.no_grad():
                text_tokens = clip.tokenize([text]).to(self.device)
                embedding = self.model.encode_text(text_tokens)
                embedding = embedding.cpu().numpy()
                embedding = embedding / np.linalg.norm(embedding)
                return embedding
        except Exception as e:
            logger.error(f"Error generating text embedding: {str(e)}")
            raise

    async def check_health(self) -> Dict[str, Any]:
        """Check health status of search engine components"""
        return {
            "model_loaded": self.model_loaded,
            "index_connected": hasattr(self, 'index'),
            "device": str(self.device),
            "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else None
        }