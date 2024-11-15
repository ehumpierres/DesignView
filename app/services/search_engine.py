from typing import Union, List, Optional, Dict, Any
from PIL import Image
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
            # Ensure model is loaded
            await self.ensure_model_loaded()
            
            # Download image
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status != 200:
                        raise ValueError(f"Failed to fetch image: {response.status}")
                    
                    image_data = await response.read()
                    image = Image.open(io.BytesIO(image_data)).convert('RGB')
                    
                    # Preprocess image
                    image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                    
                    # Generate embedding
                    with torch.no_grad():
                        embedding = self.model.encode_image(image_tensor)
                        embedding = embedding.cpu().numpy()
                        # Normalize
                        embedding = embedding / np.linalg.norm(embedding)
                    
                    # Clear memory
                    del image_data
                    del image
                    del image_tensor
                    gc.collect()
                    
                    return embedding
                    
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

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        try:
            await self.ensure_model_loaded()
            
            if query.text:
                embedding = await self._get_text_embedding(query.text)
            elif query.image_url:
                embedding = await self._get_image_embedding(query.image_url)
            else:
                raise ValueError("Either text or image_url must be provided")

            results = self.index.query(
                vector=embedding.tolist(),
                top_k=query.num_results,
                include_metadata=True
            )
            
            # Clean and validate results before creating SearchResult objects
            search_results = [
                SearchResult(
                    id=str(match.id),
                    score=float(match.score),
                    metadata={
                        **match.metadata,
                        'image_url': match.metadata.get('image_url')
                    }
                )
                for match in results.matches
            ]
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise

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