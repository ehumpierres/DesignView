from typing import Union, List, Optional
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
import pinecone
from typing import List, Dict, Any

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
            
            # Initialize Pinecone
            pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")  # Add environment
            )
            self.index = pinecone.Index(os.getenv("PINECONE_INDEX_NAME", "product-search"))

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

    async def _load_and_process_image(self, image_url: str):
        """Load and preprocess image from URL"""
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Clear memory
            del response
            del image
            gc.collect()
            
            return image_tensor
        except Exception as e:
            logger.error(f"Image loading failed for URL {image_url}: {str(e)}")
            raise

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

    async def search(self, query_image_url: Optional[str] = None, query_text: Optional[str] = None, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar products using either image or text query
        
        Args:
            query_image_url: Optional URL of query image
            query_text: Optional text query
            num_results: Number of results to return
            
        Returns:
            List of search results with scores and metadata
        """
        try:
            # Ensure model is loaded
            await self.ensure_model_loaded()
            
            # Get query embedding
            if query_image_url:
                # Process image query
                image_tensor = await self._load_and_process_image(query_image_url)
                query_embedding = await self._extract_features(image_tensor)
            elif query_text:
                # Process text query using CLIP's text encoder
                with torch.no_grad():
                    text = clip.tokenize([query_text]).to(self.device)
                    query_embedding = self.model.encode_text(text)
                    query_embedding = query_embedding.cpu().numpy()
                    query_embedding = query_embedding / np.linalg.norm(query_embedding)
            else:
                raise ValueError("Either image_url or text query must be provided")

            # Query Pinecone
            results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=num_results,
                include_metadata=True
            )
            
            # Format results
            search_results = []
            for match in results.matches:
                product_id = match.id
                if product_id in self.products:
                    product = self.products[product_id]
                    search_results.append({
                        "id": product_id,
                        "score": float(match.score),
                        "metadata": product.metadata,
                        "image_url": product.image_url
                    })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise

    async def cleanup(self):
        """Optional: Delete the index when needed"""
        try:
            pinecone.delete_index(self.index_name)
        except Exception as e:
            logger.error(f"Error cleaning up index: {str(e)}")

    async def get_embedding_from_metadata(self, product: Product) -> np.ndarray:
        """
        Get CLIP embedding for a product using both image and metadata.
        Combines image and text embeddings for a richer representation.
        """
        try:
            # Get image embedding
            image_tensor = await self._load_and_process_image(product.image_url)
            image_features = await self._extract_features(image_tensor)
            
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