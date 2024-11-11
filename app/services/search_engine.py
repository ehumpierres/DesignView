from typing import Union, List, Optional
from PIL import Image
import torch
import clip
import faiss
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

logger = logging.getLogger(__name__)

class ProductSearchEngine:
    """
    Singleton class handling product search using CLIP embeddings and FAISS index.
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
            self.device = "cpu"
            torch.set_num_threads(2)  # Reduce thread count
            self.model = None
            self.preprocess = None
            self.index = None
            self.product_mapping = {}
            self.s3_handler = S3Handler()
            self.index_key = "faiss_index/product_search_index.pkl"
            self.mapping_key = "faiss_index/product_mapping.json"
            self.initialized = True

    async def ensure_model_loaded(self):
        """Ensure CLIP model is loaded"""
        if not self.model_loaded:
            logger.info("Loading CLIP model...")
            try:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {self.device}")
                
                self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
                self.model.eval()  # Set to evaluation mode
                self.model_loaded = True
                logger.info("CLIP model loaded successfully")
            except Exception as e:
                self.model_loaded = False
                logger.error(f"Failed to load CLIP model: {str(e)}")
                raise Exception(f"Failed to load CLIP model: {str(e)}")
        return self.model_loaded

    async def _load_and_process_image(self, image_url: str) -> torch.Tensor:
        """Load and process image from URL"""
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            processed_image = self.preprocess(image).unsqueeze(0).to(self.device)
            return processed_image
        except Exception as e:
            logger.error(f"Error processing image from {image_url}: {str(e)}")
            raise

    async def _extract_features(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Extract features from processed image"""
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features = features.cpu().numpy()
            # Normalize features
            features = features / np.linalg.norm(features, axis=1, keepdims=True)
            return features

    async def build_index(self, products: List[Product]):
        """Build search index from products"""
        if not self.model_loaded:
            raise Exception("CLIP model not loaded. Call ensure_model_loaded() first")

        try:
            logger.info(f"Processing {len(products)} products")
            
            # Initialize feature matrix
            features_list = []
            valid_products = []

            # Process each product
            for product in products:
                try:
                    logger.info(f"Processing product {product.id}")
                    
                    # Load and process image
                    image_tensor = await self._load_and_process_image(str(product.image_url))
                    
                    # Extract features
                    features = await self._extract_features(image_tensor)
                    
                    features_list.append(features[0])  # features[0] because we processed single image
                    valid_products.append(product)
                    
                    logger.info(f"Successfully processed product {product.id}")
                except Exception as e:
                    logger.error(f"Failed to process product {product.id}: {str(e)}")
                    continue

            if not valid_products:
                raise Exception("No valid products to build index")

            # Build the index
            self.index = np.vstack(features_list)
            
            # Update product mapping
            self.product_mapping = {i: product for i, product in enumerate(valid_products)}
            
            self.index_loaded = True
            logger.info(f"Index built successfully with {len(valid_products)} products")
            return True

        except Exception as e:
            self.index_loaded = False
            logger.error(f"Failed to build index: {str(e)}")
            raise Exception(f"Failed to build index: {str(e)}")

    async def search(self, query_image_url: str = None, query_text: str = None, num_results: int = 5):
        """Search for similar products using image URL or text"""
        if not self.model_loaded or not self.index_loaded:
            raise Exception("Model or index not loaded")

        try:
            if query_image_url:
                # Process query image
                image_tensor = await self._load_and_process_image(query_image_url)
                query_features = await self._extract_features(image_tensor)
            elif query_text:
                # Process text query
                text = clip.tokenize([query_text]).to(self.device)
                with torch.no_grad():
                    query_features = self.model.encode_text(text)
                    query_features = query_features.cpu().numpy()
                    query_features = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
            else:
                raise ValueError("Either query_image_url or query_text must be provided")

            # Calculate similarities
            similarities = np.dot(self.index, query_features.T).squeeze()
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:num_results]
            
            results = []
            for idx in top_indices:
                product = self.product_mapping[idx]
                results.append({
                    "product_id": product.id,
                    "metadata": product.metadata,
                    "image_url": product.image_url,
                    "similarity_score": float(similarities[idx])
                })

            return results

        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise Exception(f"Search error: {str(e)}")