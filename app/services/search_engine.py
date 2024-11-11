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
        Only runs once due to singl`eton pattern.
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
                self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
                self.model_loaded = True
                logger.info(f"CLIP model loaded successfully on {self.device}")
            except Exception as e:
                logger.error(f"Error loading CLIP model: {str(e)}")
                raise

        return self.model_loaded

    async def save_index(self) -> bool:
        """
        Saves FAISS index and product mapping to S3 storage.
        
        Returns:
            bool: True if save was successful
            
        Raises:
            Exception: If serialization or upload fails
            
        Note:
            Saves both the FAISS index and product mapping as separate files
        """
        try:
            index_buffer = io.BytesIO()
            faiss.write_index(self.index, index_buffer)
            index_buffer.seek(0)
            
            await self.s3_handler.upload_file_object(
                index_buffer.getvalue(),
                self.index_key
            )
            
            mapping_json = json.dumps(self.product_mapping)
            await self.s3_handler.upload_file_object(
                mapping_json.encode('utf-8'),
                self.mapping_key
            )
            
            logger.info("Successfully saved index and mapping to S3")
            return True
        except Exception as e:
            logger.error(f"Error saving index to S3: {str(e)}")
            raise

    async def load_index(self) -> bool:
        """
        Loads FAISS index and product mapping from S3 storage.
        
        Returns:
            bool: True if load was successful
            
        Raises:
            ClientError: If S3 operations fail
            
        Note:
            Performs memory cleanup before loading new index
        """
        try:
            if self.index is not None:
                del self.index
                gc.collect()
            
            index_data = await self.s3_handler.download_file_object(self.index_key)
            index_buffer = io.BytesIO(index_data)
            self.index = faiss.read_index(index_buffer)
            
            mapping_data = await self.s3_handler.download_file_object(self.mapping_key)
            self.product_mapping = json.loads(mapping_data.decode('utf-8'))
            
            logger.info("Successfully loaded index and mapping from S3")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning("No existing index found in S3")
                return False
            raise

    async def _extract_image_features(self, image_or_url: Union[str, Image.Image]) -> np.ndarray:
        """
        Extracts CLIP features from an image or S3 URL.
        
        Args:
            image_or_url: Either a PIL Image object or S3 URL string
            
        Returns:
            np.ndarray: Normalized feature vector
            
        Raises:
            ValueError: If image loading fails
            Exception: If feature extraction fails
            
        Note:
            Includes memory optimization and cleanup after processing
        """
        try:
            await self.ensure_model_loaded()
            
            if isinstance(image_or_url, str):
                image = await self.s3_handler.get_image(image_or_url)
            else:
                image = image_or_url

            if image is None:
                raise ValueError("Failed to load image")

            image = image.convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device).half()
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
            del image_input
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
                
            return image_features.cpu().numpy()
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise

    async def build_index(self, products: List[Product]) -> bool:
        """
        Builds FAISS index from a list of products.
        
        Args:
            products: List of Product objects with image URLs
            
        Returns:
            bool: True if index was built successfully
            
        Raises:
            ValueError: If no valid products to index
            
        Note:
            Processes each product individually to handle failures gracefully
        """
        logger.info("Building product index...")
        
        features_list = []
        valid_products = []
        
        for idx, product in enumerate(products):
            try:
                features = await self._extract_image_features(product.image_url)
                if features is not None:
                    features_list.append(features.flatten())
                    valid_products.append(product)
                    self.product_mapping[str(idx)] = product.dict()
            except Exception as e:
                logger.error(f"Error processing product {product.id}: {str(e)}")
                continue

        if not features_list:
            raise ValueError("No valid products to build index")

        features_matrix = np.vstack(features_list)
        
        dimension = features_matrix.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(features_matrix)
        
        await self.save_index()
        
        logger.info(f"Index built with {len(valid_products)} products")
        return True

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Searches for similar products using image or text query.
        
        Args:
            query: SearchQuery object containing either image_url or text_query
            
        Returns:
            List[SearchResult]: Ranked list of similar products
            
        Raises:
            ValueError: If index not initialized or invalid query
            
        Note:
            Supports both image-to-image and text-to-image search
        """
        if not self.index:
            raise ValueError("Index not initialized")

        if query.image_url:
            query_features = await self._extract_image_features(query.image_url)
        elif query.text_query:
            text_tokens = clip.tokenize([query.text_query]).to(self.device)
            with torch.no_grad():
                query_features = self.model.encode_text(text_tokens)
                query_features /= query_features.norm(dim=-1, keepdim=True)
                query_features = query_features.cpu().numpy()
        else:
            raise ValueError("Either image_url or text_query must be provided")

        distances, indices = self.index.search(query_features, query.num_results)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            product_info = self.product_mapping[str(idx)]
            results.append(SearchResult(
                product_id=product_info['id'],
                metadata=product_info['metadata'],
                image_url=product_info['image_url'],
                similarity_score=float(distance)
            ))
            
        return results