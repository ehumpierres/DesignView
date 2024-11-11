from typing import Union, List
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

logger = logging.getLogger(__name__)

# Copy the entire ProductSearchEngine class from the original code
class ProductSearchEngine:
    def __init__(self, model_name: str = "ViT-B/32"):
        """Initialize the search engine."""
        self.device = "cpu"  # Force CPU usage
        torch.set_num_threads(4)  # Limit torch threads
        self.model, self.preprocess = clip.load(model_name, device=self.device, jit=True)
        self.index = None
        self.product_mapping = {}
        self.s3_handler = S3Handler()
        self.index_key = "faiss_index/product_search_index.pkl"
        self.mapping_key = "faiss_index/product_mapping.json"
        
        # Clear CUDA cache if it was used
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def save_index(self) -> bool:
        """Save FAISS index and product mapping to S3."""
        try:
            # Serialize FAISS index
            index_buffer = io.BytesIO()
            faiss.write_index(self.index, index_buffer)
            index_buffer.seek(0)
            
            # Save index
            await self.s3_handler.upload_file_object(
                index_buffer.getvalue(),
                self.index_key
            )
            
            # Save product mapping
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
        """Load FAISS index and product mapping from S3."""
        try:
            # Load index
            index_data = await self.s3_handler.download_file_object(self.index_key)
            index_buffer = io.BytesIO(index_data)
            self.index = faiss.read_index(index_buffer)
            
            # Load mapping
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
        """Extract CLIP features from an image or S3 URL."""
        try:
            if isinstance(image_or_url, str):
                image = await self.s3_handler.get_image(image_or_url)
            else:
                image = image_or_url

            if image is None:
                raise ValueError("Failed to load image")

            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
            return image_features.cpu().numpy()
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise

    async def build_index(self, products: List[Product]) -> bool:
        """Build FAISS index from product list."""
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
        
        # Save index to S3
        await self.save_index()
        
        logger.info(f"Index built with {len(valid_products)} products")
        return True

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search for similar products."""
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