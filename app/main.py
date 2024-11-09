import torch
import clip
import faiss
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Union
from pydantic import BaseModel, HttpUrl
import json
import io
import boto3
import pickle
from botocore.exceptions import ClientError
from botocore.config import Config
import os
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import logging
import time
from datetime import datetime
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class ProductMetadata(BaseModel):
    name: str
    description: str
    specifications: Dict
    category: Optional[str]

class Product(BaseModel):
    id: str
    metadata: ProductMetadata
    image_url: HttpUrl

class SearchQuery(BaseModel):
    image_url: Optional[HttpUrl]
    text_query: Optional[str]
    num_results: int = 5

class SearchResult(BaseModel):
    product_id: str
    metadata: ProductMetadata
    image_url: HttpUrl
    similarity_score: float

class S3Handler:
    def __init__(self):
        """Initialize S3 client with retry configuration."""
        config = Config(
            retries=dict(
                max_attempts=3,
                mode='adaptive'
            )
        )
        
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            config=config
        )
        self.bucket_name = os.environ.get('AWS_S3_BUCKET')

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def upload_file_object(self, file_object: bytes, s3_key: str) -> str:
        """Upload a file object to S3 with retry logic."""
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=file_object
            )
            return f"https://{self.bucket_name}.s3.amazonaws.com/{s3_key}"
        except ClientError as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def download_file_object(self, s3_key: str) -> bytes:
        """Download a file object from S3 with retry logic."""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            return response['Body'].read()
        except ClientError as e:
            logger.error(f"Error downloading from S3: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_image(self, s3_url: str) -> Image.Image:
        """Get image from S3 URL with retry logic."""
        try:
            parsed_url = urlparse(s3_url)
            key = parsed_url.path.lstrip('/')
            
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=key
            )
            image_data = response['Body'].read()
            
            return Image.open(io.BytesIO(image_data)).convert("RGB")
        except ClientError as e:
            logger.error(f"Error downloading image from S3: {str(e)}")
            raise

class ProductSearchEngine:
    def __init__(self, model_name: str = "ViT-B/32"):
        """Initialize the search engine."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.index = None
        self.product_mapping = {}
        self.s3_handler = S3Handler()
        self.index_key = "faiss_index/product_search_index.pkl"
        self.mapping_key = "faiss_index/product_mapping.json"

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

# FastAPI application
app = FastAPI(title="Product Search API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize search engine
search_engine = ProductSearchEngine()

@app.on_event("startup")
async def startup_event():
    """Load index on startup."""
    try:
        await search_engine.load_index()
    except Exception as e:
        logger.error(f"Error loading index: {str(e)}")

@app.post("/api/search", response_model=List[SearchResult])
async def search_products(query: SearchQuery):
    """Search for products by image or text."""
    try:
        results = await search_engine.search(query)
        return results
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/index/build")
async def build_index(products: List[Product], background_tasks: BackgroundTasks):
    """Build search index from product list."""
    try:
        background_tasks.add_task(search_engine.build_index, products)
        return {"message": "Index building started in background"}
    except Exception as e:
        logger.error(f"Index building error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "index_loaded": search_engine.index is not None
    }

# Example usage
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)