from fastapi import APIRouter, UploadFile, HTTPException
from app.services.search_engine import ProductSearchEngine
from app.models.product import SearchQuery, Product
import uuid
from pathlib import Path
from datetime import datetime
from typing import List
from app.services.s3_handler import S3Handler
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Get the singleton instance instead of creating new one
search_engine = ProductSearchEngine()
s3_handler = S3Handler()

@router.post('/search')
async def search_products(query: SearchQuery):
    """Search for products by image or text."""
    try:
        # Ensure model is loaded before search
        await search_engine.ensure_model_loaded()
        results = await search_engine.search(query)
        return results
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/index/build')
async def build_index(products: List[Product]):
    """Build search index from product list."""
    try:
        await search_engine.build_index(products)
        return {'message': 'Index building started in background'}
    except Exception as e:
        logger.error(f"Index building error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/health')
async def health_check():
    """Health check endpoint."""
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'index_loaded': search_engine.index is not None,
        'model_loaded': search_engine.model is not None
    }

@router.post("/upload-image")
async def upload_image(file: UploadFile):
    """Upload image to S3 and return the URL"""
    try:
        filename = f"{uuid.uuid4()}{Path(file.filename).suffix}"
        content = await file.read()
        s3_url = await s3_handler.upload_file_object(
            content,
            f"uploads/{filename}"
        )
        return {"url": s3_url}
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
async def root():
    """Root endpoint that returns API information"""
    return {
        "name": "Product Search API",
        "version": "1.0",
        "status": "running",
        "docs_url": "/docs"  # FastAPI automatic documentation
    }