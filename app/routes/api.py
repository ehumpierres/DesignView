from fastapi import APIRouter, UploadFile, HTTPException, Request
from app.services.search_engine import ProductSearchEngine
from app.models.product import SearchQuery, Product
import uuid
from pathlib import Path
from datetime import datetime
from typing import List
from app.services.s3_handler import S3Handler
import logging
import requests

logger = logging.getLogger(__name__)

router = APIRouter()

# Get the singleton instance instead of creating new one
search_engine = ProductSearchEngine()
s3_handler = S3Handler()

@router.post('/search')
async def search_products(query: SearchQuery):
    """
    Endpoint to search for similar products using image or text.
    
    Args:
        query (SearchQuery): Contains either image_url or text_query
        
    Returns:
        List[SearchResult]: Ranked list of similar products
        
    Raises:
        HTTPException: 500 if search operation fails
    """
    try:
        # Ensure model is loaded before search
        await search_engine.ensure_model_loaded()
        results = await search_engine.search(query)
        return results
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/index/build')
async def build_index(products: List[Product], request: Request):
    """
    Endpoint to build search index from product list.
    """
    try:
        logger.info(f"Received build index request with {len(products)} products")
        
        # Get search_engine instance from app state
        search_engine = request.app.state.search_engine
        
        # First ensure model is loaded
        logger.info("Loading CLIP model...")
        try:
            await search_engine.ensure_model_loaded()
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load CLIP model: {str(e)}")

        # Validate products
        if not products:
            raise HTTPException(status_code=400, detail="No products provided")

        # Build index
        try:
            logger.info(f"Building index with {len(products)} products")
            result = await search_engine.build_index(products)
            logger.info("Index built successfully")
            return {"status": "success", "message": f"Built index with {len(products)} products"}
        except Exception as e:
            logger.error(f"Error building index: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error building index: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.get('/health')
async def health_check(request: Request):
    """Check API health and model status"""
    search_engine = request.app.state.search_engine
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "model_loaded": search_engine.model_loaded,
        "index_loaded": search_engine.index_loaded
    }

@router.post("/upload-image")
async def upload_image(file: UploadFile):
    """
    Endpoint to upload an image to S3 storage.
    
    Args:
        file (UploadFile): Image file to be uploaded
        
    Returns:
        dict: Contains URL of uploaded image
        
    Raises:
        HTTPException: 500 if upload fails
    """
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

