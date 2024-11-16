from fastapi import APIRouter, UploadFile, HTTPException, Request, File, Form
from app.services.search_engine import ProductSearchEngine
from app.models.product import SearchQuery, Product, ProductMetadata
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from app.services.s3_handler import S3Handler
import logging
import requests
from pydantic import BaseModel, HttpUrl
from app.utils.encoders import safe_json_encode
from fastapi.responses import JSONResponse
from fastapi import status

logger = logging.getLogger(__name__)

router = APIRouter()

# Remove global search_engine instance
s3_handler = S3Handler()

class ProductInput(BaseModel):
    id: str
    image_url: HttpUrl
    metadata: ProductMetadata

@router.post("/api/search")
async def search(
    text: Optional[str] = Form(None),
    image_url: Optional[str] = Form(None),
    image_file: Optional[UploadFile] = File(None),
    num_results: int = Form(5)
):
    try:
        # Log received data
        logger.debug(f"Received search request - text: {bool(text)}, image_url: {bool(image_url)}, image_file: {bool(image_file)}")
        
        # Convert UploadFile to bytes if present
        image_file_bytes = None
        if image_file:
            image_file_bytes = await image_file.read()
        
        # Create SearchQuery object
        query = SearchQuery(
            text=text,
            image_url=image_url,
            image_file=image_file_bytes,
            num_results=num_results
        )
        
        # Get search engine instance
        search_engine = ProductSearchEngine()
        
        # Perform search
        results = await search_engine.search(query)
        
        return results
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/index/build")
async def build_index(products: List[ProductInput], request: Request):
    try:
        search_engine = request.app.state.search_engine
        if not search_engine:
            raise HTTPException(status_code=500, detail="Search engine not initialized")
        
        # Convert products with image URLs included in metadata
        batch_data = [{
            'id': str(p.id),
            'image_url': str(p.image_url),
            'metadata': {
                **p.metadata.dict(),
                'image_url': str(p.image_url)  # Explicitly include image_url in metadata
            }
        } for p in products]
        
        await search_engine.add_to_index(batch_data)
        
        return {
            "status": "success",
            "processed_count": len(products),
            "processed_ids": [p.id for p in products]
        }
        
    except Exception as e:
        logger.error(f"Index build failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check(request: Request):
    try:
        search_engine = request.app.state.search_engine
        if not search_engine:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "unhealthy", "error": "Search engine not initialized"}
            )
            
        health_status = await search_engine.check_health()
        return {
            "status": "healthy",
            "details": health_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )

@router.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    Endpoint to upload an image to S3 storage.
    """
    try:
        # Check file size (5MB limit)
        contents = await file.read()
        if len(contents) > 5 * 1024 * 1024:  # 5MB
            raise HTTPException(
                status_code=400,
                detail="File size too large. Maximum size is 5MB"
            )
            
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
            
        filename = f"{uuid.uuid4()}{Path(file.filename).suffix}"
        
        logger.info(f"Saving to S3 as: uploads/{filename}")
        s3_url = await s3_handler.upload_file_object(
            contents,
            f"uploads/{filename}"
        )
        
        return {"url": s3_url}
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

