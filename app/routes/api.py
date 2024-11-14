from fastapi import APIRouter, UploadFile, HTTPException, Request, File
from app.services.search_engine import ProductSearchEngine
from app.models.product import SearchQuery, Product, ProductMetadata
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from app.services.s3_handler import S3Handler
import logging
import requests
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, HttpUrl

logger = logging.getLogger(__name__)

router = APIRouter()

# Remove global search_engine instance
s3_handler = S3Handler()

class ProductInput(BaseModel):
    id: str
    image_url: HttpUrl
    metadata: ProductMetadata

@router.post('/search')
async def search_products(query: SearchQuery, request: Request):
    """
    Endpoint to search for similar products using image or text.
    
    Args:
        query (SearchQuery): Contains either image_url or text_query
        request (Request): FastAPI request object
        
    Returns:
        List[SearchResult]: Ranked list of similar products
        
    Raises:
        HTTPException: 500 if search operation fails
    """
    try:
        # Get search_engine instance from app state
        search_engine = request.app.state.search_engine
        
        # Ensure model and index are loaded
        await search_engine.ensure_model_loaded()
        
        if not search_engine.index_loaded:
            raise HTTPException(
                status_code=400, 
                detail="Search index not built. Please build index first."
            )
            
        results = await search_engine.search(
            query_image_url=query.image_url,
            query_text=query.text,
            num_results=query.num_results or 5
        )
        return results
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/index/build")
async def build_index(products: List[ProductInput], request: Request):
    try:
        search_engine = request.app.state.search_engine
        
        if not search_engine:
            raise HTTPException(status_code=500, detail="Search engine not initialized")
        
        batch_size = 10
        results = []
        
        for i in range(0, len(products), batch_size):
            batch = products[i:i + batch_size]
            
            # Convert each product using flattened metadata
            batch_data = [{
                'id': str(p.id),
                'image_url': str(p.image_url),
                'metadata': p.metadata.flatten_metadata()  # Use new flatten method
            } for p in batch]
            
            try:
                await search_engine.add_to_index(batch_data)
                results.extend([p.id for p in batch])
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size}: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing batch {i//batch_size}: {str(e)}"
                )
        
        return {
            "status": "success",
            "processed_count": len(results),
            "processed_ids": results
        }
        
    except Exception as e:
        logger.error(f"Index build failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get("/health")
async def health_check(request: Request):
    try:
        search_engine = request.app.state.search_engine
        
        if not search_engine:
            return {
                "status": "unhealthy",
                "error": "Search engine not initialized"
            }
        
        # Simplify the response structure
        index_stats: Dict[str, Any] = {}
        try:
            if search_engine.index:
                # Get basic stats without nested objects
                stats = search_engine.index.describe_index_stats()
                index_stats = {
                    "dimension": stats.get("dimension", 0),
                    "total_vector_count": stats.get("total_vector_count", 0),
                    "namespaces": list(stats.get("namespaces", {}).keys())
                }
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            index_stats = {"error": str(e)}

        response = {
            "status": "healthy",
            "model_loaded": bool(search_engine.model_loaded),
            "index_stats": index_stats
        }
        
        # Use FastAPI's encoder with custom settings
        return jsonable_encoder(
            response,
            custom_encoder={
                datetime: str,
                bytes: lambda v: v.decode(),
            },
            exclude_none=True
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

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

