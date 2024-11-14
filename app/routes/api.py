from fastapi import APIRouter, UploadFile, HTTPException, Request, File
from app.services.search_engine import ProductSearchEngine
from app.models.product import SearchQuery, Product
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from app.services.s3_handler import S3Handler
import logging
import requests
from fastapi.encoders import jsonable_encoder

logger = logging.getLogger(__name__)

router = APIRouter()

# Remove global search_engine instance
s3_handler = S3Handler()

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

        # Process products in batches to avoid memory issues
        batch_size = 100
        total_processed = 0
        
        for i in range(0, len(products), batch_size):
            batch = products[i:i + batch_size]
            vectors = []
            
            # Generate embeddings for batch
            for product in batch:
                try:
                    embedding = await search_engine.get_embedding_from_metadata(product)
                    vectors.append({
                        'id': product.id,
                        'values': embedding.tolist(),
                        'metadata': {
                            'name': product.metadata.get('name'),
                            'description': product.metadata.get('description'),
                            'category': product.metadata.get('category'),
                            'image_url': product.image_url
                        }
                    })
                except Exception as e:
                    logger.error(f"Error processing product {product.id}: {str(e)}")
                    continue
            
            # Upsert batch to Pinecone
            if vectors:
                search_engine.index.upsert(vectors=vectors)
                total_processed += len(vectors)
                logger.info(f"Processed and indexed batch of {len(vectors)} products. Total: {total_processed}")

        # Get final index stats
        stats = search_engine.index.describe_index_stats()
        logger.info(f"Index built successfully. Total vectors: {stats.total_vector_count}")
        
        return {
            "status": "success", 
            "message": f"Built index with {total_processed} products",
            "index_stats": stats
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

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

