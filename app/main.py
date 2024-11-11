from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import gc
import torch
import logging

logger = logging.getLogger(__name__)

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
search_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize search engine in the background."""
    global search_engine
    try:
        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        from app.services.search_engine import ProductSearchEngine
        search_engine = ProductSearchEngine()
        
    except Exception as e:
        logger.error(f"Error initializing search engine: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global search_engine
    if search_engine:
        del search_engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@app.get("/api/status")
async def get_status():
    """Check if the search engine is ready."""
    if not search_engine or not search_engine.index:
        raise HTTPException(status_code=503, detail="Search engine is initializing")
    return {"status": "ready"}

# Include the router
app.include_router(router, prefix="/api")