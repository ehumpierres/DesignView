from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.routes.api import router
import gc
import torch
import logging
from app.services.search_engine import ProductSearchEngine
import os
import sys

logger = logging.getLogger(__name__)

# Core app setup
app = FastAPI(title="Product Search API")

# Static files handling - needed for frontend
app.mount("/frontend", StaticFiles(directory="app/frontend"), name="frontend")

# CORS middleware - important for API access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root route for serving frontend
@app.get("/")
async def root():
    """Serve index.html from frontend directory"""
    try:
        logger.info("Attempting to serve index.html")
        file_path = 'app/frontend/index.html'
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"index.html not found at path: {file_path}")
            raise HTTPException(status_code=404, detail="Frontend file not found")
            
        logger.info("Successfully located index.html")
        return FileResponse(file_path)
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Important startup event for initializing search engine
@app.on_event("startup")
async def startup_event():
    """Initialize search engine on startup"""
    try:
        search_engine = ProductSearchEngine()
        app.state.search_engine = search_engine
        logger.info("Search engine initialized in app state")
    except Exception as e:
        logger.error(f"Failed to initialize search engine: {str(e)}")
        raise

# Cleanup on shutdown - important for resource management
@app.on_event("shutdown")
async def shutdown_event():
    """Performs cleanup operations during application shutdown."""
    if hasattr(app.state, 'search_engine'):
        del app.state.search_engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Router inclusion for API endpoints
app.include_router(router, prefix="/api")

sys.setrecursionlimit(3000)  # Increase from default 1000