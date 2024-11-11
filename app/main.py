from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.routes.api import router
import gc
import torch
import logging
from app.services.search_engine import ProductSearchEngine

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

@app.on_event("startup")
async def startup_event():
    """Initializes application components during startup."""
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Create single instance and store in app.state
        app.state.search_engine = ProductSearchEngine()
        logger.info("Search engine initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing search engine: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Performs cleanup operations during application shutdown."""
    if hasattr(app.state, 'search_engine'):
        del app.state.search_engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Include the router
app.include_router(router, prefix="/api")