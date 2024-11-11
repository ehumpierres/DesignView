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

# Initialize search engine
search_engine = ProductSearchEngine()

# Add the search_engine instance to app.state
app.state.search_engine = search_engine

@app.on_event("startup")
async def startup_event():
    """
    Initializes application components during startup.
    
    Performs:
        - Memory cleanup via garbage collection
        - CUDA cache clearing if available
        - Search engine initialization
        
    Raises:
        Logs error if search engine initialization fails
    """
    global search_engine
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        from app.services.search_engine import ProductSearchEngine
        search_engine = ProductSearchEngine()
        
    except Exception as e:
        logger.error(f"Error initializing search engine: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Performs cleanup operations during application shutdown.
    
    Performs:
        - Search engine deletion
        - Memory cleanup via garbage collection
        - CUDA cache clearing if available
    """
    global search_engine
    if search_engine:
        del search_engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@app.get("/api/status")
async def get_status():
    """
    Endpoint to check search engine readiness.
    
    Returns:
        dict: Status information with "ready" status
        
    Raises:
        HTTPException: 503 if search engine is not initialized
    """
    if not search_engine or not search_engine.index:
        raise HTTPException(status_code=503, detail="Search engine is initializing")
    return {"status": "ready"}

@app.get("/")
async def root():
    """
    Root endpoint providing API information.
    """
    return {
        "name": "DesignView AI Product Search API",
        "version": "1.0",
        "authors": "Ernesto Humpierres and Ryan Howard",
        "status": "running",
        "docs_url": "/docs"
    }

# Include the router
app.include_router(router, prefix="/api")