from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.routes.api import router
from app.services.search_engine import ProductSearchEngine
import logging
from fastapi.background import BackgroundTasks
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
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
        torch.backends.cudnn.enabled = False  # Reduce memory usage
        torch.backends.cudnn.benchmark = False
        search_engine = ProductSearchEngine()
        
        # Run index loading in background
        background_tasks = BackgroundTasks()
        background_tasks.add_task(search_engine.load_index)
    except Exception as e:
        logger.error(f"Error initializing search engine: {str(e)}")

@app.get("/api/status")
async def get_status():
    """Check if the search engine is ready."""
    if not search_engine or not search_engine.index:
        raise HTTPException(status_code=503, detail="Search engine is initializing")
    return {"status": "ready"}

# Include the router
app.include_router(router, prefix="/api")