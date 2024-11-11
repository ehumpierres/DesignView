from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.api import router
from app.services.search_engine import ProductSearchEngine
import logging

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
search_engine = ProductSearchEngine()

@app.on_event("startup")
async def startup_event():
    """Load index on startup."""
    try:
        await search_engine.load_index()
    except Exception as e:
        logger.error(f"Error loading index: {str(e)}")

# Include the router
app.include_router(router, prefix="/api")