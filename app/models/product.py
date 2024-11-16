from typing import Dict, Optional, Any, Union
from pydantic import BaseModel, Field, HttpUrl, validator, root_validator
import logging

logger = logging.getLogger(__name__)

class ProductMetadata(BaseModel):
    name: str
    description: str
    category: str
    
    def flatten_metadata(self) -> Dict[str, Any]:
        """Convert metadata to a flat dictionary with basic types"""
        return {
            "name": str(self.name),
            "description": str(self.description),
            "category": str(self.category)
        }

class Product(BaseModel):
    """Product model containing metadata and image URL"""
    id: str
    image_url: HttpUrl
    metadata: ProductMetadata
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            HttpUrl: str
        }

class SearchQuery(BaseModel):
    """Search query model supporting either image URL or text search"""
    text: Optional[str] = None
    image_url: Optional[HttpUrl] = None
    image_file: Optional[bytes] = None
    num_results: int = 5

    @validator('num_results')
    def validate_num_results(cls, v):
        if v < 1:
            raise ValueError("num_results must be greater than 0")
        return v

    @root_validator
    def validate_query(cls, values):
        text = values.get('text')
        image_url = values.get('image_url')
        image_file = values.get('image_file')
        
        # Log received values for debugging
        logger.debug(f"Validating query - text: {bool(text)}, image_url: {bool(image_url)}, image_file: {bool(image_file)}")
        
        if not any([text, image_url, image_file]):
            raise ValueError("Either text, image_url, or image_file must be provided")
        
        # Ensure only one search method is provided
        methods = sum(bool(x) for x in [text, image_url, image_file])
        if methods > 1:
            raise ValueError("Please provide only one search method")
            
        return values

class SearchResult(BaseModel):
    """Search result model containing product info and similarity score"""
    id: str
    score: float
    metadata: Dict[str, Any]
    
    def dict(self, *args, **kwargs):
        """Override dict method to ensure safe serialization"""
        return {
            "id": str(self.id),
            "score": float(self.score),
            "metadata": {
                **{k: str(v) if not isinstance(v, (str, int, float, bool, type(None)))
                   else v
                   for k, v in self.metadata.items()},
                "image_url": self.metadata.get("image_url", None)  # Ensure image_url is included
            }
        }

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            # Add any special type handling here
        }

class ProductInput(BaseModel):
    id: str
    image_url: str
    metadata: ProductMetadata

    def flatten_metadata(self) -> Dict[str, Any]:
        """Flatten metadata and ensure image_url is included"""
        flattened = self.metadata.dict()
        flattened['image_url'] = self.image_url  # Include image_url in metadata
        return flattened