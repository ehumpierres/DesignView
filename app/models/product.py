from typing import Dict, Optional, Any, Union
from pydantic import BaseModel, Field, HttpUrl, validator

class ProductMetadata(BaseModel):
    name: str
    description: str
    specifications: Dict[str, str]
    category: Optional[str]

    def flatten_metadata(self) -> Dict[str, str]:
        """Convert nested specifications into flattened metadata"""
        flattened = {
            'name': self.name,
            'description': self.description,
        }
        if self.category:
            flattened['category'] = self.category
            
        # Add specifications with prefix
        for key, value in self.specifications.items():
            flattened[f'spec_{key}'] = str(value)
            
        return flattened

class Product(BaseModel):
    """Product model containing metadata and image URL"""
    id: str
    metadata: Dict[str, Any]
    image_url: HttpUrl

class SearchQuery(BaseModel):
    """Search query model supporting either image URL or text search"""
    text: Optional[str] = None
    image_url: Optional[str] = None
    num_results: int = Field(default=5, ge=1, le=100)

    @validator('text', 'image_url', pre=True)
    def validate_query_inputs(cls, v):
        if v is not None:
            if isinstance(v, str) and len(v.strip()) == 0:
                return None
            return str(v).strip()
        return v

    @validator('num_results')
    def validate_num_results(cls, v):
        try:
            v = int(v)
            if v < 1:
                raise ValueError("Number of results must be at least 1")
            if v > 100:
                raise ValueError("Number of results cannot exceed 100")
            return v
        except (TypeError, ValueError):
            raise ValueError("Number of results must be a valid integer")

    def validate_query(self):
        """Additional validation to ensure at least one search criterion is provided"""
        if not self.text and not self.image_url:
            raise ValueError("Either text or image_url must be provided")
        return True

class SearchResult(BaseModel):
    """Search result model containing product info and similarity score"""
    id: str
    metadata: Dict[str, Any]
    image_url: Optional[HttpUrl] = None
    score: float = Field(ge=0.0, le=1.0)

    class Config:
        arbitrary_types_allowed = True