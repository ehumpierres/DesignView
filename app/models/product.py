from typing import Dict, Optional, Any, Union
from pydantic import BaseModel, Field, HttpUrl

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

    @validator('text', 'image_url')
    def validate_query_inputs(cls, v):
        if v is not None:
            return str(v)
        return v

    def validate_query(self):
        if not self.text and not self.image_url:
            raise ValueError("Either text or image_url must be provided")

class SearchResult(BaseModel):
    """Search result model containing product info and similarity score"""
    id: str
    metadata: Dict[str, Any]
    image_url: Optional[HttpUrl] = None
    score: float = Field(ge=0.0, le=1.0)

    class Config:
        arbitrary_types_allowed = True