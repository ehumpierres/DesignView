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
    image_url: Optional[HttpUrl] = Field(
        default=None, 
        description="URL of the query image"
    )
    text: Optional[str] = Field(
        default=None, 
        description="Text query for semantic search"
    )
    num_results: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return (1-20)"
    )

    class Config:
        schema_extra = {
            "example": {
                "text": "modern black table lamp",
                "num_results": 5
            }
        }

    def validate_query(self):
        """Ensure either image_url or text is provided, but not both"""
        if self.image_url is None and self.text is None:
            raise ValueError("Either image_url or text must be provided")
        if self.image_url is not None and self.text is not None:
            raise ValueError("Only one of image_url or text should be provided")
        return True

class SearchResult(BaseModel):
    """Search result model containing product info and similarity score"""
    id: str
    metadata: Dict[str, Any]
    image_url: HttpUrl
    score: float = Field(ge=0.0, le=1.0)