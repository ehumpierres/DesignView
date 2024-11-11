from typing import Dict, Optional
from pydantic import BaseModel, HttpUrl

class ProductMetadata(BaseModel):
    name: str
    description: str
    specifications: Dict
    category: Optional[str]

class Product(BaseModel):
    id: str
    metadata: ProductMetadata
    image_url: HttpUrl

class SearchQuery(BaseModel):
    image_url: Optional[HttpUrl]
    text_query: Optional[str]
    num_results: int = 5

class SearchResult(BaseModel):
    product_id: str
    metadata: ProductMetadata
    image_url: HttpUrl
    similarity_score: float