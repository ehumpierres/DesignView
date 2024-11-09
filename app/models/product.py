from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class ProductMetadata:
    name: str
    description: str
    specifications: Dict
    category: Optional[str] = None

@dataclass
class Product:
    id: str
    metadata: ProductMetadata
    image_url: str

@dataclass
class SearchQuery:
    image_url: Optional[str] = None
    text_query: Optional[str] = None
    num_results: int = 5

@dataclass
class SearchResult:
    product_id: str
    metadata: ProductMetadata
    image_url: str
    similarity_score: float