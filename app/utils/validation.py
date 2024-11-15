from typing import Optional
from pydantic import BaseModel, validator
from urllib.parse import urlparse

class SearchQueryValidator(BaseModel):
    text: Optional[str] = None
    image_url: Optional[str] = None
    num_results: int = 5

    @validator('text')
    def validate_text(cls, v):
        if v is not None and len(v.strip()) == 0:
            raise ValueError("Text query cannot be empty")
        return v

    @validator('image_url')
    def validate_image_url(cls, v):
        if v is not None:
            result = urlparse(v)
            if not all([result.scheme, result.netloc]):
                raise ValueError("Invalid image URL")
        return v

    @validator('num_results')
    def validate_num_results(cls, v):
        if v < 1 or v > 100:
            raise ValueError("num_results must be between 1 and 100")
        return v
