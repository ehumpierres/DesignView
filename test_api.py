import requests
import json
import logging
from pydantic import BaseModel, HttpUrl
from typing import Dict, Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the models to validate the data
class ProductMetadata(BaseModel):
    name: str
    description: str
    specifications: Dict
    category: Optional[str]

class Product(BaseModel):
    id: str
    metadata: ProductMetadata
    image_url: HttpUrl

def test_api():
    try:
        # Load and validate products
        with open('products.json', 'r') as file:
            products_raw = json.load(file)
            # Validate each product against the model
            products = [Product(**product).dict() for product in products_raw]
            logger.info("Products validation passed")

        api_url = "https://designview-staging-65571a6c93bd.herokuapp.com/api/index/build"
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        # Try with single product first
        logger.info("\nTesting with single product...")
        single_product = [products[0]]
        logger.info(f"Sending product: {json.dumps(single_product, indent=2)}")
        
        response = requests.post(api_url, json=single_product, headers=headers)
        logger.info(f"Response Status: {response.status_code}")
        
        try:
            response_json = response.json()
            logger.info(f"Response JSON: {json.dumps(response_json, indent=2)}")
        except:
            logger.info(f"Response Text: {response.text}")

    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    test_api()