import requests
import json
import logging
from pydantic import BaseModel, HttpUrl
from typing import Dict, Optional, List
import time

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
        # First check API health
        base_url = "https://designview-staging-65571a6c93bd.herokuapp.com/api"
        
        logger.info("Checking API health...")
        health_response = requests.get(f"{base_url}/health")
        logger.info(f"Health Status: {health_response.json()}")

        # Load and validate products
        with open('products.json', 'r') as file:
            products_raw = json.load(file)
            products = [Product(**product).dict() for product in products_raw]
            logger.info("Products validation passed")

        # Try with single product first
        logger.info("\nTesting with single product...")
        single_product = [products[0]]
        
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        # Test image accessibility first
        logger.info(f"Testing image accessibility...")
        image_url = single_product[0]['image_url']
        img_response = requests.head(image_url)
        logger.info(f"Image response status: {img_response.status_code}")
        logger.info(f"Image headers: {dict(img_response.headers)}")

        # Send request to build index
        logger.info("\nSending request to build index...")
        response = requests.post(
            f"{base_url}/index/build",
            json=single_product,
            headers=headers
        )
        
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