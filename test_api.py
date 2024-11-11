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
        base_url = "https://designview-staging-65571a6c93bd.herokuapp.com/api"
        
        # Check API health
        logger.info("Checking API health...")
        health_response = requests.get(f"{base_url}/health")
        logger.info(f"Health Status: {health_response.json()}")

        # Load and validate products
        with open('products.json', 'r') as file:
            products_raw = json.load(file)
            products = [Product(**product).dict() for product in products_raw]
            logger.info(f"Products validation passed. Found {len(products)} products")

        # Send request to build index
        logger.info("\nSending request to build index...")
        response = requests.post(
            f"{base_url}/index/build",
            json=products,
            headers={'Content-Type': 'application/json'}
        )
        
        logger.info(f"Response Status: {response.status_code}")
        try:
            response_json = response.json()
            logger.info(f"Response JSON: {json.dumps(response_json, indent=2)}")
            if response.status_code != 200:
                logger.error(f"Error building index: {response_json.get('detail', 'Unknown error')}")
        except json.JSONDecodeError:
            logger.error(f"Failed to parse response: {response.text}")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    test_api()