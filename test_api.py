import requests
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_api():
    try:
        base_url = "https://designview-staging-65571a6c93bd.herokuapp.com/api"
        
        # Check API health
        logger.info("Checking API health...")
        health_response = requests.get(f"{base_url}/health")
        logger.info(f"Health Status: {health_response.json()}")

        # Load and validate products
        with open('products.json', 'r', encoding='utf-8') as file:
            products = json.load(file)
            logger.info(f"Products validation passed. Found {len(products)} products")

        # Send request to build index
        logger.info("\nSending request to build index...")
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        response = requests.post(
            f"{base_url}/index/build",
            json=products,
            headers=headers
        )
        
        logger.info(f"Response Status: {response.status_code}")
        try:
            response_json = response.json()
            logger.info(f"Response JSON: {json.dumps(response_json, indent=2)}")
        except json.JSONDecodeError:
            logger.info(f"Response Text: {response.text}")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    test_api()