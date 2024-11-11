import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_api():
    try:
        with open('products.json', 'r') as file:
            products = json.load(file)
            
        # Print the exact payload being sent
        logger.info("Request payload:")
        logger.info(json.dumps(products, indent=2))

        api_url = "https://designview-staging-65571a6c93bd.herokuapp.com/api/index/build"
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        # Try with single product first
        logger.info("\nTesting with single product...")
        single_product = [products[0]]
        response = requests.post(api_url, json=single_product, headers=headers)
        logger.info(f"Request payload: {json.dumps(single_product, indent=2)}")
        logger.info(f"Response Status: {response.status_code}")
        logger.info(f"Response Headers: {dict(response.headers)}")
        logger.info(f"Response Body: {response.text}")

    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    test_api()