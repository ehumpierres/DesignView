import requests
import json
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_api():
    try:
        base_url = "https://designview-staging-65571a6c93bd.herokuapp.com/api"
        
        # Check API health with error handling
        logger.info("Checking API health...")
        try:
            health_response = requests.get(f"{base_url}/health")
            health_response.raise_for_status()  # Raise an error for bad status codes
            logger.info(f"Raw health response: {health_response.text}")
            logger.info(f"Response headers: {dict(health_response.headers)}")
            health_data = health_response.json()
            logger.info(f"Health Status: {health_data}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check failed: {str(e)}")
            logger.error(f"Response status code: {health_response.status_code}")
            logger.error(f"Response text: {health_response.text}")
            sys.exit(1)

        # Rest of the code only executes if health check passes
        with open('products.json', 'r', encoding='utf-8') as file:
            products = json.load(file)
            logger.info(f"Products validation passed. Found {len(products)} products")

        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        logger.info("\nSending request to build index...")
        response = requests.post(
            f"{base_url}/api/index/build",
            json=products,
            headers=headers
        )
        
        logger.info(f"Response Status: {response.status_code}")
        logger.info(f"Response Headers: {dict(response.headers)}")
        logger.info(f"Raw Response: {response.text}")

        if response.text:  # Only try to parse JSON if there's content
            try:
                response_json = response.json()
                logger.info(f"Response JSON: {json.dumps(response_json, indent=2)}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {str(e)}")
                logger.error(f"Raw response text: {response.text}")
        else:
            logger.error("Empty response received")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    test_api()