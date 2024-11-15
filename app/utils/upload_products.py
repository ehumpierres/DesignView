# app/utils/upload_products.py
import requests
import json
from typing import List
from pathlib import Path

def upload_products(api_url: str, products_json_path: str):
    """
    Upload products to the search engine from a JSON file.
    
    JSON format should be:
    [
        {
            "id": "unique_id",
            "metadata": {
                "name": "Product Name",
                "description": "Product Description",
                "specifications": {
                    "key1": "value1",
                    "key2": "value2"
                },
                "category": "Category Name"
            },
            "image_url": "https://path-to-image.jpg"
        },
        ...
    ]
    """
    
    # Load products from JSON file
    with open(products_json_path, 'r') as f:
        products = json.load(f)
    
    # Send products to API
    response = requests.post(
        f"{api_url}/api/index/build",
        json=products
    )
    
    if response.status_code == 200:
        print("Successfully uploaded products")
        print(response.json())
    else:
        print(f"Error uploading products: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    # Example usage
    API_URL = "https://designview-staging-65571a6c93bd.herokuapp.com/"  # Change this to your deployed API URL
    PRODUCTS_JSON = "products.json"     # Path to your products JSON file
    
    upload_products(API_URL, PRODUCTS_JSON)
