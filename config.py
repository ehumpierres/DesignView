import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard-to-guess-string'
    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
    AWS_S3_BUCKET = os.environ.get('AWS_S3_BUCKET')
    
    # CLIP model settings
    CLIP_MODEL_NAME = "ViT-B/32"
    
    # Search settings
    DEFAULT_SEARCH_RESULTS = 5
    
    # S3 settings
    S3_INDEX_KEY = "faiss_index/product_search_index.pkl"
    S3_MAPPING_KEY = "faiss_index/product_mapping.json"

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}