from flask import Flask
from flask_cors import CORS
from config import config

def create_app(config_name='default'):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Initialize CORS
    CORS(app)
    
    # Register blueprints
    from app.routes.api import api
    app.register_blueprint(api, url_prefix='/api')
    
    return app