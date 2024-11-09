from flask import Blueprint, request, jsonify
from app.services.search_engine import ProductSearchEngine
from app.models.product import SearchQuery, Product

api = Blueprint('api', __name__)
search_engine = ProductSearchEngine()

@api.route('/search', methods=['POST'])
def search_products():
    """Search for products by image or text."""
    try:
        data = request.get_json()
        query = SearchQuery(
            image_url=data.get('image_url'),
            text_query=data.get('text_query'),
            num_results=data.get('num_results', 5)
        )
        results = search_engine.search(query)
        return jsonify([result.__dict__ for result in results])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/index/build', methods=['POST'])
def build_index():
    """Build search index from product list."""
    try:
        data = request.get_json()
        products = [Product(**product_data) for product_data in data]
        search_engine.build_index(products)
        return jsonify({'message': 'Index building started in background'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'index_loaded': search_engine.index is not None
    })