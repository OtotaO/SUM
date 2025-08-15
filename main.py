"""
Simple main entry point without circular imports
"""
import time
import logging
from flask import Flask
from config import active_config

# Import only the working blueprints
from api.summarization import summarization_bp
from api.file_processing import file_processing_bp
from api.health import health_bp
from api.mass_processing import mass_bp
from web.routes import web_bp

logger = logging.getLogger(__name__)

def create_simple_app():
    """Create Flask app with only working components"""
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    
    # Configure application
    app.config['SECRET_KEY'] = active_config.SECRET_KEY
    app.config['MAX_CONTENT_LENGTH'] = active_config.MAX_CONTENT_LENGTH
    app.config['UPLOAD_FOLDER'] = active_config.UPLOADS_DIR
    app.config['ALLOWED_EXTENSIONS'] = active_config.ALLOWED_EXTENSIONS
    app.config['DEBUG'] = active_config.DEBUG
    
    # Register only working blueprints
    app.register_blueprint(summarization_bp, url_prefix='/api')
    app.register_blueprint(file_processing_bp, url_prefix='/api')
    app.register_blueprint(health_bp, url_prefix='/api')
    app.register_blueprint(mass_bp)  # Mass processing at root level
    app.register_blueprint(web_bp)
    
    # Add basic error handlers
    @app.errorhandler(404)
    def not_found(error):
        return {'error': 'Not found'}, 404
    
    @app.errorhandler(500)
    def server_error(error):
        return {'error': 'Internal server error'}, 500
    
    return app

def main():
    """Main entry point"""
    # Check dependencies first
    print("\n" + "="*50)
    print("🚀 SUM Platform - Starting Up")
    print("="*50)
    
    try:
        from check_dependencies import check_dependencies, check_feature_availability
        deps = check_dependencies()
        features = check_feature_availability(deps)
        
        available = sum(1 for v in features.values() if v)
        total = len(features)
        print(f"\n✅ Features Available: {available}/{total}")
        
        if not features["Semantic Memory"]:
            print("⚠️  Semantic Memory: Using basic fallback (install ChromaDB/FAISS for full features)")
        if not features["Knowledge Graphs"]:
            print("⚠️  Knowledge Graphs: Using in-memory fallback (install Neo4j for persistence)")
            
    except Exception as e:
        print(f"⚠️  Could not check dependencies: {e}")
    
    print("\n" + "="*50 + "\n")
    
    app = create_simple_app()
    app.start_time = time.time()
    
    host = active_config.HOST
    port = active_config.PORT
    debug = active_config.DEBUG
    
    logger.info(f"Starting SUM server (simplified) on {host}:{port}")
    logger.info("Note: Using simplified startup to avoid circular imports")
    logger.info(f"Web interface available at http://{host}:{port}")
    
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    main()