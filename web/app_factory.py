"""
app_factory.py - Flask Application Factory

Clean Flask app initialization following John Carmack's principles:
- Single responsibility: Create and configure Flask app
- Fast initialization with lazy loading
- Clean dependency injection setup
- Minimal external dependencies

Author: ototao
License: Apache License 2.0
"""

import os
import logging
from flask import Flask
from config import active_config


def create_app(config=None):
    """
    Create and configure Flask application.
    
    Args:
        config: Optional configuration object
        
    Returns:
        Flask: Configured Flask application
    """
    app = Flask(__name__, 
                template_folder='../templates',
                static_folder='../static')
    
    # Use provided config or default to active_config
    config = config or active_config
    
    # Configure application
    app.config['SECRET_KEY'] = config.SECRET_KEY
    app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
    app.config['UPLOAD_FOLDER'] = config.UPLOADS_DIR
    app.config['ALLOWED_EXTENSIONS'] = config.ALLOWED_EXTENSIONS
    app.config['DEBUG'] = config.DEBUG
    app.config['TESTING'] = getattr(config, 'TESTING', False)
    
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Configure logging
    configure_logging(config)
    
    # Register blueprints
    register_blueprints(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Initialize extensions
    initialize_extensions(app)
    
    return app


def configure_logging(config):
    """Configure application logging."""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config.LOG_FILE)
        ]
    )


def register_blueprints(app):
    """Register all application blueprints."""
    # Import blueprints here to avoid circular imports
    from api.summarization import summarization_bp
    from api.ai_models import ai_models_bp
    from api.compression import compression_bp
    from api.file_processing import file_processing_bp
    from api.collaborative_intelligence import collaborative_bp
    from web.routes import web_bp
    
    # Register with URL prefixes
    app.register_blueprint(summarization_bp, url_prefix='/api')
    app.register_blueprint(ai_models_bp, url_prefix='/api/ai')
    app.register_blueprint(compression_bp, url_prefix='/api')
    app.register_blueprint(file_processing_bp, url_prefix='/api')
    app.register_blueprint(collaborative_bp, url_prefix='/api/collaborative')
    app.register_blueprint(web_bp)


def register_error_handlers(app):
    """Register global error handlers."""
    @app.errorhandler(404)
    def not_found(error):
        from flask import jsonify
        return jsonify({'error': 'Not found'}), 404
    
    @app.errorhandler(500)
    def server_error(error):
        from flask import jsonify
        logger = logging.getLogger(__name__)
        logger.error(f"Server error: {error}")
        return jsonify({'error': 'Internal server error'}), 500


def initialize_extensions(app):
    """Initialize Flask extensions."""
    # Add any Flask extensions here (e.g., Flask-CORS, Flask-SQLAlchemy)
    pass