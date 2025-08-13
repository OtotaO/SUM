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
import time
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
    from api.memory_api import memory_bp
    from api.streaming import streaming_bp
    from api.feedback_api import feedback_bp
    from api.health import health_bp
    from api.async_file_processing import async_file_bp
    from web.routes import web_bp
    
    # Register with URL prefixes
    app.register_blueprint(summarization_bp, url_prefix='/api')
    app.register_blueprint(ai_models_bp, url_prefix='/api/ai')
    app.register_blueprint(compression_bp, url_prefix='/api')
    app.register_blueprint(file_processing_bp, url_prefix='/api')
    app.register_blueprint(collaborative_bp, url_prefix='/api/collaborative')
    app.register_blueprint(memory_bp, url_prefix='/api')
    app.register_blueprint(streaming_bp, url_prefix='/api')
    app.register_blueprint(feedback_bp, url_prefix='/api')
    app.register_blueprint(health_bp, url_prefix='/api')
    app.register_blueprint(async_file_bp, url_prefix='/api')
    app.register_blueprint(web_bp)


def register_error_handlers(app):
    """Register global error handlers."""
    from flask import jsonify, request
    from Utils.error_handler import SUMException, ValidationError, format_error_response, error_monitor
    
    @app.errorhandler(404)
    def not_found(error):
        error_monitor.record_error('404', str(error), {'path': request.path})
        return jsonify({
            'error': True,
            'message': 'Resource not found',
            'path': request.path
        }), 404
    
    @app.errorhandler(400)
    def bad_request(error):
        error_monitor.record_error('400', str(error), {'path': request.path})
        return jsonify({
            'error': True,
            'message': 'Bad request',
            'details': str(error)
        }), 400
    
    @app.errorhandler(ValidationError)
    def validation_error(error):
        error_monitor.record_error('validation', str(error), {'field': error.details.get('field')})
        return jsonify(format_error_response(error)), 400
    
    @app.errorhandler(SUMException)
    def sum_exception(error):
        error_monitor.record_error(error.code, str(error), error.details)
        return jsonify(format_error_response(error)), 400
    
    @app.errorhandler(500)
    def server_error(error):
        logger = logging.getLogger(__name__)
        logger.error(f"Server error: {error}", exc_info=True)
        error_monitor.record_error('500', str(error), {'path': request.path})
        
        # Don't expose internal errors in production
        if app.config.get('DEBUG'):
            return jsonify({
                'error': True,
                'message': 'Internal server error',
                'details': str(error)
            }), 500
        else:
            return jsonify({
                'error': True,
                'message': 'Internal server error',
                'error_id': f"ERR_{int(time.time())}"
            }), 500
    
    @app.errorhandler(Exception)
    def unhandled_exception(error):
        logger = logging.getLogger(__name__)
        logger.error(f"Unhandled exception: {error}", exc_info=True)
        error_monitor.record_error('unhandled', str(error), {'type': type(error).__name__})
        
        return jsonify({
            'error': True,
            'message': 'An unexpected error occurred',
            'error_id': f"ERR_{int(time.time())}"
        }), 500


def initialize_extensions(app):
    """Initialize Flask extensions."""
    # Add any Flask extensions here (e.g., Flask-CORS, Flask-SQLAlchemy)
    
    # Initialize circuit breakers for API resilience
    try:
        from api.circuit_breaker_integration import initialize_circuit_breakers
        initialize_circuit_breakers(app)
        logging.getLogger(__name__).info("Circuit breakers initialized")
    except Exception as e:
        logging.getLogger(__name__).warning(f"Circuit breakers not initialized: {e}")
    
    # Add security headers to all responses
    @app.after_request
    def add_security_headers(response):
        """Add security headers to every response."""
        # Prevent MIME type sniffing
        response.headers['X-Content-Type-Options'] = 'nosniff'
        
        # Prevent clickjacking
        response.headers['X-Frame-Options'] = 'DENY'
        
        # Enable browser XSS protection
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        # Force HTTPS (only in production)
        if not app.debug:
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        
        # Content Security Policy
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
            "font-src 'self' data: https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
            "img-src 'self' data: https:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        response.headers['Content-Security-Policy'] = csp
        
        # Referrer Policy
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        # Permissions Policy (formerly Feature Policy)
        response.headers['Permissions-Policy'] = (
            "geolocation=(), "
            "microphone=(), "
            "camera=(), "
            "payment=(), "
            "usb=(), "
            "magnetometer=(), "
            "accelerometer=()"
        )
        
        return response
    
    pass