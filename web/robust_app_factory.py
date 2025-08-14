"""
Enhanced Flask Application Factory with Robustness Improvements
"""
import os
import logging
import time
from flask import Flask, request, g
from config import active_config
from Utils.database_pool import DatabaseManager
from Utils.request_queue import RequestQueue
from Utils.error_recovery import ErrorRecoveryManager
from Utils.file_validator import FileValidator
from Utils.streaming_file_processor import StreamingFileProcessor
import asyncio

logger = logging.getLogger(__name__)

# Global instances for shared resources
db_manager = None
request_queue = None
error_manager = None
file_validator = None
streaming_processor = None

def create_robust_app(config=None):
    """
    Create Flask application with enhanced robustness features
    """
    app = Flask(__name__, 
                template_folder='../templates',
                static_folder='../static')
    
    # Use provided config or default to active_config
    config = config or active_config
    
    # Basic configuration
    app.config['SECRET_KEY'] = config.SECRET_KEY
    app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
    app.config['UPLOAD_FOLDER'] = config.UPLOADS_DIR
    app.config['ALLOWED_EXTENSIONS'] = config.ALLOWED_EXTENSIONS
    app.config['DEBUG'] = config.DEBUG
    app.config['TESTING'] = getattr(config, 'TESTING', False)
    
    # Robustness configuration
    app.config['REQUEST_TIMEOUT'] = 300  # 5 minutes
    app.config['MAX_CONCURRENT_REQUESTS'] = 50
    app.config['MEMORY_THRESHOLD_PERCENT'] = 80
    app.config['CPU_THRESHOLD_PERCENT'] = 85
    
    # Ensure directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(config.DATA_DIR, 'temp'), exist_ok=True)
    
    # Configure enhanced logging
    configure_enhanced_logging(config)
    
    # Initialize robust components
    initialize_robust_components(app, config)
    
    # Register blueprints
    register_blueprints(app)
    
    # Register enhanced error handlers
    register_enhanced_error_handlers(app)
    
    # Initialize extensions with robustness features
    initialize_robust_extensions(app)
    
    return app

def configure_enhanced_logging(config):
    """Configure enhanced logging with structured output"""
    import json
    from pythonjsonlogger import jsonlogger
    
    # JSON formatter for structured logs
    logHandler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s',
        timestamp=True
    )
    logHandler.setFormatter(formatter)
    
    # File handler with rotation
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        config.LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        handlers=[logHandler, file_handler]
    )
    
    # Add request ID to logs
    import uuid
    
    class RequestIdFilter(logging.Filter):
        def filter(self, record):
            record.request_id = getattr(g, 'request_id', 'no-request')
            return True
    
    for handler in logging.root.handlers:
        handler.addFilter(RequestIdFilter())

def initialize_robust_components(app, config):
    """Initialize robustness components"""
    global db_manager, request_queue, error_manager, file_validator, streaming_processor
    
    # Database connection pool
    db_manager = DatabaseManager(
        db_type='sqlite',
        database_path=os.path.join(config.DATA_DIR, 'sum_knowledge.db'),
        pool_size=10,
        max_overflow=20
    )
    
    # Request queue system
    request_queue = RequestQueue(
        max_concurrent_requests=app.config['MAX_CONCURRENT_REQUESTS'],
        max_queue_size=200,
        memory_threshold_percent=app.config['MEMORY_THRESHOLD_PERCENT'],
        cpu_threshold_percent=app.config['CPU_THRESHOLD_PERCENT']
    )
    
    # Error recovery manager
    error_manager = ErrorRecoveryManager()
    
    # File validator
    file_validator = FileValidator()
    
    # Streaming file processor
    streaming_processor = StreamingFileProcessor(
        temp_dir=os.path.join(config.DATA_DIR, 'temp')
    )
    
    # Store in app context
    app.db_manager = db_manager
    app.request_queue = request_queue
    app.error_manager = error_manager
    app.file_validator = file_validator
    app.streaming_processor = streaming_processor
    
    # Start request queue processing
    @app.before_first_request
    async def start_queue():
        if not request_queue.is_running:
            await request_queue.start()
            logger.info("Request queue started")

def register_blueprints(app):
    """Register all application blueprints"""
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

def register_enhanced_error_handlers(app):
    """Register enhanced error handlers with recovery"""
    from flask import jsonify
    from Utils.error_handler import SUMException, ValidationError
    
    @app.errorhandler(404)
    def not_found(error):
        error_context = error_manager.track_error(
            error, 
            {'path': request.path, 'method': request.method}
        )
        
        return jsonify({
            'error': True,
            'message': 'Resource not found',
            'path': request.path,
            'error_id': error_context.timestamp.timestamp()
        }), 404
    
    @app.errorhandler(ValidationError)
    def validation_error(error):
        error_context = error_manager.track_error(
            error,
            {'endpoint': request.endpoint, 'data': request.get_json()}
        )
        
        return jsonify({
            'error': True,
            'message': str(error),
            'field': error.details.get('field'),
            'error_id': error_context.timestamp.timestamp()
        }), 400
    
    @app.errorhandler(500)
    def server_error(error):
        error_context = error_manager.track_error(
            error,
            {
                'path': request.path,
                'method': request.method,
                'headers': dict(request.headers)
            }
        )
        
        # Check if we can recover
        recovery_func = error_manager.recovery_strategies.get(type(error))
        if recovery_func:
            try:
                recovery_result = asyncio.run(recovery_func(error, {}))
                if recovery_result.get('action') == 'retry':
                    # Log recovery attempt
                    logger.info(f"Attempting recovery for {type(error).__name__}")
            except:
                pass
        
        # Return user-friendly error
        if app.config.get('DEBUG'):
            return jsonify({
                'error': True,
                'message': 'Internal server error',
                'details': str(error),
                'error_id': error_context.timestamp.timestamp()
            }), 500
        else:
            return jsonify({
                'error': True,
                'message': 'An error occurred processing your request',
                'error_id': error_context.timestamp.timestamp(),
                'support': 'Please contact support with this error ID'
            }), 500

def initialize_robust_extensions(app):
    """Initialize Flask extensions with robustness features"""
    import signal
    from werkzeug.exceptions import RequestEntityTooLarge
    
    # Request timeout handling
    @app.before_request
    def before_request():
        # Generate request ID
        g.request_id = request.headers.get('X-Request-ID', str(int(time.time() * 1000)))
        g.start_time = time.time()
        
        # Check content length before processing
        if request.content_length and request.content_length > app.config['MAX_CONTENT_LENGTH']:
            raise RequestEntityTooLarge(f"Request too large: {request.content_length} bytes")
    
    @app.after_request
    def after_request(response):
        # Add request ID to response
        response.headers['X-Request-ID'] = g.get('request_id', 'unknown')
        
        # Add timing header
        if hasattr(g, 'start_time'):
            duration = (time.time() - g.start_time) * 1000
            response.headers['X-Response-Time'] = f"{duration:.2f}ms"
        
        # Add server timing
        response.headers['Server-Timing'] = f"total;dur={duration:.2f}" if 'duration' in locals() else "total;dur=0"
        
        # Add robustness headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        # Add cache control for API responses
        if request.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
        
        return response
    
    # Graceful shutdown handling
    def graceful_shutdown(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        
        # Stop accepting new requests
        if request_queue:
            asyncio.run(request_queue.stop())
        
        # Close database connections
        if db_manager:
            db_manager.close()
        
        # Log final statistics
        if error_manager:
            stats = error_manager.get_error_stats()
            logger.info(f"Error statistics: {stats}")
        
        logger.info("Graceful shutdown complete")
        exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)
    
    # Health check middleware
    @app.route('/health/ready')
    def readiness_check():
        """Kubernetes readiness probe"""
        checks = {
            'database': check_database_health(),
            'queue': check_queue_health(),
            'memory': check_memory_health()
        }
        
        all_healthy = all(checks.values())
        status_code = 200 if all_healthy else 503
        
        return jsonify({
            'status': 'ready' if all_healthy else 'not ready',
            'checks': checks,
            'timestamp': datetime.utcnow().isoformat()
        }), status_code
    
    @app.route('/health/live')
    def liveness_check():
        """Kubernetes liveness probe"""
        return jsonify({
            'status': 'alive',
            'timestamp': datetime.utcnow().isoformat()
        }), 200

def check_database_health():
    """Check database connection health"""
    try:
        with db_manager.get_connection() as conn:
            conn.execute("SELECT 1")
        return True
    except:
        return False

def check_queue_health():
    """Check request queue health"""
    try:
        metrics = request_queue.get_metrics()
        return metrics['active_requests'] < request_queue.max_concurrent * 0.9
    except:
        return False

def check_memory_health():
    """Check system memory health"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return memory.percent < 90
    except:
        return True  # Assume healthy if can't check