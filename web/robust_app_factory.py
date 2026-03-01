"""
Enhanced Flask Application Factory with Robustness Improvements
"""
import os
import logging
import time
from importlib import import_module
from datetime import datetime
from flask import Flask, request, g, jsonify, has_request_context
from config import active_config
from utils.database_pool import DatabaseManager
from utils.request_queue import RequestQueue
from utils.error_recovery import ErrorRecoveryManager
from utils.file_validator import FileValidator
from utils.streaming_file_processor import StreamingFileProcessor
import asyncio

logger = logging.getLogger(__name__)


def _load_optional_attr(module_name, attr_name):
    """Load optional attribute from module, logging when unavailable."""
    try:
        module = import_module(module_name)
        return getattr(module, attr_name)
    except (ImportError, AttributeError) as exc:
        logger.warning("Optional dependency unavailable: %s.%s (%s)", module_name, attr_name, exc)
        return None


DatabaseManager = _load_optional_attr('Utils.database_pool', 'DatabaseManager')
RequestQueue = _load_optional_attr('Utils.request_queue', 'RequestQueue')
ErrorRecoveryManager = _load_optional_attr('Utils.error_recovery', 'ErrorRecoveryManager')
StreamingFileProcessor = _load_optional_attr('Utils.streaming_file_processor', 'StreamingFileProcessor')

FileValidator = _load_optional_attr('Utils.file_validator', 'FileValidator')

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
    try:
        from pythonjsonlogger import jsonlogger
        formatter = jsonlogger.JsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s',
            timestamp=True
        )
    except ImportError:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        logger.warning("pythonjsonlogger unavailable; falling back to plain logging formatter")
    
    # JSON formatter for structured logs
    logHandler = logging.StreamHandler()
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
            record.request_id = getattr(g, 'request_id', 'no-request') if has_request_context() else 'no-request'
            return True
    
    for handler in logging.root.handlers:
        handler.addFilter(RequestIdFilter())

def initialize_robust_components(app, config):
    """Initialize robustness components"""
    global db_manager, request_queue, error_manager, file_validator, streaming_processor
    
    # Database connection pool
    if DatabaseManager:
        db_manager = DatabaseManager(
            db_type='sqlite',
            database_path=os.path.join(config.DATA_DIR, 'sum_knowledge.db'),
            pool_size=10,
            max_overflow=20
        )
    else:
        db_manager = None
    
    # Request queue system
    if RequestQueue:
        request_queue = RequestQueue(
            max_concurrent_requests=app.config['MAX_CONCURRENT_REQUESTS'],
            max_queue_size=200,
            memory_threshold_percent=app.config['MEMORY_THRESHOLD_PERCENT'],
            cpu_threshold_percent=app.config['CPU_THRESHOLD_PERCENT']
        )
    else:
        request_queue = None
    
    # Error recovery manager
    if ErrorRecoveryManager:
        error_manager = ErrorRecoveryManager()
    else:
        error_manager = None
    
    # File validator
    file_validator = FileValidator() if FileValidator else None
    
    # Streaming file processor
    if StreamingFileProcessor:
        streaming_processor = StreamingFileProcessor(
            temp_dir=os.path.join(config.DATA_DIR, 'temp')
        )
    else:
        streaming_processor = None
    
    # Store in app context
    app.db_manager = db_manager
    app.request_queue = request_queue
    app.error_manager = error_manager
    app.file_validator = file_validator
    app.streaming_processor = streaming_processor
    
    # Start request queue processing
    if request_queue:
        @app.before_request
        def start_queue():
            if not getattr(request_queue, 'is_running', False):
                try:
                    asyncio.run(request_queue.start())
                    logger.info("Request queue started")
                except Exception as exc:
                    logger.warning("Failed to start request queue: %s", exc)

def register_blueprints(app):
    """Register all application blueprints"""
    blueprint_specs = [
        ('api.summarization', 'summarization_bp', '/api', True),
        ('api.file_processing', 'file_processing_bp', '/api', True),
        ('api.health', 'health_bp', '/api', True),
        ('web.routes', 'web_bp', None, True),
        ('api.ai_models', 'ai_models_bp', '/api/ai', False),
        ('api.compression', 'compression_bp', '/api', False),
        ('api.collaborative_intelligence', 'collaborative_bp', '/api/collaborative', False),
        ('api.memory_api', 'memory_bp', '/api', False),
        ('api.streaming', 'streaming_bp', '/api', False),
        ('api.feedback_api', 'feedback_bp', '/api', False),
        ('api.async_file_processing', 'async_file_bp', '/api', False),
    ]

    for module_name, blueprint_name, prefix, is_core in blueprint_specs:
        blueprint = _load_optional_attr(module_name, blueprint_name)
        if blueprint is None:
            if is_core:
                logger.warning("Skipping core blueprint %s due to import failure", module_name)
            continue

        if prefix:
            app.register_blueprint(blueprint, url_prefix=prefix)
        else:
            app.register_blueprint(blueprint)

def register_enhanced_error_handlers(app):
    """Register enhanced error handlers with recovery"""
    from flask import jsonify
    from utils.error_handler import SUMException, ValidationError
    
    @app.errorhandler(404)
    def not_found(error):
        if error_manager:
            error_context = error_manager.track_error(
                error,
                {'path': request.path, 'method': request.method}
            )
            error_id = error_context.timestamp.timestamp()
        else:
            error_id = int(time.time())
        
        return jsonify({
            'error': True,
            'message': 'Resource not found',
            'path': request.path,
            'error_id': error_id
        }), 404
    
    @app.errorhandler(ValidationError)
    def validation_error(error):
        if error_manager:
            error_context = error_manager.track_error(
                error,
                {'endpoint': request.endpoint, 'data': request.get_json(silent=True)}
            )
            error_id = error_context.timestamp.timestamp()
        else:
            error_id = int(time.time())
        
        return jsonify({
            'error': True,
            'message': str(error),
            'field': error.details.get('field'),
            'error_id': error_id
        }), 400
    
    @app.errorhandler(500)
    def server_error(error):
        if error_manager:
            error_context = error_manager.track_error(
                error,
                {
                    'path': request.path,
                    'method': request.method,
                    'headers': dict(request.headers)
                }
            )
            error_id = error_context.timestamp.timestamp()
        else:
            error_id = int(time.time())
        
        # Check if we can recover
        recovery_func = error_manager.recovery_strategies.get(type(error)) if error_manager else None
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
                'error_id': error_id
            }), 500
        else:
            return jsonify({
                'error': True,
                'message': 'An error occurred processing your request',
                'error_id': error_id,
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
