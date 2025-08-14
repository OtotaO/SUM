"""
Production-Ready Entry Point for SUM Platform

Features:
- Production WSGI server (Gunicorn/uWSGI compatible)
- Graceful shutdown handling
- Health checks and monitoring
- Database connection pooling
- Request queue system
- Memory-efficient file processing
- Comprehensive error recovery
"""
import os
import sys
import logging
import signal
import atexit
from web.robust_app_factory import create_robust_app
from config import active_config

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create application instance
app = create_robust_app()

# Cleanup function for graceful shutdown
def cleanup():
    """Cleanup resources on shutdown"""
    logger.info("Starting cleanup process...")
    
    try:
        # Stop request queue
        if hasattr(app, 'request_queue') and app.request_queue:
            import asyncio
            asyncio.run(app.request_queue.stop())
            logger.info("Request queue stopped")
    except Exception as e:
        logger.error(f"Error stopping request queue: {e}")
    
    try:
        # Close database connections
        if hasattr(app, 'db_manager') and app.db_manager:
            app.db_manager.close()
            logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database: {e}")
    
    try:
        # Log final error statistics
        if hasattr(app, 'error_manager') and app.error_manager:
            stats = app.error_manager.get_error_stats()
            logger.info(f"Final error statistics: {stats}")
    except Exception as e:
        logger.error(f"Error getting error stats: {e}")
    
    logger.info("Cleanup complete")

# Register cleanup
atexit.register(cleanup)

# Signal handlers for graceful shutdown
def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}")
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Initialize async components
import asyncio

async def initialize_async_components():
    """Initialize async components like request queue"""
    if hasattr(app, 'request_queue') and app.request_queue:
        await app.request_queue.start()
        
        # Register handlers
        from api.robust_file_processing import file_analysis_handler
        app.request_queue.register_handler('file_analysis', file_analysis_handler)
        
        logger.info("Async components initialized")

# Run initialization
try:
    asyncio.run(initialize_async_components())
except Exception as e:
    logger.error(f"Failed to initialize async components: {e}")

# Production server entry point
if __name__ == '__main__':
    # Development mode - use Flask's built-in server
    if app.debug:
        logger.warning("Running in development mode. Use a production WSGI server for production!")
        app.run(
            host=active_config.HOST,
            port=active_config.PORT,
            debug=True,
            use_reloader=False  # Disable reloader to prevent duplicate initialization
        )
    else:
        # Production mode - let WSGI server handle it
        logger.info("Application ready for production WSGI server")
        logger.info(f"Configuration: {active_config.__class__.__name__}")
        
# For Gunicorn
application = app

# For uWSGI
wsgi_app = app