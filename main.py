"""
main.py - SUM Knowledge Distillation Platform Entry Point

Clean, minimal entry point following John Carmack's principles:
- Single responsibility: Application startup
- Fast initialization with lazy loading
- Clear separation of concerns
- Minimal code for maximum clarity

The monolithic 1315-line main.py has been split into focused modules:
- api/: HTTP endpoints (summarization, ai_models, compression, file_processing)
- web/: Flask app factory and middleware
- application/: Service registry and dependency injection

Author: ototao
License: Apache License 2.0
"""

import time
import logging
from web.app_factory import create_app
from config import active_config


logger = logging.getLogger(__name__)


def main():
    """
    Main entry point for the SUM platform.
    
    Creates Flask app using modular components and starts the server.
    """
    # Create Flask app with modular architecture
    app = create_app()
    
    # Record start time for uptime calculations
    app.start_time = time.time()
    
    # Get configuration
    host = active_config.HOST
    port = active_config.PORT
    debug = active_config.DEBUG
    
    # Start server
    logger.info(f"Starting SUM server on {host}:{port} (debug={debug})...")
    logger.info("Modular architecture loaded: api/, web/, application/")
    
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    main()