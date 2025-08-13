"""
circuit_breaker_integration.py - Circuit Breaker Integration for API Endpoints

Demonstrates how to integrate circuit breakers into Flask API endpoints
for improved resilience and fault tolerance.

Author: SUM Development Team
License: Apache License 2.0
"""

from flask import jsonify
from functools import wraps
import logging

from Utils.circuit_breaker import circuit_breaker, CircuitOpenError, circuit_breaker_manager

logger = logging.getLogger(__name__)


def api_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    fallback_response: dict = None
):
    """
    Decorator to apply circuit breaker pattern to API endpoints.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery
        fallback_response: Response to return when circuit is open
        
    Example:
        @app.route('/api/endpoint')
        @api_circuit_breaker(failure_threshold=3, recovery_timeout=30)
        def api_endpoint():
            # API logic that might fail
            pass
    """
    if fallback_response is None:
        fallback_response = {
            'error': 'Service temporarily unavailable',
            'message': 'Please try again later',
            'status': 'circuit_open'
        }
    
    def decorator(func):
        # Create circuit breaker for this endpoint
        breaker_name = f"api_{func.__name__}"
        
        @wraps(func)
        @circuit_breaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=Exception
        )
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except CircuitOpenError:
                logger.warning(f"Circuit breaker open for {breaker_name}")
                return jsonify(fallback_response), 503
            except Exception as e:
                logger.error(f"Error in {breaker_name}: {e}")
                raise
        
        # Register with manager for monitoring
        if hasattr(wrapper, 'circuit_breaker'):
            circuit_breaker_manager.register(wrapper.circuit_breaker)
        
        return wrapper
    
    return decorator


# Example integrations for critical endpoints

def integrate_memory_api():
    """
    Example: Integrate circuit breaker into memory API endpoints.
    """
    from api.memory_api import memory_bp
    
    # Wrap the store endpoint
    original_store = memory_bp.view_functions['store_memory']
    
    @api_circuit_breaker(failure_threshold=3, recovery_timeout=30)
    def protected_store(*args, **kwargs):
        return original_store(*args, **kwargs)
    
    memory_bp.view_functions['store_memory'] = protected_store
    
    # Wrap the search endpoint
    original_search = memory_bp.view_functions['search_memory']
    
    @api_circuit_breaker(
        failure_threshold=5,
        recovery_timeout=20,
        fallback_response={
            'results': [],
            'message': 'Search temporarily unavailable, please try again',
            'fallback': True
        }
    )
    def protected_search(*args, **kwargs):
        return original_search(*args, **kwargs)
    
    memory_bp.view_functions['search_memory'] = protected_search


def integrate_synthesis_api():
    """
    Example: Integrate circuit breaker into synthesis API endpoints.
    """
    from api.streaming import streaming_bp
    
    # Protect the synthesis endpoint
    original_synthesis = streaming_bp.view_functions.get('stream_synthesis')
    
    if original_synthesis:
        @api_circuit_breaker(
            failure_threshold=3,
            recovery_timeout=45,
            fallback_response={
                'error': 'Synthesis service temporarily unavailable',
                'message': 'Heavy load detected, please retry in a moment'
            }
        )
        def protected_synthesis(*args, **kwargs):
            return original_synthesis(*args, **kwargs)
        
        streaming_bp.view_functions['stream_synthesis'] = protected_synthesis


def integrate_file_processing():
    """
    Example: Integrate circuit breaker into file processing endpoints.
    """
    from api.async_file_processing import async_file_bp
    
    # Protect file upload endpoint
    original_upload = async_file_bp.view_functions.get('upload_file_async')
    
    if original_upload:
        @api_circuit_breaker(
            failure_threshold=2,  # Lower threshold for file uploads
            recovery_timeout=120,  # Longer recovery for file processing
            fallback_response={
                'error': 'File processing service is currently unavailable',
                'message': 'System is experiencing high load, please try again later',
                'retry_after': 120
            }
        )
        def protected_upload(*args, **kwargs):
            return original_upload(*args, **kwargs)
        
        async_file_bp.view_functions['upload_file_async'] = protected_upload


# Circuit breaker monitoring endpoint
from flask import Blueprint

circuit_monitor_bp = Blueprint('circuit_monitor', __name__)


@circuit_monitor_bp.route('/circuit-breakers/status', methods=['GET'])
def circuit_breaker_status():
    """
    Get status of all circuit breakers in the system.
    """
    stats = circuit_breaker_manager.get_all_stats()
    open_circuits = circuit_breaker_manager.get_open_circuits()
    
    return jsonify({
        'circuit_breakers': stats,
        'open_circuits': open_circuits,
        'total_circuits': len(stats)
    })


@circuit_monitor_bp.route('/circuit-breakers/reset', methods=['POST'])
def reset_circuit_breakers():
    """
    Reset all circuit breakers (admin endpoint).
    """
    circuit_breaker_manager.reset_all()
    
    return jsonify({
        'status': 'success',
        'message': 'All circuit breakers have been reset'
    })


def initialize_circuit_breakers(app):
    """
    Initialize circuit breaker protection for all critical endpoints.
    
    Call this during app initialization:
        initialize_circuit_breakers(app)
    """
    # Register monitoring blueprint
    app.register_blueprint(circuit_monitor_bp, url_prefix='/api/monitoring')
    
    # Integrate circuit breakers
    with app.app_context():
        try:
            integrate_memory_api()
            logger.info("Circuit breakers integrated for memory API")
        except Exception as e:
            logger.warning(f"Failed to integrate circuit breakers for memory API: {e}")
        
        try:
            integrate_synthesis_api()
            logger.info("Circuit breakers integrated for synthesis API")
        except Exception as e:
            logger.warning(f"Failed to integrate circuit breakers for synthesis API: {e}")
        
        try:
            integrate_file_processing()
            logger.info("Circuit breakers integrated for file processing")
        except Exception as e:
            logger.warning(f"Failed to integrate circuit breakers for file processing: {e}")
    
    logger.info("Circuit breaker integration completed")