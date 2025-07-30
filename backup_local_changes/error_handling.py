"""
error_handling.py - Centralized error handling utilities

This module provides standardized error handling utilities for the SUM platform,
ensuring consistent error management, logging, and reporting across all components.

Design principles:
- Centralized error management
- Consistent error reporting
- Detailed error information
- Graceful degradation

Author: ototao
License: Apache License 2.0
"""

import logging
import traceback
import sys
import os
from typing import Dict, Any, Optional, Callable, Type, Union, List
from functools import wraps
import json
import time

# Configure logging
logger = logging.getLogger(__name__)

# Define custom exception classes
class SUMError(Exception):
    """Base exception class for all SUM-specific exceptions."""
    
    def __init__(self, message: str, code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            code: Error code (optional)
            details: Additional error details (optional)
        """
        self.message = message
        self.code = code
        self.details = details or {}
        self.timestamp = time.time()
        
        # Call the base class constructor
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the exception to a dictionary.
        
        Returns:
            Dictionary representation of the exception
        """
        return {
            'error': True,
            'message': self.message,
            'code': self.code,
            'details': self.details,
            'timestamp': self.timestamp
        }
    
    def __str__(self) -> str:
        """
        Get string representation of the exception.
        
        Returns:
            String representation
        """
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message


class ConfigurationError(SUMError):
    """Exception raised for configuration-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details (optional)
        """
        super().__init__(message, code="CONFIG_ERROR", details=details)


class DataError(SUMError):
    """Exception raised for data-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details (optional)
        """
        super().__init__(message, code="DATA_ERROR", details=details)


class ProcessingError(SUMError):
    """Exception raised for processing-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details (optional)
        """
        super().__init__(message, code="PROCESSING_ERROR", details=details)


class ValidationError(SUMError):
    """Exception raised for validation-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details (optional)
        """
        super().__init__(message, code="VALIDATION_ERROR", details=details)


class APIError(SUMError):
    """Exception raised for API-related errors."""
    
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            status_code: HTTP status code
            details: Additional error details (optional)
        """
        details = details or {}
        details['status_code'] = status_code
        super().__init__(message, code="API_ERROR", details=details)
        self.status_code = status_code


# Error handling decorators
def handle_exceptions(
    logger_instance: Optional[logging.Logger] = None,
    default_message: str = "An unexpected error occurred",
    reraise: bool = True,
    exception_map: Optional[Dict[Type[Exception], Type[Exception]]] = None
) -> Callable:
    """
    Decorator for handling exceptions in a standardized way.
    
    Args:
        logger_instance: Logger instance to use (default: module logger)
        default_message: Default error message
        reraise: Whether to reraise the exception
        exception_map: Mapping of exception types to custom exception types
        
    Returns:
        Decorated function
    """
    log = logger_instance or logger
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get exception details
                exc_type, exc_value, exc_traceback = sys.exc_info()
                stack_trace = traceback.format_exception(exc_type, exc_value, exc_traceback)
                
                # Log the exception
                log.error(
                    f"Exception in {func.__name__}: {str(e)}\n"
                    f"{''.join(stack_trace)}"
                )
                
                # Map exception if needed
                if exception_map and type(e) in exception_map:
                    mapped_exception = exception_map[type(e)]
                    if issubclass(mapped_exception, SUMError):
                        raise mapped_exception(str(e), details={'original_exception': str(e)})
                    else:
                        raise mapped_exception(str(e))
                
                # Reraise if requested
                if reraise:
                    raise
                
                # Return default error response
                if issubclass(func.__annotations__.get('return', type(None)), dict):
                    return {'error': True, 'message': default_message}
                return None
                
        return wrapper
    return decorator


def validate_input(
    validators: Dict[str, Callable[[Any], bool]],
    error_messages: Optional[Dict[str, str]] = None
) -> Callable:
    """
    Decorator for validating function inputs.
    
    Args:
        validators: Dictionary mapping parameter names to validator functions
        error_messages: Dictionary mapping parameter names to error messages
        
    Returns:
        Decorated function
    """
    error_msgs = error_messages or {}
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        error_msg = error_messages.get(
                            param_name, 
                            f"Invalid value for parameter '{param_name}'"
                        )
                        raise ValidationError(error_msg, details={
                            'parameter': param_name,
                            'value': str(value)
                        })
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Utility functions
def format_exception(exc: Exception) -> Dict[str, Any]:
    """
    Format an exception as a dictionary.
    
    Args:
        exc: Exception to format
        
    Returns:
        Dictionary representation of the exception
    """
    if isinstance(exc, SUMError):
        return exc.to_dict()
    
    return {
        'error': True,
        'message': str(exc),
        'type': exc.__class__.__name__,
        'timestamp': time.time()
    }


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    log_error: bool = True,
    **kwargs
) -> Any:
    """
    Safely execute a function, catching any exceptions.
    
    Args:
        func: Function to execute
        *args: Positional arguments to pass to the function
        default_return: Default return value if an exception occurs
        log_error: Whether to log the error
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Function result or default_return if an exception occurs
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_error:
            logger.error(f"Error executing {func.__name__}: {e}")
        return default_return


def create_error_response(
    message: str,
    code: Optional[str] = None,
    status_code: int = 500,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a standardized error response dictionary.
    
    Args:
        message: Error message
        code: Error code
        status_code: HTTP status code
        details: Additional error details
        
    Returns:
        Error response dictionary
    """
    response = {
        'error': True,
        'message': message,
        'timestamp': time.time()
    }
    
    if code:
        response['code'] = code
        
    if details:
        response['details'] = details
        
    response['status_code'] = status_code
    
    return response


# Flask error handling utilities
def register_error_handlers(app):
    """
    Register error handlers for a Flask application.
    
    Args:
        app: Flask application instance
    """
    if not app:
        return
        
    @app.errorhandler(SUMError)
    def handle_sum_error(error):
        """Handle SUM-specific errors."""
        from flask import jsonify
        
        response = error.to_dict()
        status_code = error.details.get('status_code', 500)
        
        return jsonify(response), status_code
    
    @app.errorhandler(400)
    def handle_bad_request(error):
        """Handle bad request errors."""
        from flask import jsonify
        
        response = create_error_response(
            message="Bad request",
            code="BAD_REQUEST",
            status_code=400,
            details={'original_error': str(error)}
        )
        
        return jsonify(response), 400
    
    @app.errorhandler(404)
    def handle_not_found(error):
        """Handle not found errors."""
        from flask import jsonify
        
        response = create_error_response(
            message="Resource not found",
            code="NOT_FOUND",
            status_code=404,
            details={'original_error': str(error)}
        )
        
        return jsonify(response), 404
    
    @app.errorhandler(500)
    def handle_server_error(error):
        """Handle server errors."""
        from flask import jsonify
        
        logger.error(f"Server error: {error}")
        
        response = create_error_response(
            message="Internal server error",
            code="SERVER_ERROR",
            status_code=500
        )
        
        return jsonify(response), 500
