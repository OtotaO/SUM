"""
error_handler.py - Comprehensive Error Handling for SUM

This module provides centralized error handling with:
- Custom exception classes
- Error recovery strategies
- Consistent error responses
- Detailed logging and monitoring

Author: SUM Development Team
License: Apache License 2.0
"""

import logging
import traceback
from typing import Dict, Any, Optional, Callable
from functools import wraps
from flask import jsonify, request
import time

logger = logging.getLogger(__name__)


# Custom Exception Classes
class SUMException(Exception):
    """Base exception for all SUM-specific errors."""
    def __init__(self, message: str, code: str = "SUM_ERROR", details: Optional[Dict] = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


class ValidationError(SUMException):
    """Raised when input validation fails."""
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message, "VALIDATION_ERROR", {"field": field})


class ProcessingError(SUMException):
    """Raised when text processing fails."""
    def __init__(self, message: str, stage: Optional[str] = None):
        super().__init__(message, "PROCESSING_ERROR", {"stage": stage})


class MemoryError(SUMException):
    """Raised when memory operations fail."""
    def __init__(self, message: str, operation: Optional[str] = None):
        super().__init__(message, "MEMORY_ERROR", {"operation": operation})


class ConfigurationError(SUMException):
    """Raised when configuration is invalid."""
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(message, "CONFIG_ERROR", {"config_key": config_key})


class ResourceLimitError(SUMException):
    """Raised when resource limits are exceeded."""
    def __init__(self, message: str, limit_type: str, current: Any, limit: Any):
        super().__init__(message, "RESOURCE_LIMIT", {
            "limit_type": limit_type,
            "current": current,
            "limit": limit
        })


# Error Recovery Strategies
class ErrorRecovery:
    """Provides error recovery strategies."""
    
    @staticmethod
    def with_retry(func: Callable, max_attempts: int = 3, delay: float = 1.0):
        """Retry a function with exponential backoff."""
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (2 ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed")
            raise last_error
        return wrapper
    
    @staticmethod
    def with_fallback(primary_func: Callable, fallback_func: Callable):
        """Use fallback function if primary fails."""
        def wrapper(*args, **kwargs):
            try:
                return primary_func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Primary function failed: {e}. Using fallback.")
                return fallback_func(*args, **kwargs)
        return wrapper


# Decorators for Error Handling
def handle_errors(error_type: type = Exception, 
                 message: str = "An error occurred",
                 status_code: int = 500,
                 log_level: str = "error"):
    """Decorator to handle errors in Flask routes."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_type as e:
                error_id = f"ERR_{int(time.time())}_{hash(str(e)) % 10000}"
                
                # Log the error
                log_func = getattr(logger, log_level)
                log_func(f"[{error_id}] {message}: {str(e)}")
                
                if log_level == "error":
                    logger.debug(f"[{error_id}] Traceback: {traceback.format_exc()}")
                
                # Create error response
                error_response = {
                    "error": True,
                    "message": message,
                    "error_id": error_id,
                    "timestamp": time.time()
                }
                
                # Add details for known exceptions
                if isinstance(e, SUMException):
                    error_response["code"] = e.code
                    error_response["details"] = e.details
                
                return jsonify(error_response), status_code
                
        return wrapper
    return decorator


def validate_input(validation_func: Callable):
    """Decorator to validate input before processing."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get request data
            data = request.get_json() if request.is_json else request.form.to_dict()
            
            # Validate
            validation_result = validation_func(data)
            if validation_result is not True:
                raise ValidationError(str(validation_result))
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Input Validators
class InputValidator:
    """Provides common input validation functions."""
    
    @staticmethod
    def validate_text(data: Dict[str, Any], 
                     max_length: Optional[int] = None,
                     min_length: int = 1) -> bool:
        """Validate text input."""
        if 'text' not in data:
            return "Missing required field: text"
        
        text = data['text']
        if not isinstance(text, str):
            return "Text must be a string"
        
        if len(text) < min_length:
            return f"Text too short (minimum {min_length} characters)"
        
        if max_length and len(text) > max_length:
            return f"Text too long (maximum {max_length} characters)"
        
        return True
    
    @staticmethod
    def validate_file(file, allowed_extensions: Optional[set] = None,
                     max_size_mb: Optional[float] = None) -> bool:
        """Validate file upload."""
        if not file:
            return "No file provided"
        
        if file.filename == '':
            return "No file selected"
        
        # Check extension
        if allowed_extensions:
            ext = file.filename.rsplit('.', 1)[-1].lower()
            if ext not in allowed_extensions:
                return f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}"
        
        # Check size
        if max_size_mb:
            file.seek(0, 2)  # Seek to end
            size_mb = file.tell() / (1024 * 1024)
            file.seek(0)  # Reset to beginning
            
            if size_mb > max_size_mb:
                return f"File too large ({size_mb:.1f}MB). Maximum allowed: {max_size_mb}MB"
        
        return True
    
    @staticmethod
    def validate_json_fields(data: Dict[str, Any], 
                           required_fields: list,
                           optional_fields: Optional[list] = None) -> bool:
        """Validate JSON fields."""
        # Check required fields
        for field in required_fields:
            if field not in data:
                return f"Missing required field: {field}"
        
        # Check for unknown fields
        if optional_fields is not None:
            allowed_fields = set(required_fields + optional_fields)
            unknown_fields = set(data.keys()) - allowed_fields
            if unknown_fields:
                return f"Unknown fields: {', '.join(unknown_fields)}"
        
        return True


# Error Monitoring and Reporting
class ErrorMonitor:
    """Monitors and reports errors."""
    
    def __init__(self):
        self.error_counts = {}
        self.error_history = []
        self.max_history = 1000
    
    def record_error(self, error_type: str, error_message: str, context: Optional[Dict] = None):
        """Record an error occurrence."""
        # Update counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Add to history
        self.error_history.append({
            'type': error_type,
            'message': error_message,
            'context': context or {},
            'timestamp': time.time()
        })
        
        # Trim history if too large
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            'error_counts': dict(self.error_counts),
            'total_errors': sum(self.error_counts.values()),
            'recent_errors': self.error_history[-10:],
            'error_rate': self._calculate_error_rate()
        }
    
    def _calculate_error_rate(self) -> float:
        """Calculate errors per minute for the last hour."""
        one_hour_ago = time.time() - 3600
        recent_errors = [e for e in self.error_history if e['timestamp'] > one_hour_ago]
        return len(recent_errors) / 60.0


# Global error monitor instance
error_monitor = ErrorMonitor()


# Utility functions
def format_error_response(error: Exception, include_trace: bool = False) -> Dict[str, Any]:
    """Format an error for API response."""
    response = {
        'error': True,
        'message': str(error),
        'type': error.__class__.__name__
    }
    
    if isinstance(error, SUMException):
        response['code'] = error.code
        response['details'] = error.details
    
    if include_trace:
        response['trace'] = traceback.format_exc()
    
    return response


def safe_execute(func: Callable, default_value: Any = None, log_errors: bool = True):
    """Safely execute a function and return default on error."""
    try:
        return func()
    except Exception as e:
        if log_errors:
            logger.error(f"Error in safe_execute: {e}")
        return default_value