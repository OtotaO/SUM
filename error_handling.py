#!/usr/bin/env python3
"""
error_handling.py - Comprehensive Error Handling for SUM

Provides robust error handling, logging, and recovery mechanisms for all SUM components.
Ensures graceful degradation and helpful error messages for users.

Author: SUM Development Team
License: Apache License 2.0
"""

import logging
import traceback
import functools
import asyncio
from typing import Any, Callable, Optional, Dict, Type, Union
from datetime import datetime
import json
import sys
from enum import Enum

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('sum_errors.log', mode='a')
    ]
)


class ErrorSeverity(Enum):
    """Error severity levels for SUM system."""
    LOW = "low"        # Recoverable, minimal impact
    MEDIUM = "medium"  # Recoverable, some functionality affected
    HIGH = "high"      # Critical, major functionality affected
    CRITICAL = "critical"  # System failure, requires immediate attention


class SumError(Exception):
    """Base exception for all SUM-specific errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 error_code: Optional[str] = None, details: Optional[Dict] = None):
        super().__init__(message)
        self.severity = severity
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/API responses."""
        return {
            'error_code': self.error_code,
            'message': str(self),
            'severity': self.severity.value,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }


class ProcessingError(SumError):
    """Error during content processing."""
    pass


class ConfigurationError(SumError):
    """Error in system configuration."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(message, severity=ErrorSeverity.HIGH)
        if config_key:
            self.details['config_key'] = config_key


class ResourceError(SumError):
    """Error accessing required resources (models, files, etc.)."""
    pass


class NetworkError(SumError):
    """Network-related errors."""
    pass


class AIModelError(SumError):
    """Error with AI model operations."""
    pass


class ValidationError(SumError):
    """Input validation error."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message, severity=ErrorSeverity.LOW)
        if field:
            self.details['field'] = field


class ErrorHandler:
    """Centralized error handling for SUM system."""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.logger = logging.getLogger(component_name)
        self.error_counts: Dict[str, int] = {}
        self.last_errors: Dict[str, SumError] = {}
    
    def handle_error(self, error: Exception, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Handle an error with appropriate logging and recovery.
        
        Args:
            error: The exception that occurred
            context: Additional context about the error
            
        Returns:
            Error information dict for API responses
        """
        error_info = self._extract_error_info(error, context)
        
        # Log based on severity
        if isinstance(error, SumError):
            self._log_sum_error(error, error_info)
        else:
            self._log_generic_error(error, error_info)
        
        # Track error frequency
        self._track_error(error)
        
        # Attempt recovery if applicable
        recovery_action = self._determine_recovery_action(error)
        if recovery_action:
            error_info['recovery_action'] = recovery_action
        
        return error_info
    
    def _extract_error_info(self, error: Exception, context: Optional[Dict]) -> Dict[str, Any]:
        """Extract structured information from an error."""
        if isinstance(error, SumError):
            error_info = error.to_dict()
        else:
            error_info = {
                'error_code': error.__class__.__name__,
                'message': str(error),
                'severity': ErrorSeverity.MEDIUM.value,
                'timestamp': datetime.now().isoformat()
            }
        
        # Add context
        if context:
            error_info['context'] = context
        
        # Add traceback for debugging
        error_info['traceback'] = traceback.format_exc()
        
        return error_info
    
    def _log_sum_error(self, error: SumError, error_info: Dict[str, Any]):
        """Log SUM-specific errors based on severity."""
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"{error.error_code}: {error}", extra=error_info)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(f"{error.error_code}: {error}", extra=error_info)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"{error.error_code}: {error}", extra=error_info)
        else:
            self.logger.info(f"{error.error_code}: {error}", extra=error_info)
    
    def _log_generic_error(self, error: Exception, error_info: Dict[str, Any]):
        """Log generic Python exceptions."""
        self.logger.error(f"Unexpected error: {error}", extra=error_info, exc_info=True)
    
    def _track_error(self, error: Exception):
        """Track error frequency for monitoring."""
        error_key = error.__class__.__name__
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        if isinstance(error, SumError):
            self.last_errors[error_key] = error
    
    def _determine_recovery_action(self, error: Exception) -> Optional[str]:
        """Determine appropriate recovery action for an error."""
        if isinstance(error, NetworkError):
            return "retry_with_backoff"
        elif isinstance(error, ResourceError):
            return "use_fallback_resource"
        elif isinstance(error, AIModelError):
            return "switch_to_simpler_model"
        elif isinstance(error, ValidationError):
            return "request_valid_input"
        return None
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        return {
            'component': self.component_name,
            'error_counts': self.error_counts,
            'last_errors': {
                key: error.to_dict() 
                for key, error in self.last_errors.items()
            }
        }


def with_error_handling(component_name: str, 
                       fallback_value: Any = None,
                       reraise: bool = False):
    """
    Decorator for automatic error handling.
    
    Args:
        component_name: Name of the component for logging
        fallback_value: Value to return on error
        reraise: Whether to reraise the error after handling
    """
    def decorator(func: Callable) -> Callable:
        handler = ErrorHandler(component_name)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function': func.__name__,
                    'args': str(args)[:100],  # Truncate for logging
                    'kwargs': str(kwargs)[:100]
                }
                error_info = handler.handle_error(e, context)
                
                if reraise:
                    raise
                
                return fallback_value
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function': func.__name__,
                    'args': str(args)[:100],
                    'kwargs': str(kwargs)[:100]
                }
                error_info = handler.handle_error(e, context)
                
                if reraise:
                    raise
                
                return fallback_value
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class RetryManager:
    """Manages retry logic with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0,
                 max_delay: float = 60.0, exponential_base: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    async def retry_async(self, func: Callable, *args, **kwargs) -> Any:
        """Retry an async function with exponential backoff."""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries - 1:
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    await asyncio.sleep(delay)
        
        raise last_exception
    
    def retry_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Retry a sync function with exponential backoff."""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries - 1:
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    import time
                    time.sleep(delay)
        
        raise last_exception


# Global error handler instance
global_error_handler = ErrorHandler("SUM_Global")


def handle_api_error(error: Exception) -> Dict[str, Any]:
    """Convert exceptions to API-friendly error responses."""
    if isinstance(error, SumError):
        return {
            'success': False,
            'error': error.to_dict()
        }
    else:
        return {
            'success': False,
            'error': {
                'error_code': 'InternalError',
                'message': 'An unexpected error occurred',
                'severity': 'high',
                'timestamp': datetime.now().isoformat()
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Test error handling
    @with_error_handling("TestComponent", fallback_value="Fallback result")
    def test_function(should_fail: bool = False):
        if should_fail:
            raise ProcessingError("Test processing failed", details={'reason': 'test'})
        return "Success"
    
    # Test success case
    print("Success case:", test_function(False))
    
    # Test failure case
    print("Failure case:", test_function(True))
    
    # Test custom errors
    try:
        raise ValidationError("Invalid input format", field="email")
    except SumError as e:
        print("Validation error:", e.to_dict())
    
    # Test retry manager
    retry_manager = RetryManager(max_retries=3)
    
    attempt_count = 0
    def flaky_function():
        global attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise NetworkError("Connection failed")
        return "Success after retries"
    
    try:
        result = retry_manager.retry_sync(flaky_function)
        print("Retry result:", result)
    except Exception as e:
        print("Retry failed:", e)