"""
error_handling_enhanced.py - Enhanced Error Handling with Timeouts and Recovery

Major improvements:
- Comprehensive timeout handling for all operations
- Circuit breaker pattern for external services
- Graceful degradation strategies
- Centralized error response formatting
- Retry mechanisms with exponential backoff
- Error tracking and analytics
- Recovery strategies for different error types

Author: SUM Development Team (Enhanced)
License: Apache License 2.0
"""

import time
import logging
import traceback
import asyncio
from typing import Any, Callable, Optional, Dict, List, Union, TypeVar, Tuple
from functools import wraps
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import TimeoutError as FuturesTimeoutError
import json

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorSeverity(Enum):
    """Error severity levels for proper handling."""
    LOW = "low"          # Can be ignored or logged
    MEDIUM = "medium"    # Should be handled but not critical
    HIGH = "high"        # Must be handled, affects functionality
    CRITICAL = "critical" # System-breaking, requires immediate attention


class ErrorCategory(Enum):
    """Categories of errors for targeted handling."""
    VALIDATION = "validation"
    NETWORK = "network"
    DATABASE = "database"
    PROCESSING = "processing"
    RESOURCE = "resource"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    EXTERNAL_SERVICE = "external_service"


@dataclass
class ErrorContext:
    """Context information for errors."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    stack_trace: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False


class CircuitBreaker:
    """Circuit breaker pattern for external service calls."""
    
    class State(Enum):
        CLOSED = "closed"    # Normal operation
        OPEN = "open"        # Failing, reject calls
        HALF_OPEN = "half_open"  # Testing recovery
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = self.State.CLOSED
        self._lock = threading.Lock()
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == self.State.OPEN:
                if self._should_attempt_reset():
                    self.state = self.State.HALF_OPEN
                else:
                    raise ServiceUnavailableError(
                        "Circuit breaker is OPEN",
                        {"service": func.__name__, "state": self.state.value}
                    )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset the circuit."""
        return (
            self.last_failure_time and 
            datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
        )
    
    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            self.failure_count = 0
            self.state = self.State.CLOSED
    
    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = self.State.OPEN
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class ErrorTracker:
    """Track and analyze errors for patterns."""
    
    def __init__(self, max_history: int = 10000):
        self.errors: deque = deque(maxlen=max_history)
        self.error_counts = defaultdict(lambda: defaultdict(int))
        self._lock = threading.Lock()
    
    def track_error(self, error_context: ErrorContext):
        """Track an error occurrence."""
        with self._lock:
            self.errors.append(error_context)
            self.error_counts[error_context.category][error_context.severity] += 1
            
            # Check for error patterns
            self._analyze_patterns(error_context)
    
    def _analyze_patterns(self, error_context: ErrorContext):
        """Analyze error patterns for systemic issues."""
        # Count similar errors in last 5 minutes
        cutoff_time = datetime.now() - timedelta(minutes=5)
        recent_similar = sum(
            1 for e in self.errors 
            if e.category == error_context.category and 
            e.timestamp > cutoff_time
        )
        
        if recent_similar > 10:
            logger.critical(
                f"High error rate detected: {recent_similar} {error_context.category.value} "
                f"errors in last 5 minutes"
            )
    
    def get_error_stats(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get error statistics."""
        with self._lock:
            if time_window:
                cutoff_time = datetime.now() - time_window
                filtered_errors = [e for e in self.errors if e.timestamp > cutoff_time]
            else:
                filtered_errors = list(self.errors)
            
            stats = {
                'total_errors': len(filtered_errors),
                'by_category': defaultdict(int),
                'by_severity': defaultdict(int),
                'recovery_rate': 0.0
            }
            
            recovery_attempts = 0
            recovery_successes = 0
            
            for error in filtered_errors:
                stats['by_category'][error.category.value] += 1
                stats['by_severity'][error.severity.value] += 1
                
                if error.recovery_attempted:
                    recovery_attempts += 1
                    if error.recovery_successful:
                        recovery_successes += 1
            
            if recovery_attempts > 0:
                stats['recovery_rate'] = recovery_successes / recovery_attempts
            
            return dict(stats)


# Global error tracker instance
error_tracker = ErrorTracker()


# Enhanced Exceptions
class SumException(Exception):
    """Base exception for SUM platform."""
    
    def __init__(self, 
                 message: str,
                 details: Optional[Dict[str, Any]] = None,
                 category: ErrorCategory = ErrorCategory.PROCESSING,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.category = category
        self.severity = severity
        self.error_id = f"{category.value}_{int(time.time() * 1000)}"
        
        # Track the error
        context = ErrorContext(
            error_id=self.error_id,
            category=category,
            severity=severity,
            message=message,
            details=details,
            stack_trace=traceback.format_exc()
        )
        error_tracker.track_error(context)


class ValidationError(SumException):
    """Input validation error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, ErrorCategory.VALIDATION, ErrorSeverity.LOW)


class TimeoutError(SumException):
    """Operation timeout error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, ErrorCategory.TIMEOUT, ErrorSeverity.HIGH)


class ResourceError(SumException):
    """Resource limitation error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, ErrorCategory.RESOURCE, ErrorSeverity.HIGH)


class ServiceUnavailableError(SumException):
    """External service unavailable."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, ErrorCategory.EXTERNAL_SERVICE, ErrorSeverity.CRITICAL)


# Timeout handling decorators
def timeout(seconds: int = 30, error_message: str = "Operation timed out"):
    """Decorator to add timeout to synchronous functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(error_message, {"function": func.__name__, "timeout": seconds})
            
            # Set signal alarm
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            
            return result
        return wrapper
    return decorator


def async_timeout(seconds: int = 30, error_message: str = "Operation timed out"):
    """Decorator to add timeout to async functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                raise TimeoutError(error_message, {"function": func.__name__, "timeout": seconds})
        return wrapper
    return decorator


# Retry mechanisms
def retry(max_attempts: int = 3, 
          delay: float = 1.0,
          backoff: float = 2.0,
          exceptions: Tuple[type, ...] = (Exception,)):
    """Decorator to retry function calls with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
            
            # All attempts failed
            raise last_exception
        return wrapper
    return decorator


def async_retry(max_attempts: int = 3,
                delay: float = 1.0,
                backoff: float = 2.0,
                exceptions: Tuple[type, ...] = (Exception,)):
    """Decorator to retry async function calls."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
            
            raise last_exception
        return wrapper
    return decorator


# Centralized error response formatting
def create_error_response(error: Exception, 
                         request_id: Optional[str] = None,
                         include_trace: bool = False) -> Dict[str, Any]:
    """Create standardized error response."""
    response = {
        'error': True,
        'timestamp': datetime.now().isoformat(),
        'request_id': request_id or 'unknown'
    }
    
    if isinstance(error, SumException):
        response.update({
            'error_id': error.error_id,
            'category': error.category.value,
            'severity': error.severity.value,
            'message': error.message,
            'details': error.details
        })
    else:
        response.update({
            'error_id': f"generic_{int(time.time() * 1000)}",
            'category': ErrorCategory.PROCESSING.value,
            'severity': ErrorSeverity.MEDIUM.value,
            'message': str(error),
            'details': {}
        })
    
    if include_trace:
        response['trace'] = traceback.format_exc()
    
    return response


# Error recovery strategies
class ErrorRecovery:
    """Strategies for recovering from different error types."""
    
    @staticmethod
    def recover_from_timeout(func: Callable, *args, **kwargs) -> Any:
        """Try to recover from timeout by using cached or partial results."""
        # Check if there's a cached result
        cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
        # In real implementation, check actual cache
        
        # Return partial result or default
        logger.info(f"Attempting recovery from timeout for {func.__name__}")
        return None
    
    @staticmethod
    def recover_from_resource_error(error: ResourceError) -> Any:
        """Recover from resource errors by freeing resources."""
        import gc
        
        logger.info("Attempting to recover from resource error")
        
        # Trigger garbage collection
        gc.collect()
        
        # In real implementation, could:
        # - Clear caches
        # - Close idle connections
        # - Reduce batch sizes
        
        return True
    
    @staticmethod
    def degrade_gracefully(original_func: Callable, 
                          fallback_func: Callable,
                          *args, **kwargs) -> Any:
        """Degrade to simpler functionality when primary fails."""
        try:
            return original_func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Primary function failed: {e}. Using fallback.")
            return fallback_func(*args, **kwargs)


# Comprehensive error handler
def handle_with_recovery(exceptions: Tuple[type, ...] = (Exception,),
                        recovery_strategy: Optional[Callable] = None,
                        fallback_value: Any = None):
    """Decorator for comprehensive error handling with recovery."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                error_context = ErrorContext(
                    error_id=f"{func.__name__}_{int(time.time() * 1000)}",
                    category=getattr(e, 'category', ErrorCategory.PROCESSING),
                    severity=getattr(e, 'severity', ErrorSeverity.MEDIUM),
                    message=str(e),
                    stack_trace=traceback.format_exc(),
                    recovery_attempted=True
                )
                
                # Try recovery strategy
                if recovery_strategy:
                    try:
                        result = recovery_strategy(e, func, *args, **kwargs)
                        error_context.recovery_successful = True
                        error_tracker.track_error(error_context)
                        return result
                    except Exception as recovery_error:
                        logger.error(f"Recovery failed: {recovery_error}")
                        error_context.recovery_successful = False
                
                error_tracker.track_error(error_context)
                
                # Return fallback value or re-raise
                if fallback_value is not None:
                    return fallback_value
                raise
        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    print("Testing Enhanced Error Handling")
    print("=" * 50)
    
    # Test timeout handling
    @timeout(seconds=2)
    def slow_function():
        time.sleep(3)
        return "Success"
    
    print("\nTesting timeout handling:")
    try:
        result = slow_function()
    except TimeoutError as e:
        print(f"✓ Timeout caught: {e.message}")
    
    # Test retry mechanism
    attempt_count = 0
    
    @retry(max_attempts=3, delay=0.1)
    def flaky_function():
        global attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionError("Network error")
        return "Success"
    
    print("\nTesting retry mechanism:")
    result = flaky_function()
    print(f"✓ Succeeded after {attempt_count} attempts")
    
    # Test circuit breaker
    circuit = CircuitBreaker(failure_threshold=3, recovery_timeout=1)
    fail_count = 0
    
    def unreliable_service():
        global fail_count
        fail_count += 1
        if fail_count <= 4:
            raise ConnectionError("Service down")
        return "Service OK"
    
    print("\nTesting circuit breaker:")
    for i in range(6):
        try:
            result = circuit.call(unreliable_service)
            print(f"  Call {i+1}: Success - {result}")
        except Exception as e:
            print(f"  Call {i+1}: Failed - {type(e).__name__}: {e}")
        
        if i == 3:  # Wait for recovery timeout
            time.sleep(1.1)
    
    # Test error tracking
    print("\nError statistics:")
    stats = error_tracker.get_error_stats()
    print(f"  Total errors: {stats['total_errors']}")
    print(f"  By category: {dict(stats['by_category'])}")
    print(f"  Recovery rate: {stats['recovery_rate']:.2%}")
    
    print("\nEnhanced error handling ready!")