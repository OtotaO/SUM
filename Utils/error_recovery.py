"""
Comprehensive error recovery mechanisms for robust operation
"""
import time
import logging
import traceback
from typing import Any, Callable, Optional, Dict, List, Type
from functools import wraps
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class ErrorContext:
    """Context information for error handling"""
    error_type: str
    error_message: str
    stack_trace: str
    timestamp: datetime
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    endpoint: Optional[str] = None
    additional_data: Optional[dict] = None
    
    def to_dict(self) -> dict:
        return {
            'error_type': self.error_type,
            'error_message': self.error_message,
            'stack_trace': self.stack_trace,
            'timestamp': self.timestamp.isoformat(),
            'request_id': self.request_id,
            'user_id': self.user_id,
            'endpoint': self.endpoint,
            'additional_data': self.additional_data
        }

class ErrorRecoveryManager:
    """Manages error recovery strategies and error tracking"""
    
    def __init__(self):
        # Error tracking
        self.error_history: List[ErrorContext] = []
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.recovery_strategies: Dict[Type[Exception], Callable] = {}
        
        # Circuit breaker state
        self.circuit_states: Dict[str, dict] = {}
        
        # Register default recovery strategies
        self._register_default_strategies()
        
    def _register_default_strategies(self):
        """Register default recovery strategies for common errors"""
        
        # Database errors
        self.register_strategy(ConnectionError, self._recover_connection_error)
        self.register_strategy(TimeoutError, self._recover_timeout_error)
        
        # File errors
        self.register_strategy(IOError, self._recover_io_error)
        self.register_strategy(MemoryError, self._recover_memory_error)
        
    def register_strategy(self, 
                         error_type: Type[Exception], 
                         recovery_func: Callable[[Exception, dict], Any]):
        """Register a recovery strategy for a specific error type"""
        self.recovery_strategies[error_type] = recovery_func
        
    def track_error(self, error: Exception, context: Optional[dict] = None) -> ErrorContext:
        """Track an error occurrence"""
        error_context = ErrorContext(
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            timestamp=datetime.utcnow(),
            request_id=context.get('request_id') if context else None,
            user_id=context.get('user_id') if context else None,
            endpoint=context.get('endpoint') if context else None,
            additional_data=context
        )
        
        self.error_history.append(error_context)
        self.error_counts[error_context.error_type] += 1
        
        # Keep only recent errors (last 1000)
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
            
        return error_context
        
    def get_error_stats(self, time_window_minutes: int = 60) -> dict:
        """Get error statistics for the specified time window"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
        recent_errors = [e for e in self.error_history if e.timestamp > cutoff_time]
        
        # Group by error type
        error_types = defaultdict(int)
        for error in recent_errors:
            error_types[error.error_type] += 1
            
        return {
            'total_errors': len(recent_errors),
            'error_types': dict(error_types),
            'time_window_minutes': time_window_minutes,
            'most_common_error': max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
        }
        
    async def _recover_connection_error(self, error: Exception, context: dict) -> Any:
        """Recovery strategy for connection errors"""
        logger.info("Attempting to recover from connection error")
        
        # Wait and retry with exponential backoff
        retry_count = context.get('retry_count', 0)
        wait_time = min(2 ** retry_count, 30)  # Max 30 seconds
        
        await asyncio.sleep(wait_time)
        
        # Signal to retry the operation
        return {'action': 'retry', 'wait_time': wait_time}
        
    async def _recover_timeout_error(self, error: Exception, context: dict) -> Any:
        """Recovery strategy for timeout errors"""
        logger.info("Attempting to recover from timeout error")
        
        # Increase timeout and retry
        current_timeout = context.get('timeout', 30)
        new_timeout = min(current_timeout * 1.5, 300)  # Max 5 minutes
        
        return {'action': 'retry', 'new_timeout': new_timeout}
        
    async def _recover_io_error(self, error: Exception, context: dict) -> Any:
        """Recovery strategy for IO errors"""
        logger.info("Attempting to recover from IO error")
        
        # Check disk space
        import shutil
        try:
            stats = shutil.disk_usage('/')
            free_gb = stats.free / (1024 ** 3)
            
            if free_gb < 1:  # Less than 1GB free
                return {'action': 'fail', 'reason': 'Insufficient disk space'}
                
        except Exception:
            pass
            
        # Retry with smaller chunk size if applicable
        return {'action': 'retry', 'reduce_chunk_size': True}
        
    async def _recover_memory_error(self, error: Exception, context: dict) -> Any:
        """Recovery strategy for memory errors"""
        logger.info("Attempting to recover from memory error")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Check memory usage
        import psutil
        memory = psutil.virtual_memory()
        
        if memory.percent > 90:
            return {'action': 'fail', 'reason': 'System memory critically low'}
            
        # Suggest processing with smaller chunks
        return {'action': 'retry', 'use_streaming': True, 'reduce_batch_size': True}

def with_error_recovery(
    retry_count: int = 3,
    backoff_factor: float = 2.0,
    recoverable_errors: Optional[List[Type[Exception]]] = None,
    fallback_result: Any = None,
    error_manager: Optional[ErrorRecoveryManager] = None
):
    """
    Decorator for automatic error recovery
    
    Args:
        retry_count: Maximum number of retries
        backoff_factor: Exponential backoff factor
        recoverable_errors: List of error types to recover from
        fallback_result: Result to return if all retries fail
        error_manager: ErrorRecoveryManager instance
    """
    if recoverable_errors is None:
        recoverable_errors = [Exception]  # Catch all by default
        
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_error = None
            context = {
                'function': func.__name__,
                'args': str(args)[:100],  # Truncate for logging
                'kwargs': str(kwargs)[:100]
            }
            
            for attempt in range(retry_count):
                try:
                    return await func(*args, **kwargs)
                    
                except tuple(recoverable_errors) as e:
                    last_error = e
                    context['retry_count'] = attempt
                    
                    if error_manager:
                        error_context = error_manager.track_error(e, context)
                        
                        # Try recovery strategy
                        recovery_func = error_manager.recovery_strategies.get(type(e))
                        if recovery_func:
                            recovery_result = await recovery_func(e, context)
                            
                            if recovery_result.get('action') == 'fail':
                                logger.error(f"Recovery failed: {recovery_result.get('reason')}")
                                break
                                
                            # Apply recovery suggestions
                            if recovery_result.get('new_timeout'):
                                kwargs['timeout'] = recovery_result['new_timeout']
                            if recovery_result.get('reduce_batch_size'):
                                if 'batch_size' in kwargs:
                                    kwargs['batch_size'] = max(1, kwargs['batch_size'] // 2)
                                    
                    # Wait before retry
                    wait_time = backoff_factor ** attempt
                    logger.warning(f"Error in {func.__name__} (attempt {attempt + 1}/{retry_count}): {e}. "
                                 f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    
                except Exception as e:
                    # Non-recoverable error
                    if error_manager:
                        error_manager.track_error(e, context)
                    raise
                    
            # All retries failed
            if fallback_result is not None:
                logger.error(f"All retries failed for {func.__name__}, returning fallback result")
                return fallback_result
            else:
                raise last_error
                
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_error = None
            context = {
                'function': func.__name__,
                'args': str(args)[:100],
                'kwargs': str(kwargs)[:100]
            }
            
            for attempt in range(retry_count):
                try:
                    return func(*args, **kwargs)
                    
                except tuple(recoverable_errors) as e:
                    last_error = e
                    context['retry_count'] = attempt
                    
                    if error_manager:
                        error_manager.track_error(e, context)
                        
                    # Wait before retry
                    wait_time = backoff_factor ** attempt
                    logger.warning(f"Error in {func.__name__} (attempt {attempt + 1}/{retry_count}): {e}. "
                                 f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    
                except Exception as e:
                    # Non-recoverable error
                    if error_manager:
                        error_manager.track_error(e, context)
                    raise
                    
            # All retries failed
            if fallback_result is not None:
                logger.error(f"All retries failed for {func.__name__}, returning fallback result")
                return fallback_result
            else:
                raise last_error
                
        # Return appropriate wrapper based on function type
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator

class CircuitBreaker:
    """
    Enhanced circuit breaker with error recovery
    """
    
    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: Type[Exception] = Exception,
                 name: Optional[str] = None):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name or "CircuitBreaker"
        
        self._failure_count = 0
        self._last_failure_time = None
        self._state = 'closed'  # closed, open, half-open
        
    def __call__(self, func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not self._is_available():
                raise Exception(f"Circuit breaker {self.name} is OPEN")
                
            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.expected_exception as e:
                self._on_failure()
                raise
                
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not self._is_available():
                raise Exception(f"Circuit breaker {self.name} is OPEN")
                
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.expected_exception as e:
                self._on_failure()
                raise
                
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    def _is_available(self) -> bool:
        """Check if circuit breaker allows requests"""
        if self._state == 'closed':
            return True
            
        if self._state == 'open':
            # Check if recovery timeout has passed
            if self._last_failure_time and \
               time.time() - self._last_failure_time > self.recovery_timeout:
                self._state = 'half-open'
                logger.info(f"Circuit breaker {self.name} moved to HALF-OPEN state")
                return True
            return False
            
        # half-open state
        return True
        
    def _on_success(self):
        """Handle successful request"""
        if self._state == 'half-open':
            self._state = 'closed'
            self._failure_count = 0
            logger.info(f"Circuit breaker {self.name} moved to CLOSED state")
            
    def _on_failure(self):
        """Handle failed request"""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._failure_count >= self.failure_threshold:
            self._state = 'open'
            logger.warning(f"Circuit breaker {self.name} moved to OPEN state after {self._failure_count} failures")
            
    def get_state(self) -> dict:
        """Get current circuit breaker state"""
        return {
            'name': self.name,
            'state': self._state,
            'failure_count': self._failure_count,
            'last_failure_time': self._last_failure_time
        }