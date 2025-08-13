"""
circuit_breaker.py - Circuit Breaker Pattern Implementation

Implements the circuit breaker pattern to prevent cascading failures:
- Monitors service failures and opens circuit when threshold exceeded
- Provides fallback behavior during outages
- Automatically attempts recovery after timeout
- Thread-safe implementation for concurrent requests

Author: SUM Development Team
License: Apache License 2.0
"""

import time
import logging
from typing import Callable, Any, Optional, Dict, List
from functools import wraps
from threading import Lock
from datetime import datetime, timedelta
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is broken, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker implementation with configurable thresholds.
    
    The circuit breaker monitors failures and prevents requests to failing services.
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests fail immediately
    - HALF_OPEN: Testing recovery, limited requests allowed
    """
    
    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception,
                 name: Optional[str] = None):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to catch (others pass through)
            name: Name for logging/monitoring
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name or "CircuitBreaker"
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._lock = Lock()
        
        # Metrics
        self._success_count = 0
        self._total_calls = 0
        self._state_changes = []
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing)."""
        return self._state == CircuitState.OPEN
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker.
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitOpenError: If circuit is open
            Original exception: If function fails
        """
        with self._lock:
            self._total_calls += 1
            
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    raise CircuitOpenError(
                        f"Circuit breaker '{self.name}' is OPEN"
                    )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call async function through circuit breaker.
        
        Args:
            func: Async function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitOpenError: If circuit is open
            Original exception: If function fails
        """
        with self._lock:
            self._total_calls += 1
            
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    raise CircuitOpenError(
                        f"Circuit breaker '{self.name}' is OPEN"
                    )
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        return (
            self._last_failure_time and
            time.time() - self._last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            self._failure_count = 0
            self._success_count += 1
            
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to_closed()
    
    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._failure_count >= self.failure_threshold:
                self._transition_to_open()
    
    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._record_state_change(CircuitState.CLOSED)
        logger.info(f"Circuit breaker '{self.name}' is now CLOSED")
    
    def _transition_to_open(self):
        """Transition to OPEN state."""
        self._state = CircuitState.OPEN
        self._record_state_change(CircuitState.OPEN)
        logger.warning(
            f"Circuit breaker '{self.name}' is now OPEN after {self._failure_count} failures"
        )
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        self._state = CircuitState.HALF_OPEN
        self._record_state_change(CircuitState.HALF_OPEN)
        logger.info(f"Circuit breaker '{self.name}' is now HALF_OPEN, attempting recovery")
    
    def _record_state_change(self, new_state: CircuitState):
        """Record state change for monitoring."""
        self._state_changes.append({
            'timestamp': datetime.now(),
            'state': new_state,
            'failure_count': self._failure_count
        })
        
        # Keep only last 100 state changes
        if len(self._state_changes) > 100:
            self._state_changes = self._state_changes[-100:]
    
    def reset(self):
        """Manually reset the circuit breaker."""
        with self._lock:
            self._transition_to_closed()
            self._last_failure_time = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            success_rate = (
                self._success_count / self._total_calls 
                if self._total_calls > 0 else 0
            )
            
            return {
                'name': self.name,
                'state': self._state.value,
                'failure_count': self._failure_count,
                'success_count': self._success_count,
                'total_calls': self._total_calls,
                'success_rate': success_rate,
                'last_failure_time': self._last_failure_time,
                'state_changes': len(self._state_changes)
            }


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


def circuit_breaker(failure_threshold: int = 5,
                   recovery_timeout: int = 60,
                   expected_exception: type = Exception,
                   fallback: Optional[Callable] = None):
    """
    Decorator to apply circuit breaker pattern to functions.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery
        expected_exception: Exception type to catch
        fallback: Optional fallback function to call when circuit is open
        
    Example:
        @circuit_breaker(failure_threshold=3, recovery_timeout=30)
        def call_external_api():
            # Make API call
            pass
    """
    def decorator(func):
        # Create circuit breaker instance for this function
        breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            name=func.__name__
        )
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return breaker.call(func, *args, **kwargs)
            except CircuitOpenError:
                if fallback:
                    logger.info(f"Circuit open for {func.__name__}, using fallback")
                    return fallback(*args, **kwargs)
                raise
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await breaker.call_async(func, *args, **kwargs)
            except CircuitOpenError:
                if fallback:
                    logger.info(f"Circuit open for {func.__name__}, using fallback")
                    if asyncio.iscoroutinefunction(fallback):
                        return await fallback(*args, **kwargs)
                    return fallback(*args, **kwargs)
                raise
        
        # Attach breaker instance for monitoring
        wrapper.circuit_breaker = breaker
        async_wrapper.circuit_breaker = breaker
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator


class CircuitBreakerManager:
    """Manages multiple circuit breakers for monitoring and control."""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = Lock()
    
    def register(self, breaker: CircuitBreaker):
        """Register a circuit breaker for monitoring."""
        with self._lock:
            self._breakers[breaker.name] = breaker
    
    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._breakers.get(name)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        with self._lock:
            return {
                name: breaker.get_stats()
                for name, breaker in self._breakers.items()
            }
    
    def reset_all(self):
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
    
    def get_open_circuits(self) -> List[str]:
        """Get list of open circuits."""
        with self._lock:
            return [
                name for name, breaker in self._breakers.items()
                if breaker.is_open
            ]


# Global circuit breaker manager
circuit_breaker_manager = CircuitBreakerManager()


# Example usage
if __name__ == "__main__":
    import random
    
    # Simulate unreliable service
    @circuit_breaker(failure_threshold=3, recovery_timeout=5)
    def unreliable_service():
        if random.random() < 0.6:  # 60% failure rate
            raise Exception("Service failure")
        return "Success"
    
    # Fallback function
    def fallback_service():
        return "Fallback response"
    
    # Test circuit breaker
    for i in range(20):
        try:
            result = unreliable_service()
            print(f"Call {i+1}: {result}")
        except Exception as e:
            print(f"Call {i+1}: Failed - {e}")
        
        time.sleep(0.5)
        
        # Print stats
        if hasattr(unreliable_service, 'circuit_breaker'):
            stats = unreliable_service.circuit_breaker.get_stats()
            print(f"  State: {stats['state']}, Failures: {stats['failure_count']}")