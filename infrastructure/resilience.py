"""
Resilience Infrastructure for SUM Platform
==========================================

Production-grade resilience patterns including:
- Circuit breaker pattern for fault tolerance
- Retry logic with exponential backoff
- Bulkhead pattern for resource isolation
- Timeout management
- Fallback mechanisms
- Health checks and monitoring

Based on Netflix Hystrix and modern resilience engineering principles.

Author: ototao
License: Apache License 2.0
"""

import asyncio
import functools
import logging
import random
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
import psutil

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states following the pattern"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking calls due to failures
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    
    # Failure thresholds
    failure_threshold: int = 5              # Failures before opening
    failure_rate_threshold: float = 0.5     # Failure rate to open (50%)
    
    # Recovery settings
    recovery_timeout: float = 60.0          # Seconds before attempting recovery
    success_threshold: int = 2              # Successes needed to close circuit
    
    # Window settings
    window_size: int = 10                   # Size of rolling window
    window_duration: float = 60.0           # Duration of time window
    
    # Request volume
    min_request_volume: int = 10            # Minimum requests before opening
    
    # Monitoring
    enable_metrics: bool = True
    enable_logging: bool = True


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open"""
    
    def __init__(self, message: str, circuit_name: str, state: CircuitState):
        super().__init__(message)
        self.circuit_name = circuit_name
        self.state = state


class RollingWindow:
    """
    Rolling window for tracking metrics over time.
    Used for calculating failure rates and response times.
    """
    
    def __init__(self, window_size: int = 10, window_duration: float = 60.0):
        """
        Initialize rolling window.
        
        Args:
            window_size: Maximum number of entries
            window_duration: Maximum age of entries in seconds
        """
        self.window_size = window_size
        self.window_duration = window_duration
        self.entries = deque(maxlen=window_size)
        self.lock = threading.Lock()
    
    def add(self, success: bool, response_time: float):
        """Add an entry to the window"""
        with self.lock:
            entry = {
                'timestamp': time.time(),
                'success': success,
                'response_time': response_time
            }
            self.entries.append(entry)
            self._cleanup()
    
    def _cleanup(self):
        """Remove entries older than window duration"""
        cutoff_time = time.time() - self.window_duration
        while self.entries and self.entries[0]['timestamp'] < cutoff_time:
            self.entries.popleft()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Calculate metrics from window"""
        with self.lock:
            self._cleanup()
            
            if not self.entries:
                return {
                    'total': 0,
                    'successes': 0,
                    'failures': 0,
                    'failure_rate': 0.0,
                    'avg_response_time': 0.0
                }
            
            total = len(self.entries)
            successes = sum(1 for e in self.entries if e['success'])
            failures = total - successes
            failure_rate = failures / total if total > 0 else 0.0
            avg_response_time = sum(e['response_time'] for e in self.entries) / total
            
            return {
                'total': total,
                'successes': successes,
                'failures': failures,
                'failure_rate': failure_rate,
                'avg_response_time': avg_response_time,
                'p50': self._percentile(50),
                'p95': self._percentile(95),
                'p99': self._percentile(99)
            }
    
    def _percentile(self, p: float) -> float:
        """Calculate percentile of response times"""
        if not self.entries:
            return 0.0
        
        times = sorted(e['response_time'] for e in self.entries)
        index = int(len(times) * p / 100)
        return times[min(index, len(times) - 1)]


class CircuitBreaker:
    """
    Advanced circuit breaker with rolling windows and metrics.
    
    Features:
    - Rolling window for failure rate calculation
    - Automatic state transitions
    - Configurable thresholds
    - Comprehensive metrics
    - Thread-safe operation
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.
        
        Args:
            name: Name for identification and logging
            config: Configuration settings
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self.state = CircuitState.CLOSED
        self.last_state_change = datetime.now()
        self.last_failure_time = None
        
        self.window = RollingWindow(
            window_size=self.config.window_size,
            window_duration=self.config.window_duration
        )
        
        self.consecutive_successes = 0
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        
        self.lock = threading.Lock()
        
        if self.config.enable_logging:
            logger.info(f"Circuit breaker '{name}' initialized")
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        
        Raises:
            CircuitBreakerError: If circuit is open
        """
        # Check if call is allowed
        if not self._can_execute():
            self.total_calls += 1
            raise CircuitBreakerError(
                f"Circuit breaker '{self.name}' is {self.state.value}",
                self.name,
                self.state
            )
        
        # Execute function with monitoring
        start_time = time.time()
        success = False
        
        try:
            result = func(*args, **kwargs)
            success = True
            self._record_success(time.time() - start_time)
            return result
        
        except Exception as e:
            self._record_failure(time.time() - start_time)
            raise
        
        finally:
            self.total_calls += 1
    
    async def async_call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute async function through circuit breaker.
        
        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        
        Raises:
            CircuitBreakerError: If circuit is open
        """
        # Check if call is allowed
        if not self._can_execute():
            self.total_calls += 1
            raise CircuitBreakerError(
                f"Circuit breaker '{self.name}' is {self.state.value}",
                self.name,
                self.state
            )
        
        # Execute function with monitoring
        start_time = time.time()
        success = False
        
        try:
            result = await func(*args, **kwargs)
            success = True
            self._record_success(time.time() - start_time)
            return result
        
        except Exception as e:
            self._record_failure(time.time() - start_time)
            raise
        
        finally:
            self.total_calls += 1
    
    def _can_execute(self) -> bool:
        """Check if execution is allowed"""
        with self.lock:
            if self.state == CircuitState.CLOSED:
                return True
            
            if self.state == CircuitState.OPEN:
                # Check if we should transition to half-open
                if self.last_failure_time:
                    elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                    if elapsed >= self.config.recovery_timeout:
                        self._transition_to(CircuitState.HALF_OPEN)
                        return True
                return False
            
            if self.state == CircuitState.HALF_OPEN:
                return True
            
            return False
    
    def _record_success(self, response_time: float):
        """Record successful execution"""
        with self.lock:
            self.window.add(True, response_time)
            self.total_successes += 1
            
            if self.state == CircuitState.HALF_OPEN:
                self.consecutive_successes += 1
                if self.consecutive_successes >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            
            elif self.state == CircuitState.CLOSED:
                # Reset consecutive failures
                self.consecutive_successes = 0
    
    def _record_failure(self, response_time: float):
        """Record failed execution"""
        with self.lock:
            self.window.add(False, response_time)
            self.total_failures += 1
            self.last_failure_time = datetime.now()
            
            if self.state == CircuitState.HALF_OPEN:
                # Immediately open on failure in half-open state
                self._transition_to(CircuitState.OPEN)
            
            elif self.state == CircuitState.CLOSED:
                # Check if we should open the circuit
                metrics = self.window.get_metrics()
                
                if metrics['total'] >= self.config.min_request_volume:
                    if metrics['failure_rate'] >= self.config.failure_rate_threshold:
                        self._transition_to(CircuitState.OPEN)
                    elif metrics['failures'] >= self.config.failure_threshold:
                        self._transition_to(CircuitState.OPEN)
    
    def _transition_to(self, new_state: CircuitState):
        """Transition to new state"""
        if self.config.enable_logging:
            logger.info(
                f"Circuit breaker '{self.name}' transitioning "
                f"from {self.state.value} to {new_state.value}"
            )
        
        self.state = new_state
        self.last_state_change = datetime.now()
        
        if new_state == CircuitState.CLOSED:
            self.consecutive_successes = 0
        elif new_state == CircuitState.HALF_OPEN:
            self.consecutive_successes = 0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        with self.lock:
            window_metrics = self.window.get_metrics()
            
            return {
                'name': self.name,
                'state': self.state.value,
                'last_state_change': self.last_state_change.isoformat(),
                'total_calls': self.total_calls,
                'total_successes': self.total_successes,
                'total_failures': self.total_failures,
                'overall_success_rate': (
                    self.total_successes / self.total_calls
                    if self.total_calls > 0 else 0.0
                ),
                'window_metrics': window_metrics
            }
    
    def reset(self):
        """Reset circuit breaker to closed state"""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.last_state_change = datetime.now()
            self.consecutive_successes = 0
            self.window = RollingWindow(
                window_size=self.config.window_size,
                window_duration=self.config.window_duration
            )
            
            if self.config.enable_logging:
                logger.info(f"Circuit breaker '{self.name}' reset to CLOSED")


class RetryConfig:
    """Configuration for retry behavior"""
    
    def __init__(self,
                 max_retries: int = 3,
                 initial_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        """
        Initialize retry configuration.
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries
            exponential_base: Base for exponential backoff
            jitter: Add random jitter to delays
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


class RetryManager:
    """
    Retry manager with exponential backoff and jitter.
    
    Features:
    - Exponential backoff
    - Random jitter to prevent thundering herd
    - Configurable retry conditions
    - Comprehensive logging
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry manager.
        
        Args:
            config: Retry configuration
        """
        self.config = config or RetryConfig()
    
    def execute_with_retry(self,
                          func: Callable[..., T],
                          *args,
                          retry_on: Optional[List[type]] = None,
                          **kwargs) -> T:
        """
        Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            retry_on: List of exception types to retry on
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        
        Raises:
            Last exception if all retries fail
        """
        retry_on = retry_on or [Exception]
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return func(*args, **kwargs)
            
            except tuple(retry_on) as e:
                last_exception = e
                
                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{self.config.max_retries} "
                        f"after {delay:.2f}s delay. Error: {str(e)}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {self.config.max_retries} retries failed. "
                        f"Last error: {str(e)}"
                    )
        
        raise last_exception
    
    async def async_execute_with_retry(self,
                                      func: Callable[..., T],
                                      *args,
                                      retry_on: Optional[List[type]] = None,
                                      **kwargs) -> T:
        """
        Execute async function with retry logic.
        
        Args:
            func: Async function to execute
            *args: Function arguments
            retry_on: List of exception types to retry on
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        
        Raises:
            Last exception if all retries fail
        """
        retry_on = retry_on or [Exception]
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            
            except tuple(retry_on) as e:
                last_exception = e
                
                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{self.config.max_retries} "
                        f"after {delay:.2f}s delay. Error: {str(e)}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"All {self.config.max_retries} retries failed. "
                        f"Last error: {str(e)}"
                    )
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        delay = self.config.initial_delay * (self.config.exponential_base ** attempt)
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            # Add random jitter (±25%)
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)


class Bulkhead:
    """
    Bulkhead pattern for resource isolation.
    
    Prevents resource exhaustion by limiting concurrent executions
    and providing separate thread pools for different operations.
    """
    
    def __init__(self,
                 name: str,
                 max_concurrent: int = 10,
                 max_queue_size: int = 100,
                 timeout: float = 30.0):
        """
        Initialize bulkhead.
        
        Args:
            name: Bulkhead name for identification
            max_concurrent: Maximum concurrent executions
            max_queue_size: Maximum queue size
            timeout: Execution timeout in seconds
        """
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.timeout = timeout
        
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.queue_size = 0
        self.active_count = 0
        
        self.total_executed = 0
        self.total_rejected = 0
        self.total_timeout = 0
        
        self.lock = threading.Lock()
    
    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with bulkhead protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        
        Raises:
            RuntimeError: If bulkhead is full
            asyncio.TimeoutError: If execution times out
        """
        # Check queue size
        with self.lock:
            if self.queue_size >= self.max_queue_size:
                self.total_rejected += 1
                raise RuntimeError(f"Bulkhead '{self.name}' queue is full")
            self.queue_size += 1
        
        try:
            # Acquire semaphore
            async with self.semaphore:
                with self.lock:
                    self.queue_size -= 1
                    self.active_count += 1
                
                try:
                    # Execute with timeout
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self.timeout
                    )
                    
                    with self.lock:
                        self.total_executed += 1
                    
                    return result
                
                except asyncio.TimeoutError:
                    with self.lock:
                        self.total_timeout += 1
                    raise
                
                finally:
                    with self.lock:
                        self.active_count -= 1
        
        except Exception:
            with self.lock:
                self.queue_size = max(0, self.queue_size - 1)
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get bulkhead metrics"""
        with self.lock:
            return {
                'name': self.name,
                'max_concurrent': self.max_concurrent,
                'active_count': self.active_count,
                'queue_size': self.queue_size,
                'total_executed': self.total_executed,
                'total_rejected': self.total_rejected,
                'total_timeout': self.total_timeout,
                'utilization': self.active_count / self.max_concurrent
            }


class ResilienceManager:
    """
    Central manager for all resilience patterns.
    
    Provides unified interface for circuit breakers, retries, and bulkheads.
    """
    
    def __init__(self):
        """Initialize resilience manager"""
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.bulkheads: Dict[str, Bulkhead] = {}
        self.retry_managers: Dict[str, RetryManager] = {}
        
        # Global metrics
        self.metrics = {
            'total_circuit_breakers': 0,
            'open_circuits': 0,
            'total_bulkheads': 0,
            'total_retries': 0
        }
    
    def create_circuit_breaker(self,
                              name: str,
                              config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Create or get circuit breaker"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name, config)
            self.metrics['total_circuit_breakers'] += 1
        
        return self.circuit_breakers[name]
    
    def create_bulkhead(self,
                       name: str,
                       max_concurrent: int = 10,
                       **kwargs) -> Bulkhead:
        """Create or get bulkhead"""
        if name not in self.bulkheads:
            self.bulkheads[name] = Bulkhead(name, max_concurrent, **kwargs)
            self.metrics['total_bulkheads'] += 1
        
        return self.bulkheads[name]
    
    def create_retry_manager(self,
                           name: str,
                           config: Optional[RetryConfig] = None) -> RetryManager:
        """Create or get retry manager"""
        if name not in self.retry_managers:
            self.retry_managers[name] = RetryManager(config)
        
        return self.retry_managers[name]
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        # Count open circuits
        open_circuits = sum(
            1 for cb in self.circuit_breakers.values()
            if cb.state == CircuitState.OPEN
        )
        
        # Calculate bulkhead utilization
        bulkhead_utilization = {}
        for name, bulkhead in self.bulkheads.items():
            metrics = bulkhead.get_metrics()
            bulkhead_utilization[name] = metrics['utilization']
        
        # System resource metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'circuit_breakers': {
                'total': len(self.circuit_breakers),
                'open': open_circuits,
                'half_open': sum(
                    1 for cb in self.circuit_breakers.values()
                    if cb.state == CircuitState.HALF_OPEN
                )
            },
            'bulkheads': bulkhead_utilization,
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024 ** 3)
            },
            'health_score': self._calculate_health_score(
                open_circuits,
                bulkhead_utilization,
                cpu_percent,
                memory.percent
            )
        }
    
    def _calculate_health_score(self,
                               open_circuits: int,
                               bulkhead_utilization: Dict[str, float],
                               cpu_percent: float,
                               memory_percent: float) -> float:
        """Calculate overall health score (0-100)"""
        score = 100.0
        
        # Deduct for open circuits
        score -= open_circuits * 10
        
        # Deduct for high bulkhead utilization
        if bulkhead_utilization:
            avg_utilization = sum(bulkhead_utilization.values()) / len(bulkhead_utilization)
            if avg_utilization > 0.8:
                score -= (avg_utilization - 0.8) * 50
        
        # Deduct for high resource usage
        if cpu_percent > 80:
            score -= (cpu_percent - 80) * 0.5
        
        if memory_percent > 80:
            score -= (memory_percent - 80) * 0.5
        
        return max(0, min(100, score))


# Global resilience manager instance
resilience_manager = ResilienceManager()


# Decorator for applying resilience patterns
def with_resilience(circuit_breaker: Optional[str] = None,
                    retry: bool = True,
                    bulkhead: Optional[str] = None,
                    fallback: Optional[Callable] = None):
    """
    Decorator to apply resilience patterns to functions.
    
    Args:
        circuit_breaker: Name of circuit breaker to use
        retry: Enable retry logic
        bulkhead: Name of bulkhead to use
        fallback: Fallback function if all else fails
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                # Apply circuit breaker
                if circuit_breaker:
                    cb = resilience_manager.create_circuit_breaker(circuit_breaker)
                    if retry:
                        rm = resilience_manager.create_retry_manager(f"{circuit_breaker}_retry")
                        return await rm.async_execute_with_retry(
                            cb.async_call, func, *args, **kwargs
                        )
                    else:
                        return await cb.async_call(func, *args, **kwargs)
                
                # Apply bulkhead
                elif bulkhead:
                    bh = resilience_manager.create_bulkhead(bulkhead)
                    return await bh.execute(func, *args, **kwargs)
                
                # Just retry
                elif retry:
                    rm = resilience_manager.create_retry_manager(f"{func.__name__}_retry")
                    return await rm.async_execute_with_retry(func, *args, **kwargs)
                
                # No resilience patterns
                else:
                    return await func(*args, **kwargs)
            
            except Exception as e:
                if fallback:
                    logger.warning(f"Executing fallback for {func.__name__}: {str(e)}")
                    return fallback(*args, **kwargs)
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                # Apply circuit breaker
                if circuit_breaker:
                    cb = resilience_manager.create_circuit_breaker(circuit_breaker)
                    if retry:
                        rm = resilience_manager.create_retry_manager(f"{circuit_breaker}_retry")
                        return rm.execute_with_retry(
                            cb.call, func, *args, **kwargs
                        )
                    else:
                        return cb.call(func, *args, **kwargs)
                
                # Just retry
                elif retry:
                    rm = resilience_manager.create_retry_manager(f"{func.__name__}_retry")
                    return rm.execute_with_retry(func, *args, **kwargs)
                
                # No resilience patterns
                else:
                    return func(*args, **kwargs)
            
            except Exception as e:
                if fallback:
                    logger.warning(f"Executing fallback for {func.__name__}: {str(e)}")
                    return fallback(*args, **kwargs)
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Example usage
if __name__ == "__main__":
    # Example: Function with resilience patterns
    @with_resilience(
        circuit_breaker="external_api",
        retry=True,
        fallback=lambda: {"status": "fallback", "data": []}
    )
    def call_external_api():
        """Example function that might fail"""
        import random
        if random.random() < 0.5:
            raise ConnectionError("API unavailable")
        return {"status": "success", "data": [1, 2, 3]}
    
    # Test the resilient function
    for i in range(10):
        try:
            result = call_external_api()
            print(f"Call {i+1}: {result}")
        except Exception as e:
            print(f"Call {i+1} failed: {e}")
    
    # Get system health
    print("\nSystem Health:", resilience_manager.get_system_health())
