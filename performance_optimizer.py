#!/usr/bin/env python3
"""
performance_optimizer.py - Performance Optimization Utilities for SUM

Provides performance monitoring, profiling, and optimization utilities
to ensure SUM runs efficiently at scale.

Author: SUM Development Team
License: Apache License 2.0
"""

import time
import psutil
import functools
import asyncio
from typing import Dict, Any, Callable, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import cProfile
import pstats
import io
from contextlib import contextmanager
import threading
from collections import defaultdict, deque
import gc
import sys

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    function_name: str
    execution_time: float
    memory_used: float
    cpu_percent: float
    timestamp: datetime
    success: bool
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'function': self.function_name,
            'execution_time_ms': self.execution_time * 1000,
            'memory_mb': self.memory_used / (1024 * 1024),
            'cpu_percent': self.cpu_percent,
            'timestamp': self.timestamp.isoformat(),
            'success': self.success,
            'error': self.error
        }


class PerformanceMonitor:
    """Monitor and track performance metrics for SUM operations."""
    
    def __init__(self, name: str = "SUM"):
        self.name = name
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.current_metrics: Dict[str, PerformanceMetrics] = {}
        self._lock = threading.Lock()
        
    def record_metric(self, metric: PerformanceMetrics):
        """Record a performance metric."""
        with self._lock:
            self.metrics_history[metric.function_name].append(metric)
            self.current_metrics[metric.function_name] = metric
    
    def get_summary(self, function_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary statistics."""
        with self._lock:
            if function_name:
                metrics = list(self.metrics_history.get(function_name, []))
            else:
                metrics = []
                for metric_list in self.metrics_history.values():
                    metrics.extend(metric_list)
            
            if not metrics:
                return {'message': 'No metrics recorded'}
            
            # Calculate statistics
            execution_times = [m.execution_time for m in metrics if m.success]
            memory_usage = [m.memory_used for m in metrics if m.success]
            cpu_usage = [m.cpu_percent for m in metrics if m.success]
            
            return {
                'total_calls': len(metrics),
                'successful_calls': len([m for m in metrics if m.success]),
                'failed_calls': len([m for m in metrics if not m.success]),
                'avg_execution_time_ms': sum(execution_times) / len(execution_times) * 1000 if execution_times else 0,
                'max_execution_time_ms': max(execution_times) * 1000 if execution_times else 0,
                'min_execution_time_ms': min(execution_times) * 1000 if execution_times else 0,
                'avg_memory_mb': sum(memory_usage) / len(memory_usage) / (1024 * 1024) if memory_usage else 0,
                'avg_cpu_percent': sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0
            }
    
    def get_slow_operations(self, threshold_ms: float = 1000) -> List[Dict[str, Any]]:
        """Get operations that exceeded time threshold."""
        slow_ops = []
        
        with self._lock:
            for func_name, metrics in self.metrics_history.items():
                for metric in metrics:
                    if metric.execution_time * 1000 > threshold_ms:
                        slow_ops.append({
                            'function': func_name,
                            'execution_time_ms': metric.execution_time * 1000,
                            'timestamp': metric.timestamp.isoformat()
                        })
        
        return sorted(slow_ops, key=lambda x: x['execution_time_ms'], reverse=True)


# Global performance monitor
_performance_monitor = PerformanceMonitor()


def measure_performance(monitor: Optional[PerformanceMonitor] = None):
    """
    Decorator to measure function performance.
    
    Args:
        monitor: Optional custom monitor, uses global if None
    """
    def decorator(func: Callable) -> Callable:
        perf_monitor = monitor or _performance_monitor
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Get initial metrics
            process = psutil.Process()
            start_time = time.time()
            start_memory = process.memory_info().rss
            
            success = True
            error = None
            result = None
            
            try:
                # Execute function
                result = func(*args, **kwargs)
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                # Calculate metrics
                end_time = time.time()
                end_memory = process.memory_info().rss
                
                metric = PerformanceMetrics(
                    function_name=func.__name__,
                    execution_time=end_time - start_time,
                    memory_used=end_memory - start_memory,
                    cpu_percent=process.cpu_percent(),
                    timestamp=datetime.now(),
                    success=success,
                    error=error
                )
                
                perf_monitor.record_metric(metric)
                
                # Log if slow
                if metric.execution_time > 1.0:  # 1 second
                    logger.warning(f"Slow operation: {func.__name__} took {metric.execution_time:.2f}s")
            
            return result
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get initial metrics
            process = psutil.Process()
            start_time = time.time()
            start_memory = process.memory_info().rss
            
            success = True
            error = None
            result = None
            
            try:
                # Execute function
                result = await func(*args, **kwargs)
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                # Calculate metrics
                end_time = time.time()
                end_memory = process.memory_info().rss
                
                metric = PerformanceMetrics(
                    function_name=func.__name__,
                    execution_time=end_time - start_time,
                    memory_used=end_memory - start_memory,
                    cpu_percent=process.cpu_percent(),
                    timestamp=datetime.now(),
                    success=success,
                    error=error
                )
                
                perf_monitor.record_metric(metric)
                
                # Log if slow
                if metric.execution_time > 1.0:
                    logger.warning(f"Slow operation: {func.__name__} took {metric.execution_time:.2f}s")
            
            return result
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


@contextmanager
def profile_code(name: str = "profile", print_stats: bool = True, 
                 sort_by: str = 'cumulative', limit: int = 20):
    """
    Context manager for profiling code blocks.
    
    Usage:
        with profile_code("my_operation"):
            # Code to profile
            expensive_operation()
    """
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        yield profiler
    finally:
        profiler.disable()
        
        if print_stats:
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats(sort_by)
            ps.print_stats(limit)
            
            logger.info(f"Profile results for '{name}':\n{s.getvalue()}")


class MemoryOptimizer:
    """Utilities for memory optimization."""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / (1024 * 1024)
        }
    
    @staticmethod
    def optimize_memory():
        """Run garbage collection and optimize memory."""
        # Force garbage collection
        collected = gc.collect()
        
        # Get memory stats before and after
        before = MemoryOptimizer.get_memory_usage()
        
        # Clear caches if possible
        if hasattr(sys, 'intern'):
            # Clear interned strings (Python 3.8+)
            pass
        
        after = MemoryOptimizer.get_memory_usage()
        
        freed_mb = before['rss_mb'] - after['rss_mb']
        
        logger.info(f"Memory optimization: collected {collected} objects, freed {freed_mb:.2f} MB")
        
        return {
            'objects_collected': collected,
            'memory_freed_mb': freed_mb,
            'current_usage': after
        }
    
    @staticmethod
    @contextmanager
    def memory_limit(max_memory_mb: float):
        """
        Context manager to limit memory usage.
        
        Raises MemoryError if limit exceeded.
        """
        def check_memory():
            usage = MemoryOptimizer.get_memory_usage()
            if usage['rss_mb'] > max_memory_mb:
                raise MemoryError(f"Memory limit exceeded: {usage['rss_mb']:.2f} MB > {max_memory_mb} MB")
        
        # Check periodically
        timer = threading.Timer(1.0, check_memory)
        timer.daemon = True
        timer.start()
        
        try:
            yield
        finally:
            timer.cancel()


class CacheOptimizer:
    """Optimize caching for better performance."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.access_count: Dict[str, int] = defaultdict(int)
        self.hit_count = 0
        self.miss_count = 0
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                
                # Check if expired
                if time.time() - timestamp > self.ttl_seconds:
                    del self.cache[key]
                    self.miss_count += 1
                    return None
                
                self.access_count[key] += 1
                self.hit_count += 1
                return value
            
            self.miss_count += 1
            return None
    
    def set(self, key: str, value: Any):
        """Set value in cache."""
        with self._lock:
            # Evict least recently used if at capacity
            if len(self.cache) >= self.max_size:
                # Find least accessed key
                lru_key = min(self.cache.keys(), 
                            key=lambda k: self.access_count.get(k, 0))
                del self.cache[lru_key]
                if lru_key in self.access_count:
                    del self.access_count[lru_key]
            
            self.cache[key] = (value, time.time())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }
    
    def clear(self):
        """Clear the cache."""
        with self._lock:
            self.cache.clear()
            self.access_count.clear()
            self.hit_count = 0
            self.miss_count = 0


def optimize_for_batch_processing(items: List[Any], batch_size: int = 100) -> List[List[Any]]:
    """
    Split items into optimal batches for processing.
    
    Args:
        items: List of items to process
        batch_size: Maximum items per batch
        
    Returns:
        List of batches
    """
    # Adjust batch size based on available memory
    memory = MemoryOptimizer.get_memory_usage()
    available_mb = memory['available_mb']
    
    # Reduce batch size if memory is low
    if available_mb < 500:  # Less than 500MB available
        batch_size = max(10, batch_size // 4)
    elif available_mb < 1000:  # Less than 1GB available
        batch_size = max(20, batch_size // 2)
    
    # Create batches
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    
    logger.info(f"Created {len(batches)} batches of size {batch_size}")
    return batches


# Performance testing utilities
def benchmark_function(func: Callable, *args, iterations: int = 100, **kwargs) -> Dict[str, Any]:
    """
    Benchmark a function over multiple iterations.
    
    Returns:
        Dictionary with benchmark results
    """
    times = []
    memory_usage = []
    
    for _ in range(iterations):
        process = psutil.Process()
        
        start_time = time.time()
        start_memory = process.memory_info().rss
        
        func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = process.memory_info().rss
        
        times.append(end_time - start_time)
        memory_usage.append(end_memory - start_memory)
    
    return {
        'function': func.__name__,
        'iterations': iterations,
        'avg_time_ms': sum(times) / len(times) * 1000,
        'min_time_ms': min(times) * 1000,
        'max_time_ms': max(times) * 1000,
        'avg_memory_kb': sum(memory_usage) / len(memory_usage) / 1024,
        'total_time_s': sum(times)
    }


# Example usage and testing
if __name__ == "__main__":
    print("Testing Performance Optimization Utilities")
    print("=" * 50)
    
    # Test performance monitoring
    @measure_performance()
    def slow_function(duration: float = 0.5):
        time.sleep(duration)
        return "Done"
    
    # Run function multiple times
    for i in range(3):
        slow_function(0.1 * (i + 1))
    
    # Get performance summary
    summary = _performance_monitor.get_summary("slow_function")
    print("\nPerformance Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Test memory optimization
    print("\nMemory Usage:")
    memory_stats = MemoryOptimizer.get_memory_usage()
    for key, value in memory_stats.items():
        print(f"  {key}: {value:.2f}")
    
    # Test caching
    cache = CacheOptimizer(max_size=10)
    
    # Add items to cache
    for i in range(15):
        cache.set(f"key_{i}", f"value_{i}")
    
    # Test cache hits/misses
    cache.get("key_14")  # Hit
    cache.get("key_0")   # Miss (evicted)
    
    print("\nCache Statistics:")
    cache_stats = cache.get_stats()
    for key, value in cache_stats.items():
        print(f"  {key}: {value}")
    
    # Test profiling
    print("\nProfiling Example:")
    with profile_code("test_operation", limit=5):
        # Simulate some work
        result = sum(i ** 2 for i in range(10000))
    
    print("\nPerformance optimization utilities ready!")