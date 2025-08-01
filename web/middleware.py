"""
middleware.py - Flask Middleware and Decorators

Clean middleware implementation following Carmack's principles:
- Each decorator has ONE clear purpose
- Fast execution with minimal overhead
- Thread-safe implementations
- Clear interfaces

Author: ototao
License: Apache License 2.0
"""

import time
from functools import wraps
from threading import Lock
from flask import request, jsonify, current_app


class RateLimiter:
    """Thread-safe rate limiter with time-based cleanup."""
    
    def __init__(self):
        self.calls = {}
        self.lock = Lock()
    
    def check_rate_limit(self, client_id, max_calls, time_frame):
        """
        Check if client has exceeded rate limit.
        
        Returns:
            bool: True if within limit, False if exceeded
        """
        current_time = time.time()
        
        with self.lock:
            # Clean old entries
            if client_id in self.calls:
                self.calls[client_id] = [
                    timestamp for timestamp in self.calls[client_id]
                    if current_time - timestamp < time_frame
                ]
                
                if not self.calls[client_id]:
                    del self.calls[client_id]
            
            # Check limit
            if client_id in self.calls and len(self.calls[client_id]) >= max_calls:
                return False
            
            # Record call
            if client_id not in self.calls:
                self.calls[client_id] = []
            self.calls[client_id].append(current_time)
            
            return True


# Global rate limiter instance
_rate_limiter = RateLimiter()


def rate_limit(max_calls=10, time_frame=60):
    """
    Rate limiting decorator.
    
    Args:
        max_calls: Maximum calls allowed in time frame
        time_frame: Time frame in seconds
    """
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            client_id = request.remote_addr
            
            if not _rate_limiter.check_rate_limit(client_id, max_calls, time_frame):
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'retry_after': time_frame
                }), 429
            
            return f(*args, **kwargs)
        return wrapped
    return decorator


def validate_json_input():
    """Decorator to validate JSON input for API endpoints."""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            if not request.is_json:
                return jsonify({'error': 'Request must be JSON'}), 400
            
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'Empty JSON provided'}), 400
            except Exception as e:
                return jsonify({'error': f'Invalid JSON: {str(e)}'}), 400
            
            return f(*args, **kwargs)
        return wrapped
    return decorator


class TimedLRUCache:
    """Time-based LRU cache with expiration."""
    
    def __init__(self, max_size=128, expiration=3600):
        self.max_size = max_size
        self.expiration = expiration
        self.cache = {}
        self.insertion_times = {}
        self.access_times = {}
        self.lock = Lock()
    
    def get(self, key):
        """Get value from cache."""
        current_time = time.time()
        
        with self.lock:
            # Clean expired entries
            self._clean_expired(current_time)
            
            if key in self.cache:
                self.access_times[key] = current_time
                return self.cache[key]
            
            return None
    
    def set(self, key, value):
        """Set value in cache."""
        current_time = time.time()
        
        with self.lock:
            # Clean expired entries
            self._clean_expired(current_time)
            
            # Evict LRU if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
                del self.cache[lru_key]
                del self.insertion_times[lru_key]
                del self.access_times[lru_key]
            
            # Set new value
            self.cache[key] = value
            self.insertion_times[key] = current_time
            self.access_times[key] = current_time
    
    def _clean_expired(self, current_time):
        """Remove expired entries."""
        expired_keys = [
            k for k, t in self.insertion_times.items()
            if current_time - t > self.expiration
        ]
        for k in expired_keys:
            del self.cache[k]
            del self.insertion_times[k]
            del self.access_times[k]


def timed_lru_cache(max_size=128, expiration=3600):
    """
    Time-based LRU cache decorator.
    
    Args:
        max_size: Maximum cache size
        expiration: Entry lifetime in seconds
    """
    cache = TimedLRUCache(max_size, expiration)
    
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            # Create cache key
            key_parts = [f.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            key = ":".join(key_parts)
            
            # Check cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Compute and cache
            result = f(*args, **kwargs)
            cache.set(key, result)
            
            return result
        return wrapped
    return decorator


def allowed_file(filename):
    """Check if filename has an allowed extension."""
    if '.' not in filename:
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    allowed = current_app.config.get('ALLOWED_EXTENSIONS', set())
    return ext in allowed