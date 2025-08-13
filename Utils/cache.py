"""
cache.py - Simple Caching Layer for SUM

Provides a lightweight caching solution for:
- API responses
- Embeddings
- Processed results
- Temporary data

Author: SUM Development Team
License: Apache License 2.0
"""

import time
import json
import hashlib
import logging
from typing import Any, Optional, Dict, Callable
from functools import wraps
from threading import Lock
import pickle
import os

logger = logging.getLogger(__name__)


class SimpleCache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of items to cache
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = Lock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check if expired
                if time.time() > entry['expires_at']:
                    del self.cache[key]
                    del self.access_times[key]
                    self.misses += 1
                    return None
                
                # Update access time
                self.access_times[key] = time.time()
                self.hits += 1
                return entry['value']
            
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        with self.lock:
            # Evict old entries if at capacity
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Store value
            ttl = ttl or self.default_ttl
            self.cache[key] = {
                'value': value,
                'expires_at': time.time() + ttl,
                'created_at': time.time()
            }
            self.access_times[key] = time.time()
            return True
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                del self.access_times[key]
                return True
            return False
    
    def clear(self) -> int:
        """Clear all cache entries."""
        with self.lock:
            count = len(self.cache)
            self.cache.clear()
            self.access_times.clear()
            return count
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self.access_times:
            return
        
        # Find LRU key
        lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': round(hit_rate, 2),
                'total_requests': total_requests
            }


class FileCache:
    """File-based cache for larger objects."""
    
    def __init__(self, cache_dir: str = "./cache", max_size_mb: int = 100):
        """
        Initialize file cache.
        
        Args:
            cache_dir: Directory for cache files
            max_size_mb: Maximum total cache size in MB
        """
        self.cache_dir = cache_dir
        self.max_size_mb = max_size_mb
        self.metadata_file = os.path.join(cache_dir, ".metadata.json")
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load metadata
        self._load_metadata()
    
    def _load_metadata(self):
        """Load cache metadata from disk."""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
        except Exception as e:
            logger.error(f"Error loading cache metadata: {e}")
            self.metadata = {}
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {e}")
    
    def _get_cache_path(self, key: str) -> str:
        """Get file path for cache key."""
        # Hash key to create filename
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.cache")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from file cache."""
        if key not in self.metadata:
            return None
        
        # Check expiration
        entry = self.metadata[key]
        if time.time() > entry['expires_at']:
            self.delete(key)
            return None
        
        # Load from file
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading from cache: {e}")
            self.delete(key)
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in file cache."""
        cache_path = self._get_cache_path(key)
        
        try:
            # Serialize to file
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            
            # Update metadata
            file_size = os.path.getsize(cache_path)
            self.metadata[key] = {
                'expires_at': time.time() + ttl,
                'created_at': time.time(),
                'size_bytes': file_size
            }
            self._save_metadata()
            
            # Check total size and evict if needed
            self._check_size_limit()
            
            return True
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from file cache."""
        if key not in self.metadata:
            return False
        
        cache_path = self._get_cache_path(key)
        try:
            if os.path.exists(cache_path):
                os.remove(cache_path)
            del self.metadata[key]
            self._save_metadata()
            return True
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False
    
    def _check_size_limit(self):
        """Check and enforce size limit."""
        total_size = sum(entry['size_bytes'] for entry in self.metadata.values())
        max_size_bytes = self.max_size_mb * 1024 * 1024
        
        if total_size > max_size_bytes:
            # Sort by creation time and remove oldest
            sorted_keys = sorted(
                self.metadata.items(),
                key=lambda x: x[1]['created_at']
            )
            
            while total_size > max_size_bytes and sorted_keys:
                key, entry = sorted_keys.pop(0)
                self.delete(key)
                total_size -= entry['size_bytes']


# Global cache instances
_memory_cache = None
_file_cache = None


def get_memory_cache() -> SimpleCache:
    """Get or create memory cache instance."""
    global _memory_cache
    if _memory_cache is None:
        _memory_cache = SimpleCache()
    return _memory_cache


def get_file_cache() -> FileCache:
    """Get or create file cache instance."""
    global _file_cache
    if _file_cache is None:
        from config import Config
        cache_dir = os.path.join(Config.BASE_DIR, '.cache')
        _file_cache = FileCache(cache_dir=cache_dir)
    return _file_cache


# Decorators for caching
def cache_result(ttl: int = 3600, key_func: Optional[Callable] = None):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time-to-live in seconds
        key_func: Function to generate cache key from arguments
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)
            
            # Check cache
            cache = get_memory_cache()
            result = cache.get(cache_key)
            
            if result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return result
            
            # Call function
            logger.debug(f"Cache miss for {func.__name__}")
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result, ttl)
            
            return result
        
        # Add cache management methods
        wrapper.clear_cache = lambda: get_memory_cache().clear()
        wrapper.cache_stats = lambda: get_memory_cache().get_stats()
        
        return wrapper
    return decorator


def cache_embedding(embedding_func):
    """Decorator specifically for caching embeddings."""
    @wraps(embedding_func)
    def wrapper(text: str, *args, **kwargs):
        # Generate cache key from text hash
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        cache_key = f"embedding:{text_hash}"
        
        # Try memory cache first
        cache = get_memory_cache()
        embedding = cache.get(cache_key)
        
        if embedding is not None:
            return embedding
        
        # Try file cache for larger embeddings
        file_cache = get_file_cache()
        embedding = file_cache.get(cache_key)
        
        if embedding is not None:
            # Store in memory cache for faster access
            cache.set(cache_key, embedding, ttl=3600)
            return embedding
        
        # Generate embedding
        embedding = embedding_func(text, *args, **kwargs)
        
        # Store in both caches
        cache.set(cache_key, embedding, ttl=3600)
        file_cache.set(cache_key, embedding, ttl=86400)  # 24 hours
        
        return embedding
    
    return wrapper