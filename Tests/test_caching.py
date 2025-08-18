"""
Test suite for smart caching functionality

Tests cache operations, performance, and memory management.
"""

import pytest
import time
import os
import tempfile
import shutil
from smart_cache import SmartCache, get_cache, cache_result


class TestSmartCache:
    """Test suite for SmartCache class."""
    
    @pytest.fixture
    def cache_dir(self):
        """Create a temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def cache(self, cache_dir):
        """Create a SmartCache instance with temporary directory."""
        return SmartCache(
            cache_dir=cache_dir,
            max_size_mb=10,
            default_ttl_hours=1
        )
    
    def test_cache_initialization(self, cache):
        """Test cache initialization."""
        assert cache.cache_dir.exists()
        assert cache.max_size_bytes == 10 * 1024 * 1024
        assert cache.default_ttl == 3600  # 1 hour in seconds
    
    def test_basic_put_get(self, cache):
        """Test basic put and get operations."""
        text = "This is a test text for caching."
        model = "simple"
        config = {"maxTokens": 50}
        result = {"summary": "Test summary", "tags": ["test"]}
        
        # Put item in cache
        cache.put(text, model, config, result, processing_time=0.1)
        
        # Get item from cache
        cached_result = cache.get(text, model, config)
        
        assert cached_result is not None
        assert cached_result['summary'] == "Test summary"
        assert cached_result['tags'] == ["test"]
    
    def test_cache_miss(self, cache):
        """Test cache miss."""
        result = cache.get("non-existent text", "simple", {})
        assert result is None
    
    def test_different_configs(self, cache):
        """Test that different configs create different cache entries."""
        text = "Same text"
        model = "simple"
        result1 = {"summary": "Summary 1"}
        result2 = {"summary": "Summary 2"}
        
        # Same text but different configs
        cache.put(text, model, {"maxTokens": 50}, result1, 0.1)
        cache.put(text, model, {"maxTokens": 100}, result2, 0.1)
        
        # Should get different results
        cached1 = cache.get(text, model, {"maxTokens": 50})
        cached2 = cache.get(text, model, {"maxTokens": 100})
        
        assert cached1['summary'] == "Summary 1"
        assert cached2['summary'] == "Summary 2"
    
    def test_ttl_expiration(self, cache):
        """Test TTL expiration."""
        # Create cache with very short TTL
        short_cache = SmartCache(
            cache_dir=cache.cache_dir,
            default_ttl_hours=0.0001  # ~0.36 seconds
        )
        
        text = "Expiring text"
        short_cache.put(text, "simple", {}, {"summary": "Test"}, 0.1)
        
        # Should be available immediately
        assert short_cache.get(text, "simple", {}) is not None
        
        # Wait for expiration
        time.sleep(0.5)
        
        # Should be expired
        assert short_cache.get(text, "simple", {}) is None
    
    def test_memory_cache(self, cache):
        """Test in-memory caching."""
        text = "Memory cached text"
        model = "simple"
        config = {}
        result = {"summary": "Memory test"}
        
        cache.put(text, model, config, result, 0.1)
        
        # First get should populate memory cache
        cached1 = cache.get(text, model, config)
        assert cached1 is not None
        
        # Check memory cache is populated
        cache_key = cache._get_cache_key(text, model, config)
        assert cache_key in cache.memory_cache
        
        # Second get should be from memory (faster)
        start = time.time()
        cached2 = cache.get(text, model, config)
        memory_time = time.time() - start
        
        assert cached2 is not None
        assert memory_time < 0.01  # Should be very fast
    
    def test_cache_stats(self, cache):
        """Test cache statistics."""
        # Add some entries
        for i in range(5):
            cache.put(f"Text {i}", "simple", {}, {"summary": f"Summary {i}"}, 0.1)
        
        stats = cache.get_stats()
        
        assert stats['total_entries'] == 5
        assert stats['memory_entries'] <= 5
        assert stats['total_size_mb'] > 0
        assert stats['max_size_mb'] == 10.0
    
    def test_clear_cache(self, cache):
        """Test cache clearing."""
        # Add entries
        cache.put("Text 1", "simple", {}, {"summary": "Summary 1"}, 0.1)
        cache.put("Text 2", "simple", {}, {"summary": "Summary 2"}, 0.1)
        
        # Clear cache
        cleared = cache.clear()
        
        assert cleared == 2
        assert cache.get("Text 1", "simple", {}) is None
        assert cache.get("Text 2", "simple", {}) is None
        assert len(cache.memory_cache) == 0
    
    def test_clear_by_pattern(self, cache):
        """Test clearing cache by pattern."""
        # Add entries
        cache.put("Hello world", "simple", {}, {"summary": "S1"}, 0.1)
        cache.put("Hello there", "simple", {}, {"summary": "S2"}, 0.1)
        cache.put("Goodbye", "simple", {}, {"summary": "S3"}, 0.1)
        
        # Clear entries containing "Hello"
        cleared = cache.clear(pattern="Hello")
        
        assert cleared == 2
        assert cache.get("Hello world", "simple", {}) is None
        assert cache.get("Hello there", "simple", {}) is None
        assert cache.get("Goodbye", "simple", {}) is not None
    
    def test_size_limit_enforcement(self, cache):
        """Test cache size limit enforcement."""
        # Create cache with tiny size limit
        tiny_cache = SmartCache(
            cache_dir=cache.cache_dir,
            max_size_mb=0.001  # 1KB
        )
        
        # Add large entry
        large_text = "X" * 10000  # 10KB
        tiny_cache.put(large_text, "simple", {}, {"summary": "Large"}, 0.1)
        
        # Should handle gracefully (might evict or refuse)
        stats = tiny_cache.get_stats()
        assert stats['total_size_mb'] <= 0.001 or stats['total_entries'] == 0
    
    def test_concurrent_access(self, cache):
        """Test concurrent cache access."""
        import threading
        
        results = []
        
        def cache_operation(i):
            cache.put(f"Text {i}", "simple", {}, {"summary": f"S{i}"}, 0.1)
            result = cache.get(f"Text {i}", "simple", {})
            results.append(result is not None)
        
        # Run concurrent operations
        threads = []
        for i in range(10):
            t = threading.Thread(target=cache_operation, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All operations should succeed
        assert all(results)
    
    def test_hit_rate_tracking(self, cache):
        """Test cache hit rate tracking."""
        # Add entry
        cache.put("Text", "simple", {}, {"summary": "S"}, 0.1)
        
        # Reset stats
        cache.hits = 0
        cache.misses = 0
        
        # Some hits
        for _ in range(7):
            cache.get("Text", "simple", {})
        
        # Some misses
        for _ in range(3):
            cache.get("Missing", "simple", {})
        
        stats = cache.get_stats()
        assert stats['recent_hits'] == 7
        assert stats['hit_rate'] == 0.7  # 7 hits / 10 total


class TestCacheDecorator:
    """Test cache_result decorator."""
    
    @pytest.fixture
    def cache(self):
        """Get cache instance."""
        return get_cache()
    
    def test_decorator_basic(self, cache):
        """Test basic decorator functionality."""
        call_count = 0
        
        @cache_result(model='test')
        def expensive_function(text):
            nonlocal call_count
            call_count += 1
            return {"result": f"Processed: {text}"}
        
        # First call
        result1 = expensive_function("Test text")
        assert result1["result"] == "Processed: Test text"
        assert call_count == 1
        
        # Second call should use cache
        result2 = expensive_function("Test text")
        assert result2["result"] == "Processed: Test text"
        assert call_count == 1  # Not called again
    
    def test_decorator_with_config(self, cache):
        """Test decorator with configuration."""
        @cache_result(model='test')
        def process_with_config(text, config=None):
            return {"result": f"Config: {config}"}
        
        # Different configs should cache separately
        result1 = process_with_config("Text", config={"a": 1})
        result2 = process_with_config("Text", config={"a": 2})
        
        assert result1["result"] == "Config: {'a': 1}"
        assert result2["result"] == "Config: {'a': 2}"


class TestGlobalCache:
    """Test global cache instance."""
    
    def test_get_cache_singleton(self):
        """Test that get_cache returns singleton."""
        cache1 = get_cache()
        cache2 = get_cache()
        
        assert cache1 is cache2
    
    def test_global_cache_operations(self):
        """Test operations on global cache."""
        cache = get_cache()
        
        # Clear first
        cache.clear()
        
        # Add entry
        cache.put("Global test", "simple", {}, {"summary": "Global"}, 0.1)
        
        # Retrieve
        result = cache.get("Global test", "simple", {})
        assert result is not None
        assert result["summary"] == "Global"


class TestCachePerformance:
    """Performance tests for caching."""
    
    @pytest.fixture
    def cache(self):
        """Create cache for performance testing."""
        return SmartCache(max_entries=1000)
    
    @pytest.mark.slow
    def test_large_cache_performance(self, cache):
        """Test performance with many entries."""
        # Add many entries
        start = time.time()
        for i in range(100):
            cache.put(f"Text {i}", "simple", {}, {"summary": f"S{i}"}, 0.01)
        put_time = time.time() - start
        
        # Retrieve entries
        start = time.time()
        for i in range(100):
            cache.get(f"Text {i}", "simple", {})
        get_time = time.time() - start
        
        # Performance assertions
        assert put_time < 1.0  # Should complete within 1 second
        assert get_time < 0.1  # Gets should be very fast
        
        print(f"Put 100 entries: {put_time:.3f}s")
        print(f"Get 100 entries: {get_time:.3f}s")
    
    def test_memory_efficiency(self, cache):
        """Test memory efficiency of cache."""
        import sys
        
        # Add large text
        large_text = "X" * 100000  # 100KB
        cache.put(large_text, "simple", {}, {"summary": "Large"}, 0.1)
        
        # Check memory usage is reasonable
        stats = cache.get_stats()
        
        # Cache should compress or limit storage
        assert stats['total_size_mb'] < 1.0  # Should be much less than raw text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])