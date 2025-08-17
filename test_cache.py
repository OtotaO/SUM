#!/usr/bin/env python3
"""
Test script for smart caching functionality
"""

import requests
import time
import json

BASE_URL = "http://localhost:5001"

def test_cache_performance():
    """Test cache performance improvement."""
    print("\n=== Testing Cache Performance ===")
    
    # Test text
    test_text = """
    The concept of artificial intelligence has evolved significantly over the past decades. 
    From simple rule-based systems to complex neural networks, AI has transformed how we 
    approach problem-solving. Machine learning algorithms now power recommendation systems, 
    natural language processing, and computer vision applications. Deep learning has enabled 
    breakthroughs in areas previously thought impossible for computers to master.
    """ * 10  # Make it longer
    
    # First request (cache miss)
    print("\n1. First request (cache miss):")
    start = time.time()
    response = requests.post(f"{BASE_URL}/api/process_text", json={
        "text": test_text,
        "model": "hierarchical",
        "config": {
            "max_summary_tokens": 200,
            "use_cache": True
        }
    })
    first_time = time.time() - start
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Success in {first_time:.3f}s")
        print(f"   Cached: {result.get('cached', False)}")
        print(f"   Summary length: {len(result.get('summary', ''))}")
    else:
        print(f"   ✗ Failed: {response.status_code}")
        return
    
    # Second request (cache hit)
    print("\n2. Second request (cache hit):")
    start = time.time()
    response = requests.post(f"{BASE_URL}/api/process_text", json={
        "text": test_text,
        "model": "hierarchical",
        "config": {
            "max_summary_tokens": 200,
            "use_cache": True
        }
    })
    second_time = time.time() - start
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Success in {second_time:.3f}s")
        print(f"   Cached: {result.get('cached', False)}")
        print(f"   Speedup: {first_time/second_time:.1f}x faster")
    else:
        print(f"   ✗ Failed: {response.status_code}")
        
def test_cache_variations():
    """Test cache with different configurations."""
    print("\n=== Testing Cache Variations ===")
    
    test_text = "This is a test of the caching system with different configurations."
    
    # Different configurations
    configs = [
        {"max_summary_tokens": 100},
        {"max_summary_tokens": 200},
        {"max_summary_tokens": 100, "threshold": 0.5}
    ]
    
    for i, config in enumerate(configs):
        print(f"\n{i+1}. Config: {json.dumps(config)}")
        response = requests.post(f"{BASE_URL}/api/process_text", json={
            "text": test_text,
            "model": "basic",
            "config": {**config, "use_cache": True}
        })
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Cached: {result.get('cached', False)}")
        else:
            print(f"   ✗ Failed: {response.status_code}")
            
def test_cache_stats():
    """Test cache statistics endpoint."""
    print("\n=== Testing Cache Statistics ===")
    
    response = requests.get(f"{BASE_URL}/api/cache/stats")
    
    if response.status_code == 200:
        stats = response.json()
        print(f"✓ Cache Statistics:")
        print(f"  Total entries: {stats.get('total_entries', 0)}")
        print(f"  Memory entries: {stats.get('memory_entries', 0)}")
        print(f"  Total size: {stats.get('total_size_mb', 0):.2f} MB")
        print(f"  Recent hits: {stats.get('recent_hits', 0)}")
    else:
        print(f"✗ Failed to get stats: {response.status_code}")
        
def test_cache_clear():
    """Test cache clearing."""
    print("\n=== Testing Cache Clear ===")
    
    # Add something to cache first
    test_text = "This text will be cached and then cleared."
    requests.post(f"{BASE_URL}/api/process_text", json={
        "text": test_text,
        "model": "basic",
        "config": {"use_cache": True}
    })
    
    # Clear cache
    response = requests.post(f"{BASE_URL}/api/cache/clear", json={})
    
    if response.status_code == 200:
        print("✓ Cache cleared successfully")
        
        # Verify it's cleared by checking if next request is not cached
        response = requests.post(f"{BASE_URL}/api/process_text", json={
            "text": test_text,
            "model": "basic",
            "config": {"use_cache": True}
        })
        
        if response.status_code == 200:
            result = response.json()
            if not result.get('cached', False):
                print("✓ Verified: Cache was cleared (next request not cached)")
            else:
                print("✗ Error: Request was still cached after clear")
    else:
        print(f"✗ Failed to clear cache: {response.status_code}")

def test_large_text_caching():
    """Test caching with large texts."""
    print("\n=== Testing Large Text Caching ===")
    
    # Create a 500KB text
    large_text = "The future of technology is rapidly evolving. " * 10000
    
    print(f"Text size: {len(large_text):,} bytes")
    
    # First request
    print("\n1. First request (cache miss):")
    start = time.time()
    response = requests.post(f"{BASE_URL}/api/process_text", json={
        "text": large_text,
        "model": "unlimited",
        "config": {"use_cache": True}
    })
    first_time = time.time() - start
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Success in {first_time:.3f}s")
        print(f"   Processing method: {result.get('processing_method', 'unknown')}")
    else:
        print(f"   ✗ Failed: {response.status_code}")
        return
    
    # Second request (should be cached)
    print("\n2. Second request (cache hit):")
    start = time.time()
    response = requests.post(f"{BASE_URL}/api/process_text", json={
        "text": large_text,
        "model": "unlimited",
        "config": {"use_cache": True}
    })
    second_time = time.time() - start
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Success in {second_time:.3f}s")
        print(f"   Cached: {result.get('cached', False)}")
        print(f"   Speedup: {first_time/second_time:.1f}x faster")
    else:
        print(f"   ✗ Failed: {response.status_code}")

if __name__ == "__main__":
    print("Smart Cache Testing")
    print("==================")
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        if response.status_code != 200:
            print("Error: Server not responding")
            exit(1)
    except:
        print("Error: Cannot connect to server")
        print("Please start the server with: python main.py")
        exit(1)
    
    # Run tests
    test_cache_performance()
    test_cache_variations()
    test_cache_stats()
    test_large_text_caching()
    test_cache_clear()  # Run this last as it clears the cache
    
    print("\n✓ All cache tests completed!")