#!/usr/bin/env python3
"""
Test script for unlimited text processing
"""

import os
import requests
import json
import time

def create_large_text(size_mb):
    """Create a large text of specified size in MB."""
    # Each paragraph is roughly 100 bytes
    paragraph = "The quick brown fox jumps over the lazy dog. " * 2
    paragraphs_per_mb = (1024 * 1024) // len(paragraph)
    
    text = []
    for i in range(int(size_mb * paragraphs_per_mb)):
        text.append(f"Paragraph {i}: {paragraph}")
    
    return "\n\n".join(text)

def test_text_sizes():
    """Test different text sizes."""
    base_url = "http://localhost:5001"
    
    # Test sizes: 100KB, 1MB, 10MB, 100MB
    test_sizes = [
        (0.1, "100KB"),
        (1, "1MB"),
        (10, "10MB"),
        (100, "100MB")
    ]
    
    for size_mb, label in test_sizes:
        print(f"\nTesting {label} text...")
        
        # Create text
        text = create_large_text(size_mb)
        actual_size = len(text.encode('utf-8'))
        print(f"Created text of {actual_size:,} bytes")
        
        # Test direct text processing
        start_time = time.time()
        response = requests.post(
            f"{base_url}/api/process_text",
            json={
                "text": text,
                "model": "unlimited",
                "config": {
                    "max_summary_tokens": 500
                }
            }
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Success in {elapsed:.2f}s")
            print(f"  Processing method: {result.get('processing_method', 'unknown')}")
            print(f"  Chunks processed: {result.get('chunks_processed', 0)}")
            print(f"  Summary length: {len(result.get('summary', ''))}")
        else:
            print(f"✗ Failed with status {response.status_code}")
            print(f"  Error: {response.text}")

def test_file_upload():
    """Test file upload for large text."""
    base_url = "http://localhost:5001"
    
    # Create a 5MB test file
    print("\nTesting file upload (5MB)...")
    text = create_large_text(5)
    
    # Write to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(text)
        temp_path = f.name
    
    try:
        # Upload file
        start_time = time.time()
        with open(temp_path, 'rb') as f:
            response = requests.post(
                f"{base_url}/api/process_unlimited",
                files={'file': ('test.txt', f, 'text/plain')}
            )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Success in {elapsed:.2f}s")
            print(f"  Processing method: {result.get('processing_method', 'unknown')}")
            print(f"  Summary length: {len(result.get('summary', ''))}")
        else:
            print(f"✗ Failed with status {response.status_code}")
            print(f"  Error: {response.text}")
    finally:
        # Cleanup
        os.unlink(temp_path)

def test_streaming_chunks():
    """Test that chunking preserves context."""
    base_url = "http://localhost:5001"
    
    # Create text with specific pattern
    sections = []
    for i in range(10):
        section = f"Section {i}: This is important information about topic {i}. " * 50
        sections.append(section)
    
    text = "\n\n".join(sections)
    print(f"\nTesting context preservation with {len(sections)} sections...")
    
    response = requests.post(
        f"{base_url}/api/process_text",
        json={
            "text": text,
            "model": "unlimited",
            "config": {
                "max_summary_tokens": 1000
            }
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        summary = result.get('summary', '')
        
        # Check if all sections are represented
        sections_found = 0
        for i in range(10):
            if f"topic {i}" in summary or f"Section {i}" in summary:
                sections_found += 1
        
        print(f"✓ Found {sections_found}/10 sections in summary")
        print(f"  Chunk summaries included: {len(result.get('chunk_summaries', []))}")
    else:
        print(f"✗ Failed: {response.text}")

if __name__ == "__main__":
    print("Testing Unlimited Text Processing")
    print("=================================")
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:5001/api/health")
        if response.status_code != 200:
            print("Error: Server not responding. Please start the server first.")
            exit(1)
    except:
        print("Error: Cannot connect to server at http://localhost:5001")
        print("Please start the server with: python main.py")
        exit(1)
    
    # Run tests
    test_text_sizes()
    test_file_upload()
    test_streaming_chunks()
    
    print("\n✓ All tests completed!")