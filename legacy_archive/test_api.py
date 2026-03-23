#!/usr/bin/env python3
"""Quick API test to verify fixes"""

import requests
import json

# Test the simple summarization endpoint
def test_simple_summarization():
    url = "http://localhost:5001/summarize"
    data = {
        "text": "The quick brown fox jumps over the lazy dog. This is a test of the summarization API. We want to make sure everything works properly after our bug fixes.",
        "density": "medium"
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Testing summarization API...")
    test_simple_summarization()