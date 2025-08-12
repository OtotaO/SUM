#!/usr/bin/env python3
"""
test_simple.py - Quick test to verify SUM works

Run this to see the simplified version in action!
"""

import requests
import time
import json

def test_sum_api():
    """Test the SUM API with various examples"""
    
    print("🧪 Testing SUM v2 API")
    print("=" * 50)
    
    # Test texts
    test_cases = [
        {
            "name": "Short Business Text",
            "text": """
            The company reported strong Q4 earnings with revenue up 23% year-over-year. 
            This growth was driven primarily by increased demand in cloud services and 
            enterprise software solutions. Operating margins improved to 35%, reflecting 
            operational efficiency gains. The CEO expressed optimism about future growth 
            prospects and announced plans to expand into new markets.
            """
        },
        {
            "name": "Technical Article",
            "text": """
            Machine learning models have revolutionized natural language processing. 
            Transformer architectures, particularly models like BERT and GPT, use 
            self-attention mechanisms to understand context in ways previous models couldn't. 
            These models are pre-trained on vast amounts of text data and can be fine-tuned 
            for specific tasks. The key innovation is the attention mechanism, which allows 
            the model to focus on relevant parts of the input when making predictions.
            This has led to breakthroughs in translation, summarization, and question answering.
            """
        },
        {
            "name": "News Article",
            "text": """
            Scientists have discovered a new exoplanet orbiting a distant star in the 
            constellation Cygnus. The planet, designated Kepler-452b, is located in the 
            habitable zone of its star, meaning liquid water could potentially exist on 
            its surface. With a radius 60% larger than Earth and a 385-day orbit, this 
            'super-Earth' represents one of the most Earth-like planets ever discovered. 
            The discovery was made using data from NASA's Kepler Space Telescope and 
            confirmed through ground-based observations. Researchers are excited about 
            the implications for finding life beyond our solar system.
            """
        }
    ]
    
    # API endpoint
    url = "http://localhost:3000/summarize"
    
    # Test each case
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test['name']}")
        print("-" * 40)
        
        # Clean up text (remove extra whitespace)
        text = " ".join(test['text'].split())
        
        # Make request
        try:
            start_time = time.time()
            response = requests.post(
                url,
                json={"text": text},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success! (took {elapsed:.2f}s)")
                print(f"📝 Summary: {data['summary']}")
                print(f"💾 Cached: {data.get('cached', False)}")
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Could not connect to API. Is it running?")
            print("   Run: python quickstart_local.py")
            return
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test caching
    print("\n" + "=" * 50)
    print("🔄 Testing Cache Performance")
    print("-" * 40)
    
    # Send same request twice
    test_text = "This is a test to see if caching works properly."
    
    # First request
    start = time.time()
    r1 = requests.post(url, json={"text": test_text})
    time1 = time.time() - start
    
    # Second request (should be cached)
    start = time.time()
    r2 = requests.post(url, json={"text": test_text})
    time2 = time.time() - start
    
    if r1.status_code == 200 and r2.status_code == 200:
        speedup = time1 / time2 if time2 > 0 else 999
        print(f"First request:  {time1:.3f}s (computed)")
        print(f"Second request: {time2:.3f}s (cached)")
        print(f"⚡ Cache speedup: {speedup:.1f}x faster!")
    
    # Test API endpoints
    print("\n" + "=" * 50)
    print("🔍 Testing Other Endpoints")
    print("-" * 40)
    
    # Health check
    try:
        r = requests.get("http://localhost:3000/health")
        if r.status_code == 200:
            print(f"✅ Health check: {r.json()}")
    except:
        print("❌ Health check failed")
    
    # Stats
    try:
        r = requests.get("http://localhost:3000/stats")
        if r.status_code == 200:
            print(f"✅ Stats: {r.json()}")
    except:
        print("❌ Stats check failed")
    
    print("\n" + "=" * 50)
    print("✨ Testing complete!")

if __name__ == "__main__":
    test_sum_api()