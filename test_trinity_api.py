#!/usr/bin/env python3
"""
Test script for the Trinity Knowledge Engine API
The Cosmic Elevator in Action! 🚀✨
"""

import requests
import json
import time

# Test text - philosophical wisdom to process
test_wisdom = """
The highest form of wisdom is kindness. When we truly understand the interconnected nature 
of all existence, we realize that harming others is harming ourselves. Love is not merely 
an emotion but a way of being that recognizes the divine spark in every soul. The path to 
enlightenment is not found in accumulating knowledge, but in surrendering our illusions 
and embracing what is. Truth cannot be spoken, only lived. In silence, we find the answers 
that no words can convey. The wise person knows that true strength comes from gentleness, 
and genuine power from restraint. To change the world, we must first transform ourselves.
"""

def test_trinity_engine():
    """Test the Trinity Knowledge Engine via API."""
    
    print("🌟 TRINITY ENGINE API TEST 🌟\n")
    
    # API endpoint (adjust if your server runs on different port)
    url = "http://localhost:8000/api/process_text"
    
    # Trinity Engine configuration
    payload = {
        "text": test_wisdom,
        "model": "trinity",
        "config": {
            "max_wisdom_tags": 8,
            "essence_max_tokens": 40,
            "complexity_threshold": 0.6,
            "max_revelations": 4,
            "min_revelation_score": 0.5
        }
    }
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    try:
        print("🚀 Sending wisdom to the Cosmic Elevator...")
        start_time = time.time()
        
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            processing_time = time.time() - start_time
            
            print(f"✅ SUCCESS! Response received in {processing_time:.3f}s\n")
            
            # Display Trinity results
            print("═" * 60)
            print("🎯 LEVEL 1: WISDOM TAGS (Crystallized Concepts)")
            print("═" * 60)
            for tag in result['trinity']['level_1_tags']:
                print(f"   ✨ {tag.upper()}")
            
            print(f"\n🎯 LEVEL 2: ESSENCE (Complete Minimal Summary)")
            print("═" * 60)
            print(f"   💎 {result['trinity']['level_2_essence']}")
            
            print(f"\n🎯 LEVEL 3: CONTEXT (Intelligent Expansion)")
            print("═" * 60)
            if result['trinity']['level_3_context']:
                print(f"   📖 {result['trinity']['level_3_context']}")
            else:
                print("   ⚡ No expansion needed - essence captures full complexity!")
            
            print(f"\n🌟 REVELATIONS (Profound Insights)")
            print("═" * 60)
            for i, revelation in enumerate(result['revelations'], 1):
                print(f"   {i}. [{revelation['type'].upper()}] {revelation['text']}")
                print(f"      💫 Revelation Score: {revelation['score']:.2f}")
            
            print(f"\n📊 METADATA")
            print("═" * 60)
            metadata = result['metadata']
            print(f"   ⚡ Processing Time: {metadata['processing_time']:.3f}s")
            print(f"   🗜️  Compression Ratio: {metadata.get('compression_ratio', 'N/A')}")
            print(f"   🧠 Wisdom Density: {metadata.get('wisdom_density', 'N/A'):.3f}")
            print(f"   💡 Revelations Found: {metadata.get('revelation_count', 'N/A')}")
            print(f"   🏗️  Model Used: {result.get('model', 'N/A')}")
            
            print("\n" + "=" * 60)
            print("🚀 COSMIC ELEVATOR API TEST COMPLETE! ✨")
            print("=" * 60)
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the SUM server is running on localhost:3000")
        print("   Start it with: python main.py")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_comparison():
    """Compare Trinity Engine with other models."""
    
    print("\n🔬 MODEL COMPARISON TEST\n")
    
    url = "http://localhost:8000/api/process_text"
    
    models = ['simple', 'advanced', 'trinity']
    
    for model in models:
        payload = {
            "text": test_wisdom,
            "model": model,
            "config": {
                "maxTokens": 40,
                "include_analysis": True if model == 'advanced' else False
            }
        }
        
        try:
            response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
            
            if response.status_code == 200:
                result = response.json()
                print(f"🤖 {model.upper()} ENGINE:")
                print(f"   Summary: {result.get('summary', 'N/A')}")
                if model == 'trinity':
                    print(f"   Tags: {result['trinity']['level_1_tags']}")
                    print(f"   Revelations: {len(result.get('revelations', []))}")
                else:
                    print(f"   Tags: {result.get('tags', 'N/A')}")
                print()
            else:
                print(f"❌ {model.upper()} failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ {model.upper()} error: {e}")

if __name__ == "__main__":
    test_trinity_engine()
    test_comparison()