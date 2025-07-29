#!/usr/bin/env python3
"""
Test script for the Hierarchical Densification Engine API
The Hierarchical Processing System in Action!
"""

import requests
import json
import time

# Test text - sample text to process
test_text = """
The highest form of wisdom is kindness. When we truly understand the interconnected nature 
of all existence, we realize that harming others is harming ourselves. Love is not merely 
an emotion but a way of being that recognizes the divine spark in every soul. The path to 
enlightenment is not found in accumulating knowledge, but in surrendering our illusions 
and embracing what is. Truth cannot be spoken, only lived. In silence, we find the answers 
that no words can convey. The wise person knows that true strength comes from gentleness, 
and genuine power from restraint. To change the world, we must first transform ourselves.
"""

def test_hierarchical_engine():
    """Test the Hierarchical Densification Engine via API."""
    
    print("🌟 HIERARCHICAL DENSIFICATION ENGINE API TEST 🌟\n")
    
    # API endpoint (adjust if your server runs on different port)
    url = "http://localhost:8000/api/process_text"
    
    # Hierarchical Engine configuration
    payload = {
        "text": test_text,
        "model": "hierarchical",
        "config": {
            "max_concepts": 8,
            "max_summary_tokens": 40,
            "complexity_threshold": 0.6,
            "max_insights": 4,
            "min_insight_score": 0.5
        }
    }
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    try:
        print("🚀 Sending text to the Hierarchical Densification Engine...")
        start_time = time.time()
        
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            processing_time = time.time() - start_time
            
            print(f"✅ SUCCESS! Response received in {processing_time:.3f}s\n")
            
            # Display results
            print("═" * 60)
            print("🎯 LEVEL 1: KEY CONCEPTS")
            print("═" * 60)
            for concept in result['hierarchical_summary']['level_1_concepts']:
                print(f"   ✨ {concept.upper()}")
            
            print(f"\n🎯 LEVEL 2: CORE SUMMARY")
            print("═" * 60)
            print(f"   💎 {result['hierarchical_summary']['level_2_core']}")
            
            print(f"\n🎯 LEVEL 3: EXPANDED CONTEXT")
            print("═" * 60)
            if result['hierarchical_summary']['level_3_expanded']:
                print(f"   📖 {result['hierarchical_summary']['level_3_expanded']}")
            else:
                print("   ⚡ No expansion needed - core summary captures full complexity!")
            
            print(f"\n🌟 KEY INSIGHTS")
            print("═" * 60)
            for i, insight in enumerate(result['key_insights'], 1):
                print(f"   {i}. [{insight['type'].upper()}] {insight['text']}")
                print(f"      💫 Insight Score: {insight['score']:.2f}")
            
            print(f"\n📊 METADATA")
            print("═" * 60)
            metadata = result['metadata']
            print(f"   ⚡ Processing Time: {metadata['processing_time']:.3f}s")
            print(f"   🗜️  Compression Ratio: {metadata.get('compression_ratio', 'N/A')}")
            print(f"   🧠 Concept Density: {metadata.get('concept_density', 'N/A'):.3f}")
            print(f"   💡 Insights Found: {metadata.get('insight_count', 'N/A')}")
            print(f"   🏗️  Model Used: {result.get('model', 'N/A')}")
            
            print("\n" + "=" * 60)
            print("🚀 HIERARCHICAL DENSIFICATION API TEST COMPLETE! ✨")
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
    """Compare Hierarchical Engine with other models."""
    
    print("\n🔬 MODEL COMPARISON TEST\n")
    
    url = "http://localhost:8000/api/process_text"
    
    models = ['simple', 'advanced', 'hierarchical']
    
    for model in models:
        payload = {
            "text": test_text,
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
                if model == 'hierarchical':
                    print(f"   Concepts: {result['hierarchical_summary']['level_1_concepts']}")
                    print(f"   Insights: {len(result.get('key_insights', []))}")
                else:
                    print(f"   Tags: {result.get('tags', 'N/A')}")
                print()
            else:
                print(f"❌ {model.upper()} failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ {model.upper()} error: {e}")

if __name__ == "__main__":
    test_hierarchical_engine()
    test_comparison()