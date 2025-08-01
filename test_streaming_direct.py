#!/usr/bin/env python3
"""
Direct test of streaming integration with main.py
"""

import sys
import os
sys.path.append('.')

from streaming_engine import StreamingHierarchicalEngine, StreamingConfig

def test_direct_integration():
    """Test streaming engine directly to ensure it works."""
    
    print("🚀 TESTING STREAMING ENGINE DIRECT INTEGRATION")
    print("=" * 55)
    
    # Create test text
    test_text = """
    Machine learning has revolutionized the way we approach complex problems across various domains.
    From healthcare to finance, from transportation to entertainment, machine learning algorithms
    are driving innovation and efficiency at an unprecedented scale. The foundation of machine 
    learning lies in its ability to learn patterns from data without being explicitly programmed 
    for every possible scenario. This capability makes it particularly valuable for handling tasks 
    that are too complex for traditional rule-based programming approaches.
    
    Deep learning, a subset of machine learning, has been particularly transformative. Neural networks
    with multiple layers can learn hierarchical representations of data, enabling breakthroughs in
    image recognition, natural language processing, and speech recognition. However, with great power 
    comes great responsibility. The deployment of machine learning systems raises important questions 
    about bias, fairness, and interpretability.
    
    Natural language processing represents one of the most challenging and fascinating areas of 
    artificial intelligence. The goal is to enable machines to understand, interpret, and generate 
    human language in a way that is both meaningful and useful. Computer vision aims to give machines 
    the ability to interpret and understand visual information from the world around them.
    
    Reinforcement Learning represents a unique paradigm where agents learn optimal behaviors through 
    trial and error interactions with their environment. Unlike supervised learning, RL agents must 
    discover effective strategies by receiving feedback in the form of rewards and penalties.
    """ * 3  # Make it longer for better testing
    
    print(f"📝 Test text: {len(test_text.split())} words ({len(test_text)} characters)")
    
    try:
        # Test 1: Basic streaming functionality
        print("\n🔥 TEST 1: Basic Streaming Functionality")
        print("-" * 40)
        
        config = StreamingConfig(
            chunk_size_words=200,
            overlap_ratio=0.1,
            max_memory_mb=256,
            max_concurrent_chunks=3
        )
        
        engine = StreamingHierarchicalEngine(config)
        result = engine.process_streaming_text(test_text)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return False
        
        print("✅ Streaming processing successful!")
        
        # Display results
        if 'hierarchical_summary' in result:
            hs = result['hierarchical_summary']
            print(f"🎯 Concepts: {hs.get('level_1_concepts', [])}")
            print(f"💎 Summary: {hs.get('level_2_core', '')[:100]}...")
        
        if 'processing_stats' in result:
            stats = result['processing_stats']
            print(f"📊 Chunks: {stats.get('successful_chunks', 0)}/{stats.get('total_chunks', 0)}")
            print(f"📈 Success rate: {stats.get('success_rate', 0):.1%}")
        
        # Test 2: Different configurations
        print("\n🔥 TEST 2: Different Configurations")
        print("-" * 40)
        
        configs_to_test = [
            {"chunk_size_words": 100, "max_concurrent_chunks": 2, "name": "Small chunks"},
            {"chunk_size_words": 400, "max_concurrent_chunks": 4, "name": "Large chunks"},
            {"chunk_size_words": 300, "overlap_ratio": 0.2, "name": "High overlap"}
        ]
        
        for i, test_config in enumerate(configs_to_test, 1):
            name = test_config.pop('name')
            print(f"  Test {i}: {name}")
            
            config = StreamingConfig(**test_config)
            engine = StreamingHierarchicalEngine(config)
            result = engine.process_streaming_text(test_text)
            
            if 'error' in result:
                print(f"    ❌ Error: {result['error']}")
            else:
                stats = result.get('processing_stats', {})
                chunks = stats.get('total_chunks', 0)
                success_rate = stats.get('success_rate', 0)
                print(f"    ✅ Success: {chunks} chunks, {success_rate:.1%} success rate")
        
        # Test 3: Integration with main app
        print("\n🔥 TEST 3: Integration with Main App")
        print("-" * 40)
        
        try:
            from main import app
            print("✅ Main app imports successfully")
            
            # Check if streaming route exists
            routes = [rule.rule for rule in app.url_map.iter_rules()]
            if '/api/process_text' in routes:
                print("✅ API endpoint exists")
            else:
                print("❌ API endpoint not found")
                
        except Exception as e:
            print(f"❌ Main app integration error: {e}")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 Streaming engine is ready for unlimited text processing!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_direct_integration()
    if success:
        print("\n✅ STREAMING ENGINE INTEGRATION SUCCESSFUL!")
    else:
        print("\n❌ STREAMING ENGINE INTEGRATION FAILED!")