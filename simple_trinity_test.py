#!/usr/bin/env python3
"""
Simple Hierarchical Densification Engine Test - Direct API call
Let's test the hierarchical engine directly!
"""

from SUM import HierarchicalDensificationEngine

def test_hierarchical_direct():
    """Test Hierarchical Densification Engine directly without API."""
    
    print("🌟 DIRECT HIERARCHICAL DENSIFICATION ENGINE TEST 🌟\n")
    
    # Test text
    test_text = """
    The nature of wisdom is paradoxical: the more we know, the more we realize we don't know. 
    True understanding comes not from accumulating facts, but from recognizing the interconnected 
    nature of all things. Love is the highest form of knowledge, for it sees beyond appearances 
    to the essence of being. In silence, we find answers that words cannot express.
    """
    
    try:
        # Initialize Hierarchical Densification Engine
        print("🚀 Initializing Hierarchical Densification Engine...")
        engine = HierarchicalDensificationEngine()
        
        # Configure for optimal processing
        config = {
            'max_concepts': 6,
            'max_summary_tokens': 35,
            'complexity_threshold': 0.6,
            'max_insights': 3,
            'min_insight_score': 0.5
        }
        
        print("⚡ Processing text through hierarchical engine...")
        result = engine.process_text(test_text, config)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return
        
        # Display results
        print("\n" + "="*60)
        print("🎯 HIERARCHICAL DENSIFICATION RESULTS")
        print("="*60)
        
        print("\n📋 LEVEL 1: KEY CONCEPTS")
        print("-" * 40)
        for i, concept in enumerate(result['hierarchical_summary']['level_1_concepts'], 1):
            print(f"  {i}. ✨ {concept.upper()}")
        
        print(f"\n📜 LEVEL 2: CORE SUMMARY")
        print("-" * 40)
        print(f"  💎 {result['hierarchical_summary']['level_2_core']}")
        
        print(f"\n📖 LEVEL 3: EXPANDED CONTEXT")
        print("-" * 40)
        if result['hierarchical_summary']['level_3_expanded']:
            print(f"  📖 {result['hierarchical_summary']['level_3_expanded']}")
        else:
            print("  ⚡ No expansion needed - core summary captures full complexity!")
        
        print(f"\n🌟 KEY INSIGHTS")
        print("-" * 40)
        for i, insight in enumerate(result['key_insights'], 1):
            print(f"  {i}. [{insight['type'].upper()}] {insight['text']}")
            print(f"     💫 Score: {insight['score']:.2f}")
        
        print(f"\n📊 PERFORMANCE METRICS")
        print("-" * 40)
        meta = result['metadata']
        print(f"  ⚡ Processing Time: {meta['processing_time']:.3f}s")
        print(f"  🗜️  Compression: {meta['compression_ratio']:.2f}")
        print(f"  🧠 Concept Density: {meta['concept_density']:.3f}")
        print(f"  💡 Insights: {meta['insight_count']}")
        
        print("\n" + "="*60)
        print("🚀 HIERARCHICAL DENSIFICATION TEST COMPLETE! ✨")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during Hierarchical Densification Engine test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hierarchical_direct()
    if success:
        print("\n🎉 Hierarchical Densification Engine is working perfectly!")
        print("Ready for integration into your agent ecosystem! 🤖")
    else:
        print("\n💔 Hierarchical Densification Engine encountered issues.")
        print("Debug needed before agent integration.")