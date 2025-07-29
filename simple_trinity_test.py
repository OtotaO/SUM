#!/usr/bin/env python3
"""
Simple Trinity Engine Test - Direct API call
Let's test the cosmic elevator directly! 🚀
"""

from SUM import TrinityKnowledgeEngine

def test_trinity_direct():
    """Test Trinity Engine directly without API."""
    
    print("🌟 DIRECT TRINITY ENGINE TEST 🌟\n")
    
    # Philosophical test text
    wisdom_text = """
    The nature of wisdom is paradoxical: the more we know, the more we realize we don't know. 
    True understanding comes not from accumulating facts, but from recognizing the interconnected 
    nature of all things. Love is the highest form of knowledge, for it sees beyond appearances 
    to the essence of being. In silence, we find answers that words cannot express.
    """
    
    try:
        # Initialize Trinity Engine
        print("🚀 Initializing Trinity Knowledge Engine...")
        trinity = TrinityKnowledgeEngine()
        
        # Configure for optimal wisdom extraction
        config = {
            'max_wisdom_tags': 6,
            'essence_max_tokens': 35,
            'complexity_threshold': 0.6,
            'max_revelations': 3,
            'min_revelation_score': 0.5
        }
        
        print("⚡ Processing wisdom through the cosmic elevator...")
        result = trinity.process_text(wisdom_text, config)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return
        
        # Display beautiful results
        print("\n" + "="*60)
        print("🎯 TRINITY KNOWLEDGE DENSIFICATION RESULTS")
        print("="*60)
        
        print("\n📋 LEVEL 1: WISDOM TAGS (Crystallized Concepts)")
        print("-" * 40)
        for i, tag in enumerate(result['trinity']['level_1_tags'], 1):
            print(f"  {i}. ✨ {tag.upper()}")
        
        print(f"\n📜 LEVEL 2: ESSENCE (Complete Minimal Summary)")
        print("-" * 40)
        print(f"  💎 {result['trinity']['level_2_essence']}")
        
        print(f"\n📖 LEVEL 3: CONTEXT (Intelligent Expansion)")
        print("-" * 40)
        if result['trinity']['level_3_context']:
            print(f"  📖 {result['trinity']['level_3_context']}")
        else:
            print("  ⚡ No expansion needed - essence captures full complexity!")
        
        print(f"\n🌟 REVELATIONS (Profound Insights)")
        print("-" * 40)
        for i, rev in enumerate(result['revelations'], 1):
            print(f"  {i}. [{rev['type'].upper()}] {rev['text']}")
            print(f"     💫 Score: {rev['score']:.2f}")
        
        print(f"\n📊 PERFORMANCE METRICS")
        print("-" * 40)
        meta = result['metadata']
        print(f"  ⚡ Processing Time: {meta['processing_time']:.3f}s")
        print(f"  🗜️  Compression: {meta['compression_ratio']:.2f}")
        print(f"  🧠 Wisdom Density: {meta['wisdom_density']:.3f}")
        print(f"  💡 Revelations: {meta['revelation_count']}")
        
        print("\n" + "="*60)
        print("🚀 COSMIC ELEVATOR TEST COMPLETE! ✨")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during Trinity Engine test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_trinity_direct()
    if success:
        print("\n🎉 Trinity Engine is working perfectly!")
        print("Ready for integration into your agent ecosystem! 🤖")
    else:
        print("\n💔 Trinity Engine encountered issues.")
        print("Debug needed before agent integration.")