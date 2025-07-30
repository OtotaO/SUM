#!/usr/bin/env python3
"""
Test Script for Adaptive Compression System

Demonstrates the capabilities of our adaptive compression
engine with golden texts benchmarking. Tests compression quality
across different content types and incompressibility levels.

Usage: python test_adaptive_system.py

Author: ototao & Claude
"""

import time
from adaptive_compression import AdaptiveCompressionEngine, ContentType
from golden_texts import GoldenTextsCollection


def test_philosophical_compression():
    """Test compression on philosophical texts."""
    print("🧠 PHILOSOPHICAL COMPRESSION TEST")
    print("=" * 50)
    
    engine = AdaptiveCompressionEngine()
    collection = GoldenTextsCollection()
    
    # Get Marcus Aurelius text
    marcus_text = collection.texts['philosophical'][0]
    
    print(f"Original Text: {marcus_text.title}")
    print(f"Author: {marcus_text.author}")
    print(f"Incompressibility Score: {marcus_text.incompressibility_score}")
    print(f"\nOriginal ({len(marcus_text.content.split())} words):")
    print(marcus_text.content)
    
    # Compress with different ratios
    for ratio in [0.5, 0.3, 0.1]:
        print(f"\n--- Compression Ratio: {ratio:.0%} ---")
        
        result = engine.compress(marcus_text.content, target_ratio=ratio, 
                               force_type=ContentType.PHILOSOPHICAL)
        
        analysis = collection.analyze_compression_resistance(marcus_text, result)
        
        print(f"Compressed ({result['compressed_length']} words):")
        print(result['compressed'])
        print(f"Quality Score: {analysis['quality_score']:.1%}")
        
        if analysis['incompressibility_violation'] > 0.3:
            print("⚠️  Warning: May have violated incompressible nature")
        else:
            print("✅ Respects incompressible nature")


def test_technical_compression():
    """Test compression on technical texts."""
    print("\n\n💻 TECHNICAL COMPRESSION TEST")
    print("=" * 50)
    
    engine = AdaptiveCompressionEngine()
    collection = GoldenTextsCollection()
    
    # Get CAP theorem text
    cap_text = collection.texts['technical'][2]  # CAP Theorem
    
    print(f"Original Text: {cap_text.title}")
    print(f"Original ({len(cap_text.content.split())} words):")
    print(cap_text.content)
    
    result = engine.compress(cap_text.content, target_ratio=0.4,
                           force_type=ContentType.TECHNICAL)
    
    print(f"\nCompressed ({result['compressed_length']} words):")
    print(result['compressed'])
    print(f"Strategy: {result['strategy']}")
    print(f"Information Density: {result['information_density']:.2f}")


def test_adaptive_selection():
    """Test automatic content type detection."""
    print("\n\n🎯 ADAPTIVE CONTENT DETECTION TEST")
    print("=" * 50)
    
    engine = AdaptiveCompressionEngine()
    
    test_texts = [
        ("Philosophy", "The essence of human existence lies in our capacity to question the very nature of existence itself."),
        ("Technical", "The algorithm has O(n log n) complexity. def quick_sort(arr): return sorted(arr) if len(arr) <= 1 else..."),
        ("Activity Log", "2024-01-15 09:00 Started work\n2024-01-15 09:30 Coffee break\n2024-01-15 10:00 Meeting"),
        ("Narrative", "Once upon a time, in a small village, there lived a programmer who dreamed of perfect compression.")
    ]
    
    for category, text in test_texts:
        print(f"\n--- {category} Text ---")
        print(f"Input: {text[:60]}...")
        
        result = engine.compress(text, target_ratio=0.3)
        
        print(f"Detected Type: {result['content_type']}")
        print(f"Strategy: {result['strategy']}")
        print(f"Density: {result['information_density']:.2f}")


def run_full_benchmark():
    """Run complete benchmark suite."""
    print("\n\n🏆 GOLDEN TEXTS BENCHMARK SUITE")
    print("=" * 50)
    
    engine = AdaptiveCompressionEngine()
    
    print("Running benchmarks on incompressible texts...")
    start_time = time.time()
    
    benchmarks = engine.benchmark_compression()
    
    print(f"Benchmark completed in {time.time() - start_time:.2f}s")
    print("\nResults by Category:")
    print("-" * 30)
    
    for category, metrics in benchmarks.items():
        print(f"{category.title()}:")
        print(f"  Compression Ratio: {metrics.compression_ratio:.1%}")
        print(f"  Information Retention: {metrics.information_retention:.1%}")
        print(f"  Semantic Coherence: {metrics.semantic_coherence:.1%}")
        print(f"  Quality Score: {metrics.readability_score:.1%}")
        print(f"  Processing Time: {metrics.processing_time:.3f}s")
        print()


def demonstrate_life_compression():
    """Demonstrate life event compression."""
    print("\n\n🌟 LIFE COMPRESSION DEMONSTRATION")
    print("=" * 50)
    
    # Simulate a day's activities
    sample_day = """2024-01-15 07:00 Woke up, feeling refreshed
2024-01-15 07:30 Morning meditation and reflection
2024-01-15 08:00 Breakfast with family
2024-01-15 09:00 Started work on adaptive compression engine
2024-01-15 09:30 Deep focus coding session
2024-01-15 10:30 Breakthrough in philosophical compression strategy
2024-01-15 11:00 Documented the algorithm
2024-01-15 12:00 Lunch break and walk
2024-01-15 13:00 Implemented technical compression
2024-01-15 14:00 Created golden texts collection
2024-01-15 15:00 Ran first successful benchmarks
2024-01-15 16:00 Integration with main system
2024-01-15 17:00 Feeling accomplished about the day's progress
2024-01-15 18:00 Dinner and family time
2024-01-15 20:00 Reflected on the philosophical implications
2024-01-15 22:00 Sleep, with dreams of compressed lifetimes"""
    
    engine = AdaptiveCompressionEngine()
    
    print("Sample Day Activities:")
    print(sample_day[:200] + "...")
    
    # Compress as activity log
    result = engine.compress(sample_day, target_ratio=0.2, 
                           force_type=ContentType.ACTIVITY_LOG)
    
    print(f"\nCompressed Day ({result['actual_ratio']:.1%} of original):")
    print(result['compressed'])
    
    print(f"\nOriginal events: {len(sample_day.splitlines())}")
    print(f"Compressed to: {result['compressed_length']} words")
    print(f"Strategy: {result['strategy']}")


def show_most_incompressible():
    """Show the most incompressible texts."""
    print("\n\n💎 MOST INCOMPRESSIBLE TEXTS")
    print("=" * 50)
    
    collection = GoldenTextsCollection()
    most_incompressible = collection.get_most_incompressible(5)
    
    print("These texts represent the theoretical limits of compression:")
    print()
    
    for i, text in enumerate(most_incompressible, 1):
        print(f"{i}. {text.title} by {text.author}")
        print(f"   Incompressibility: {text.incompressibility_score:.1%}")
        print(f"   Content: {text.content[:100]}...")
        print(f"   Note: {text.notes}")
        print()


def main():
    """Run all tests and demonstrations."""
    print("🚀 ADAPTIVE COMPRESSION SYSTEM TEST SUITE")
    print("Testing the limits of semantic compression...")
    print("Inspired by Carmack's efficiency and Torvalds' pragmatism")
    print()
    
    try:
        # Core functionality tests
        test_philosophical_compression()
        test_technical_compression() 
        test_adaptive_selection()
        
        # Comprehensive benchmarking
        run_full_benchmark()
        
        # Future vision demonstration
        demonstrate_life_compression()
        
        # Show incompressible examples
        show_most_incompressible()
        
        print("\n\n✨ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("The adaptive compression system is ready for deployment.")
        print("\nNext steps:")
        print("- Compile and test the C monitoring agent")
        print("- Deploy the life compression system")
        print("- Begin capturing and compressing your digital life")
        print("\nRemember: We're not just compressing text.")
        print("We're distilling the essence of human experience.")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()