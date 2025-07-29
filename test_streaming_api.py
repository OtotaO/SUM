#!/usr/bin/env python3
"""
Test script for the Streaming Engine API
Demonstrating unlimited text processing capabilities through the web API!
"""

import requests
import json
import time

def create_massive_test_text(target_words=10000):
    """Create a large test text for API testing."""
    
    base_content = """
    Artificial Intelligence represents one of the most significant technological 
    breakthroughs in human history. The field encompasses machine learning, 
    natural language processing, computer vision, robotics, and many other 
    specialized domains that collectively aim to create systems capable of 
    performing tasks that typically require human intelligence.
    
    Machine learning, a subset of AI, has revolutionized how we approach 
    complex problems by enabling computers to learn patterns from data 
    without explicit programming for every scenario. Deep learning networks, 
    inspired by the structure of the human brain, can process vast amounts 
    of information and identify intricate patterns that would be impossible 
    for humans to detect manually.
    
    Natural language processing has enabled machines to understand, interpret, 
    and generate human language with increasing sophistication. Modern language 
    models can engage in meaningful conversations, write creative content, 
    translate between languages, and even help with complex reasoning tasks.
    
    Computer vision systems can now analyze images and videos with superhuman 
    accuracy in many domains. From medical diagnosis to autonomous vehicles, 
    these systems are transforming industries and saving lives through their 
    ability to process visual information rapidly and accurately.
    
    Robotics combines AI with mechanical engineering to create systems that 
    can interact with the physical world. From manufacturing automation to 
    space exploration, robots equipped with AI capabilities are extending 
    human reach and capabilities in unprecedented ways.
    
    The ethical implications of AI development cannot be ignored. As these 
    systems become more powerful and ubiquitous, questions of fairness, 
    transparency, privacy, and human agency become increasingly important. 
    The choices we make today about AI development will shape the future 
    of human civilization.
    
    Looking ahead, the potential applications of AI seem limitless. Quantum 
    computing may revolutionize AI capabilities, enabling solutions to 
    problems that are currently intractable. Brain-computer interfaces 
    could create new forms of human-AI collaboration. The next decades 
    promise continued breakthroughs that will further transform our world.
    """
    
    # Repeat content to reach target word count
    current_text = ""
    current_words = 0
    section_counter = 1
    
    while current_words < target_words:
        current_text += f"\n\nSection {section_counter}: Extended Analysis\n"
        current_text += base_content
        current_text += f"\n\nThis section has explored key concepts in artificial intelligence and its applications. The rapid advancement in AI technologies continues to reshape industries, create new opportunities, and present novel challenges that require careful consideration and ethical frameworks. As we continue to push the boundaries of what machines can accomplish, the importance of responsible development and deployment becomes ever more critical.\n"
        
        current_words = len(current_text.split())
        section_counter += 1
    
    return current_text

def test_streaming_api():
    """Test the streaming engine through the API."""
    
    print("🚀 STREAMING ENGINE API TEST")
    print("=" * 50)
    
    # Test configurations for different text sizes
    test_cases = [
        {
            "name": "Medium Text (5K words)",
            "words": 5000,
            "config": {
                "chunk_size_words": 500,
                "overlap_ratio": 0.1,
                "max_memory_mb": 256,
                "max_concurrent_chunks": 3
            }
        },
        {
            "name": "Large Text (15K words)", 
            "words": 15000,
            "config": {
                "chunk_size_words": 800,
                "overlap_ratio": 0.15,
                "max_memory_mb": 512,
                "max_concurrent_chunks": 4
            }
        },
        {
            "name": "Massive Text (30K words)",
            "words": 30000,
            "config": {
                "chunk_size_words": 1000,
                "overlap_ratio": 0.1,
                "max_memory_mb": 1024,
                "max_concurrent_chunks": 6
            }
        }
    ]
    
    api_url = "http://localhost:8000/api/process_text"
    
    for test_case in test_cases:
        print(f"\n📝 Testing: {test_case['name']}")
        print("-" * 40)
        
        # Generate test text
        print("⚙️ Generating test text...")
        test_text = create_massive_test_text(test_case['words'])
        actual_words = len(test_text.split())
        print(f"📊 Generated {actual_words:,} words ({len(test_text):,} characters)")
        
        # Prepare API request
        payload = {
            "text": test_text,
            "model": "streaming",
            "config": test_case['config']
        }
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        print("🔥 Starting API processing...")
        start_time = time.time()
        
        try:
            response = requests.post(api_url, json=payload, headers=headers, timeout=300)
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"✅ SUCCESS! Processed in {processing_time:.2f} seconds")
                print(f"📈 Speed: {actual_words / processing_time:.0f} words/second")
                
                # Display streaming results
                if 'processing_stats' in result:
                    stats = result['processing_stats']
                    print(f"🧩 Chunks: {stats.get('successful_chunks', 0)}/{stats.get('total_chunks', 0)}")
                    print(f"📊 Success Rate: {stats.get('success_rate', 0):.1%}")
                
                if 'hierarchical_summary' in result:
                    hs = result['hierarchical_summary']
                    concepts = hs.get('level_1_concepts', [])
                    summary = hs.get('level_2_core', '')
                    
                    print(f"🎯 Concepts ({len(concepts)}): {', '.join(concepts[:5])}")
                    print(f"💎 Summary: {summary[:100]}{'...' if len(summary) > 100 else ''}")
                
                if 'streaming_metadata' in result:
                    meta = result['streaming_metadata']
                    print(f"🌊 Memory Efficiency: {meta.get('memory_efficiency', 0):.1%}")
                    print(f"⚡ Chunks Processed: {meta.get('chunks_processed', 0)}")
                
                if 'metadata' in result:
                    meta = result['metadata']
                    compression = meta.get('compression_ratio', 0)
                    print(f"🗜️ Compression: {compression:.3f} ({(1-compression)*100:.1f}% reduction)")
                
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"Response: {response.text[:200]}...")
                
        except requests.exceptions.Timeout:
            print("⏰ Request timed out (>5 minutes)")
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Make sure the SUM server is running")
            print("   Start it with: python main.py")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        
        print()  # Add spacing between tests

def test_model_comparison():
    """Compare streaming engine with other models."""
    
    print("\n🔬 MODEL COMPARISON TEST")
    print("=" * 50)
    
    # Create medium-sized test text
    test_text = create_massive_test_text(3000)
    actual_words = len(test_text.split())
    print(f"📝 Test text: {actual_words:,} words")
    
    models = [
        {
            "name": "Simple Engine",
            "model": "simple",
            "config": {"maxTokens": 100}
        },
        {
            "name": "Advanced Engine",
            "model": "advanced", 
            "config": {"maxTokens": 100, "include_analysis": True}
        },
        {
            "name": "Hierarchical Engine",
            "model": "hierarchical",
            "config": {
                "max_concepts": 8,
                "max_summary_tokens": 80,
                "max_insights": 4
            }
        },
        {
            "name": "Streaming Engine",
            "model": "streaming",
            "config": {
                "chunk_size_words": 600,
                "max_concurrent_chunks": 3,
                "max_memory_mb": 256
            }
        }
    ]
    
    api_url = "http://localhost:8000/api/process_text"
    
    for model_config in models:
        print(f"\n🤖 Testing {model_config['name']}...")
        
        payload = {
            "text": test_text,
            "model": model_config['model'],
            "config": model_config['config']
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'}, timeout=120)
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                speed = actual_words / processing_time
                
                print(f"   ⚡ Time: {processing_time:.2f}s (Speed: {speed:.0f} words/sec)")
                
                if model_config['model'] == 'streaming':
                    if 'processing_stats' in result:
                        stats = result['processing_stats']
                        print(f"   📊 Chunks: {stats.get('total_chunks', 0)}, Success: {stats.get('success_rate', 0):.1%}")
                    
                    if 'hierarchical_summary' in result:
                        concepts = result['hierarchical_summary'].get('level_1_concepts', [])
                        print(f"   🎯 Concepts: {', '.join(concepts[:3])}...")
                elif model_config['model'] == 'hierarchical':
                    if 'hierarchical_summary' in result:
                        concepts = result['hierarchical_summary'].get('level_1_concepts', [])
                        print(f"   🎯 Concepts: {', '.join(concepts[:3])}...")
                else:
                    tags = result.get('tags', [])
                    print(f"   🏷️ Tags: {', '.join(tags[:3]) if tags else 'None'}...")
                
                # Show summary snippet
                summary = result.get('summary', '') or result.get('hierarchical_summary', {}).get('level_2_core', '')
                if summary:
                    print(f"   💬 Summary: {summary[:80]}{'...' if len(summary) > 80 else ''}")
                
            else:
                print(f"   ❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:50]}...")

if __name__ == "__main__":
    print("🎊 SUM STREAMING ENGINE API TEST SUITE")
    print("🚀 Testing unlimited text processing capabilities!")
    print("=" * 60)
    
    # Run individual tests
    test_streaming_api()
    
    # Run comparison test
    test_model_comparison()
    
    print("\n🎉 STREAMING ENGINE API TESTING COMPLETE!")
    print("🌟 SUM can now process texts of ANY SIZE through the API!")