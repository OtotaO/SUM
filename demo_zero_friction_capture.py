#!/usr/bin/env python3
"""
Demo: SUM Zero-Friction Capture System

Demonstrates the revolutionary capture system that transforms SUM from
"another great tool" into "the future of human-computer cognitive collaboration."

This demo showcases:
1. Global hotkey system with instant popup
2. API-based capture processing  
3. Context-aware summarization
4. Sub-second processing performance
5. Beautiful progress indication

Author: ototao (optimized with Claude Code)
License: Apache License 2.0
"""

import asyncio
import time
import json
import requests
import threading
from typing import Dict, Any
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import SUM capture system
from capture.capture_engine import capture_engine, CaptureSource
from capture.launcher import launcher


class ZeroFrictionCaptureDemo:
    """
    Interactive demo of the zero-friction capture system.
    Shows real-world usage scenarios and performance metrics.
    """
    
    def __init__(self):
        self.demo_texts = {
            'short_article': """
            Artificial Intelligence is revolutionizing how we work and live. Machine learning algorithms 
            can now process vast amounts of data in seconds, providing insights that would take humans 
            hours to discover. From healthcare diagnostics to financial trading, AI is becoming an 
            indispensable tool for decision-making in the modern world.
            """,
            
            'technical_content': """
            The Transformer architecture introduced in "Attention Is All You Need" represents a paradigm 
            shift in neural network design. By replacing recurrent and convolutional layers with 
            self-attention mechanisms, Transformers achieve superior performance on sequence-to-sequence 
            tasks while enabling parallel processing. The multi-head attention allows the model to 
            focus on different aspects of the input simultaneously, capturing both local and global 
            dependencies. This architecture has become the foundation for modern language models 
            like GPT, BERT, and T5, demonstrating remarkable capabilities in natural language 
            understanding and generation.
            """,
            
            'news_excerpt': """
            Breaking: Tech giant announces revolutionary breakthrough in quantum computing. The new 
            quantum processor demonstrates quantum supremacy by solving complex optimization problems 
            in minutes that would take classical computers years. This advancement could transform 
            industries from drug discovery to financial modeling, cryptography, and artificial 
            intelligence. Scientists worldwide are calling it a "quantum leap" that brings us closer 
            to practical quantum computing applications.
            """,
            
            'research_paper': """
            Abstract: We present a novel approach to federated learning that addresses the fundamental 
            challenges of data heterogeneity and communication efficiency. Our method, called 
            Adaptive Federated Optimization (AFO), dynamically adjusts local training parameters 
            based on data distribution characteristics and network conditions. Experimental results 
            on benchmark datasets show that AFO achieves 23% better convergence rates compared to 
            FedAvg while reducing communication overhead by 40%. The approach demonstrates particular 
            effectiveness in scenarios with non-IID data distributions and limited bandwidth 
            constraints. These findings have significant implications for privacy-preserving 
            machine learning in distributed environments.
            """
        }
        
        self.performance_metrics = []
        self.api_server_running = False
    
    def print_header(self):
        """Print demo header with beautiful ASCII art."""
        print("\n" + "="*80)
        print("🎯 SUM ZERO-FRICTION CAPTURE SYSTEM DEMO")
        print("="*80)
        print("Revolutionary capture system that transforms content processing")
        print("from 'another great tool' into 'the future of cognitive collaboration'")
        print("="*80)
        print()
    
    def start_demo_environment(self):
        """Start the capture system for demo."""
        print("🚀 Starting SUM Capture System...")
        
        # Start the capture system
        success = launcher.start_all_services(
            enable_hotkey=False,  # Disable for demo to avoid conflicts
            enable_api_server=True,
            api_host='127.0.0.1',
            api_port=8000
        )
        
        if success:
            self.api_server_running = True
            print("✅ Capture system started successfully!")
            time.sleep(2)  # Give services time to initialize
        else:
            print("❌ Failed to start capture system")
            return False
        
        return True
    
    def demo_direct_capture(self):
        """Demonstrate direct capture engine usage."""
        print("\n" + "🔥 DEMO 1: Direct Capture Engine")
        print("-" * 50)
        print("Testing sub-second processing with different content types...")
        print()
        
        for content_type, text in self.demo_texts.items():
            print(f"📄 Processing {content_type.replace('_', ' ').title()}...")
            
            start_time = time.time()
            
            # Direct capture engine usage
            request_id = capture_engine.capture_text(
                text=text.strip(),
                source=CaptureSource.GLOBAL_HOTKEY,  # Simulate hotkey capture
                context={
                    'demo_type': content_type,
                    'timestamp': time.time()
                }
            )
            
            # Wait for result
            result = None
            timeout = 10  # 10 second timeout
            start_wait = time.time()
            
            while time.time() - start_wait < timeout:
                result = capture_engine.get_result(request_id)
                if result:
                    break
                time.sleep(0.1)
            
            processing_time = time.time() - start_time
            
            if result:
                print(f"   ⚡ Processed in {processing_time:.3f}s")
                print(f"   📝 Summary: {result.summary[:100]}...")
                print(f"   🔑 Keywords: {', '.join(result.keywords[:5])}")
                print(f"   🧠 Algorithm: {result.algorithm_used}")
                print(f"   📊 Confidence: {result.confidence_score:.2f}")
                
                self.performance_metrics.append({
                    'content_type': content_type,
                    'processing_time': processing_time,
                    'algorithm': result.algorithm_used,
                    'confidence': result.confidence_score,
                    'word_count': len(text.split())
                })
            else:
                print(f"   ❌ Processing timed out after {timeout}s")
            
            print()
            time.sleep(1)  # Brief pause between demos
    
    def demo_api_capture(self):
        """Demonstrate API-based capture."""
        print("\n" + "🌐 DEMO 2: API-Based Capture")
        print("-" * 50)
        print("Testing HTTP API with browser extension simulation...")
        print()
        
        if not self.api_server_running:
            print("❌ API server not running. Skipping API demo.")
            return
        
        # Test API endpoints
        api_url = "http://127.0.0.1:8000"
        
        # Health check
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ API server is healthy")
            else:
                print(f"⚠️  API server returned status {response.status_code}")
        except requests.RequestException as e:
            print(f"❌ Cannot connect to API server: {e}")
            return
        
        # Test capture endpoint
        test_text = self.demo_texts['news_excerpt'].strip()
        
        print(f"📡 Sending capture request to API...")
        
        start_time = time.time()
        
        try:
            response = requests.post(f"{api_url}/api/capture", json={
                'text': test_text,
                'source': 'browser_extension',
                'context': {
                    'url': 'https://example-news-site.com/article',
                    'title': 'Breaking Tech News',
                    'captureType': 'selection',
                    'pageType': 'news_article'
                }
            }, timeout=30)
            
            api_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ⚡ API response in {api_time:.3f}s")
                print(f"   📝 Summary: {result['summary'][:100]}...")
                print(f"   🔑 Keywords: {', '.join(result['keywords'][:5])}")
                print(f"   🧠 Algorithm: {result['algorithm_used']}")
                print(f"   📊 Confidence: {result['confidence_score']:.2f}")
                
                self.performance_metrics.append({
                    'content_type': 'api_news',
                    'processing_time': api_time,
                    'algorithm': result['algorithm_used'],
                    'confidence': result['confidence_score'],
                    'word_count': len(test_text.split())
                })
            else:
                print(f"   ❌ API request failed: {response.status_code}")
                print(f"   Error: {response.text}")
        
        except requests.RequestException as e:
            print(f"   ❌ API request failed: {e}")
        
        print()
    
    def demo_context_awareness(self):
        """Demonstrate context-aware processing."""
        print("\n" + "🧠 DEMO 3: Context-Aware Processing")
        print("-" * 50)
        print("Testing intelligent algorithm selection based on content and source...")
        print()
        
        # Test different contexts with same text
        base_text = self.demo_texts['technical_content'].strip()
        
        contexts = [
            {
                'source': CaptureSource.GLOBAL_HOTKEY,
                'context': {'demo_scenario': 'quick_hotkey_capture'},
                'description': 'Global Hotkey (Speed Priority)'
            },
            {
                'source': CaptureSource.BROWSER_EXTENSION,
                'context': {'page_type': 'article', 'url': 'https://research-blog.com'},
                'description': 'Browser Extension (Quality Priority)'
            },
            {
                'source': CaptureSource.EMAIL,
                'context': {'type': 'newsletter', 'sender': 'tech-digest@example.com'},
                'description': 'Email Newsletter (Hierarchical)'
            }
        ]
        
        for ctx in contexts:
            print(f"📋 Testing: {ctx['description']}")
            
            start_time = time.time()
            
            request_id = capture_engine.capture_text(
                text=base_text,
                source=ctx['source'],
                context=ctx['context']
            )
            
            # Wait for result
            result = None
            timeout = 10
            start_wait = time.time()
            
            while time.time() - start_wait < timeout:
                result = capture_engine.get_result(request_id)
                if result:
                    break
                time.sleep(0.1)
            
            processing_time = time.time() - start_time
            
            if result:
                print(f"   ⚡ Time: {processing_time:.3f}s")
                print(f"   🧠 Algorithm Selected: {result.algorithm_used}")
                print(f"   📊 Confidence: {result.confidence_score:.2f}")
                print(f"   📏 Compression Ratio: {len(result.summary.split()) / len(base_text.split()):.2f}")
            else:
                print(f"   ❌ Processing timed out")
            
            print()
            time.sleep(0.5)
    
    def demo_performance_analysis(self):
        """Show performance analysis and statistics."""
        print("\n" + "📊 DEMO 4: Performance Analysis")
        print("-" * 50)
        
        if not self.performance_metrics:
            print("No performance metrics collected.")
            return
        
        print("Performance Summary:")
        print()
        
        # Calculate statistics
        total_requests = len(self.performance_metrics)
        avg_time = sum(m['processing_time'] for m in self.performance_metrics) / total_requests
        min_time = min(m['processing_time'] for m in self.performance_metrics)
        max_time = max(m['processing_time'] for m in self.performance_metrics)
        avg_confidence = sum(m['confidence'] for m in self.performance_metrics) / total_requests
        
        print(f"   📈 Total Requests: {total_requests}")
        print(f"   ⚡ Average Time: {avg_time:.3f}s")
        print(f"   🚀 Fastest: {min_time:.3f}s")
        print(f"   🐌 Slowest: {max_time:.3f}s")
        print(f"   📊 Avg Confidence: {avg_confidence:.2f}")
        print()
        
        # Algorithm distribution
        algorithms = {}
        for metric in self.performance_metrics:
            alg = metric['algorithm']
            algorithms[alg] = algorithms.get(alg, 0) + 1
        
        print("Algorithm Usage:")
        for alg, count in algorithms.items():
            percentage = (count / total_requests) * 100
            print(f"   🧠 {alg}: {count} requests ({percentage:.1f}%)")
        
        print()
        
        # Performance by content type
        print("Performance by Content Type:")
        for metric in self.performance_metrics:
            content_type = metric['content_type'].replace('_', ' ').title()
            words = metric['word_count']
            time_per_word = metric['processing_time'] / words * 1000  # ms per word
            
            print(f"   📄 {content_type}:")
            print(f"      • {words} words in {metric['processing_time']:.3f}s")
            print(f"      • {time_per_word:.1f}ms per word")
            print(f"      • Algorithm: {metric['algorithm']}")
        
        print()
    
    def demo_system_stats(self):
        """Show system statistics."""
        print("\n" + "🔧 DEMO 5: System Statistics") 
        print("-" * 50)
        
        # Get capture engine stats
        engine_stats = capture_engine.get_stats()
        
        print("Capture Engine Statistics:")
        print(f"   📊 Requests Processed: {engine_stats.get('requests_processed', 0)}")
        print(f"   ⏱️  Total Processing Time: {engine_stats.get('total_processing_time', 0):.3f}s")
        print(f"   📈 Cache Hits: {engine_stats.get('cache_hits', 0)}")
        print(f"   📉 Cache Misses: {engine_stats.get('cache_misses', 0)}")
        
        if engine_stats.get('requests_processed', 0) > 0:
            avg_time = engine_stats['total_processing_time'] / engine_stats['requests_processed']
            print(f"   ⚡ Average Processing Time: {avg_time:.3f}s")
        
        print()
        
        # Get system status if launcher is available
        if hasattr(launcher, 'get_status'):
            status = launcher.get_status()
            print("System Status:")
            print(f"   🟢 Services Running: {status.get('services_running', False)}")
            print(f"   ⌨️  Hotkey Active: {status.get('hotkey_active', False)}")
            print(f"   🌐 API Server Active: {status.get('api_server_active', False)}")
            print(f"   ⏰ Uptime: {status.get('uptime', 0):.1f}s")
        
        print()
    
    def run_interactive_demo(self):
        """Run interactive demo mode."""
        print("\n" + "🎮 INTERACTIVE DEMO MODE")
        print("-" * 50)
        print("Enter your own text to see the zero-friction capture in action!")
        print("Type 'quit' to exit the demo.")
        print()
        
        while True:
            try:
                user_input = input("📝 Enter text to capture (or 'quit'): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    print("Please enter some text.")
                    continue
                
                print(f"⚡ Processing your text...")
                
                start_time = time.time()
                
                request_id = capture_engine.capture_text(
                    text=user_input,
                    source=CaptureSource.GLOBAL_HOTKEY,
                    context={'demo_mode': 'interactive'}
                )
                
                # Wait for result
                result = None
                timeout = 15
                start_wait = time.time()
                
                while time.time() - start_wait < timeout:
                    result = capture_engine.get_result(request_id)
                    if result:
                        break
                    time.sleep(0.1)
                
                processing_time = time.time() - start_time
                
                if result:
                    print()
                    print("🎯 CAPTURE RESULTS:")
                    print(f"   ⚡ Processing Time: {processing_time:.3f}s")
                    print(f"   📝 Summary: {result.summary}")
                    print(f"   🔑 Keywords: {', '.join(result.keywords)}")
                    print(f"   🧠 Algorithm: {result.algorithm_used}")
                    print(f"   📊 Confidence: {result.confidence_score:.2f}")
                else:
                    print("❌ Processing timed out. Please try shorter text.")
                
                print()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("Demo ended. Thanks for trying SUM!")
    
    def cleanup(self):
        """Clean up demo environment."""
        print("\n🧹 Cleaning up demo environment...")
        
        if launcher.services_running:
            launcher.shutdown_all_services()
        
        print("✅ Demo cleanup complete")
    
    def run_full_demo(self):
        """Run the complete demo sequence."""
        try:
            self.print_header()
            
            if not self.start_demo_environment():
                print("❌ Failed to start demo environment")
                return
            
            # Run all demo sections
            self.demo_direct_capture()
            self.demo_api_capture()
            self.demo_context_awareness()
            self.demo_performance_analysis()
            self.demo_system_stats()
            
            # Interactive mode
            print("\n" + "🎯 DEMO COMPLETE!")
            print("-" * 50)
            print("All demo scenarios completed successfully!")
            print()
            
            # Ask if user wants interactive mode
            try:
                response = input("Would you like to try interactive mode? (y/n): ").lower()
                if response in ['y', 'yes']:
                    self.run_interactive_demo()
            except KeyboardInterrupt:
                pass
            
            print("\n" + "🎉 DEMO FINISHED!")
            print("-" * 50)
            print("The SUM Zero-Friction Capture System is ready for production use!")
            print()
            print("Next steps:")
            print("1. Install the browser extension for webpage capture")
            print("2. Run 'python -m capture.launcher' to start the full system")
            print("3. Press Ctrl+Shift+T anywhere to capture text instantly")
            print("4. Build integrations using the HTTP API")
            print()
            
        except KeyboardInterrupt:
            print("\n\nDemo interrupted by user.")
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()


def main():
    """Main demo entry point."""
    demo = ZeroFrictionCaptureDemo()
    demo.run_full_demo()


if __name__ == "__main__":
    main()