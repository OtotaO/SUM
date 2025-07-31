#!/usr/bin/env python3
"""
test_knowledge_os_integration.py - Test Knowledge OS Integration

Quick integration test to ensure the Knowledge Operating System
works properly within the SUM platform ecosystem.

Author: ototao
License: Apache License 2.0
"""

import sys
import time
import json
from datetime import datetime

def test_knowledge_os_core():
    """Test the core Knowledge OS functionality."""
    print("🧠 Testing Knowledge OS Core Functionality")
    print("=" * 50)
    
    try:
        from knowledge_os import KnowledgeOperatingSystem
        
        # Initialize Knowledge OS
        print("1. Initializing Knowledge OS...")
        knowledge_os = KnowledgeOperatingSystem("test_knowledge_data")
        print("   ✅ Knowledge OS initialized successfully")
        
        # Test thought capture
        print("\n2. Testing thought capture...")
        test_thoughts = [
            "I'm thinking about how artificial intelligence could revolutionize personal knowledge management",
            "The integration between different AI systems seems crucial for creating seamless user experiences",
            "Privacy-first approaches to AI are becoming increasingly important in our digital age",
            "The concept of a Knowledge Operating System feels like the natural evolution of personal computing"
        ]
        
        captured_thoughts = []
        for i, thought_content in enumerate(test_thoughts, 1):
            thought = knowledge_os.capture_thought(thought_content)
            if thought:
                captured_thoughts.append(thought)
                print(f"   ✅ Captured thought {i}: {thought.id}")
            else:
                print(f"   ❌ Failed to capture thought {i}")
        
        # Wait for background processing
        print("\n3. Waiting for background intelligence processing...")
        time.sleep(3)  # Give background processing time to work
        
        # Test insights
        print("\n4. Testing system insights...")
        insights = knowledge_os.get_system_insights()
        if insights:
            print(f"   ✅ Generated insights:")
            print(f"      - Total thoughts: {insights.get('thinking_patterns', {}).get('total_thoughts', 0)}")
            print(f"      - Concepts tracked: {insights.get('intelligence_summary', {}).get('concepts_tracked', 0)}")
            print(f"      - Beautiful summary: {insights.get('beautiful_summary', 'N/A')[:100]}...")
        
        # Test densification opportunities
        print("\n5. Testing densification opportunities...")
        opportunities = knowledge_os.check_densification_opportunities()
        print(f"   ✅ Found {len(opportunities)} densification opportunities")
        
        # Test search
        print("\n6. Testing thought search...")
        search_results = knowledge_os.search_thoughts("artificial intelligence")
        print(f"   ✅ Found {len(search_results)} search results for 'artificial intelligence'")
        
        print("\n🎉 Knowledge OS core functionality test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Knowledge OS core test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_knowledge_os_interface():
    """Test the Knowledge OS web interface."""
    print("\n🌐 Testing Knowledge OS Web Interface")
    print("=" * 50)
    
    try:
        from knowledge_os_interface import app, init_knowledge_os
        
        # Test Flask app initialization
        print("1. Testing Flask app initialization...")
        with app.test_client() as client:
            # Initialize Knowledge OS
            init_knowledge_os()
            print("   ✅ Flask app and Knowledge OS initialized")
            
            # Test prompt endpoint
            print("\n2. Testing /api/prompt endpoint...")
            response = client.get('/api/prompt')
            if response.status_code == 200:
                data = json.loads(response.data)
                if data.get('success'):
                    print(f"   ✅ Prompt: {data.get('prompt', '')[:50]}...")
                else:
                    print(f"   ❌ Prompt request failed: {data.get('error')}")
            else:
                print(f"   ❌ Prompt request failed with status {response.status_code}")
            
            # Test thought capture endpoint
            print("\n3. Testing /api/capture endpoint...")
            test_thought = {
                'content': 'This is a test thought for the web interface integration'
            }
            response = client.post('/api/capture', 
                                 data=json.dumps(test_thought),
                                 content_type='application/json')
            
            if response.status_code == 200:
                data = json.loads(response.data)
                if data.get('success'):
                    print(f"   ✅ Captured thought: {data.get('thought_id')}")
                else:
                    print(f"   ❌ Capture failed: {data.get('error')}")
            else:
                print(f"   ❌ Capture request failed with status {response.status_code}")
            
            # Test recent thoughts endpoint
            print("\n4. Testing /api/recent-thoughts endpoint...")
            response = client.get('/api/recent-thoughts')
            if response.status_code == 200:
                data = json.loads(response.data)
                if data.get('success'):
                    thoughts = data.get('thoughts', [])
                    print(f"   ✅ Retrieved {len(thoughts)} recent thoughts")
                else:
                    print(f"   ❌ Recent thoughts failed: {data.get('error')}")
            else:
                print(f"   ❌ Recent thoughts request failed with status {response.status_code}")
            
            # Test insights endpoint
            print("\n5. Testing /api/insights endpoint...")
            response = client.get('/api/insights')
            if response.status_code == 200:
                data = json.loads(response.data)
                if data.get('success'):
                    print("   ✅ Retrieved system insights")
                else:
                    print(f"   ❌ Insights failed: {data.get('error')}")
            else:
                print(f"   ❌ Insights request failed with status {response.status_code}")
        
        print("\n🎉 Knowledge OS web interface test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Knowledge OS web interface test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sum_platform_integration():
    """Test Knowledge OS integration with main SUM platform."""
    print("\n🔗 Testing SUM Platform Integration")
    print("=" * 50)
    
    try:
        from main_with_summail import app
        
        print("1. Testing main SUM app with Knowledge OS...")
        with app.test_client() as client:
            # Test system status includes Knowledge OS
            print("   Testing /api/system/status...")
            response = client.get('/api/system/status')
            if response.status_code == 200:
                data = json.loads(response.data)
                system_info = data.get('system', {})
                if system_info.get('knowledge_os_available'):
                    print("   ✅ Knowledge OS reported as available in system status")
                    
                    # Check active modes
                    active_modes = data.get('active_modes', [])
                    if 'knowledge' in active_modes:
                        print("   ✅ Knowledge mode listed in active modes")
                    else:
                        print("   ❌ Knowledge mode not in active modes")
                    
                    # Check capabilities
                    capabilities = data.get('capabilities', {})
                    if 'knowledge_os' in capabilities:
                        print("   ✅ Knowledge OS capabilities included")
                    else:
                        print("   ❌ Knowledge OS capabilities missing")
                else:
                    print("   ❌ Knowledge OS not available in system status")
            else:
                print(f"   ❌ System status request failed with status {response.status_code}")
            
            # Test Knowledge OS API endpoints through main app
            print("\n2. Testing Knowledge OS API through main app...")
            
            # Test capture
            test_thought = {'content': 'Integration test thought from main SUM app'}
            response = client.post('/api/knowledge/capture',
                                 data=json.dumps(test_thought),
                                 content_type='application/json')
            
            if response.status_code == 200:
                data = json.loads(response.data)
                if data.get('success'):
                    print("   ✅ Thought capture through main app successful")
                else:
                    print(f"   ❌ Thought capture failed: {data.get('error')}")
            else:
                print(f"   ❌ Thought capture request failed with status {response.status_code}")
            
            # Test prompt
            response = client.get('/api/knowledge/prompt')
            if response.status_code == 200:
                data = json.loads(response.data)
                if data.get('success'):
                    print("   ✅ Prompt generation through main app successful")
                else:
                    print(f"   ❌ Prompt generation failed: {data.get('error')}")
            else:
                print(f"   ❌ Prompt request failed with status {response.status_code}")
        
        print("\n🎉 SUM Platform integration test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ SUM Platform integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """Run all Knowledge OS integration tests."""
    print("🚀 Knowledge OS Integration Test Suite")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = []
    
    # Run tests
    results.append(("Core Functionality", test_knowledge_os_core()))
    results.append(("Web Interface", test_knowledge_os_interface()))
    results.append(("SUM Platform Integration", test_sum_platform_integration()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Knowledge OS integration is working perfectly.")
        print("\nNext steps:")
        print("• Run the Knowledge OS interface: python knowledge_os_interface.py")
        print("• Run the main SUM platform: python main_with_summail.py")
        print("• Visit http://localhost:5000/knowledge for the Knowledge OS interface")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    run_comprehensive_test()