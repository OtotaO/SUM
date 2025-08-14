#!/usr/bin/env python3
"""
Production readiness test - simulates real-world usage
"""
import os
import sys
import time
import requests
import threading
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
BASE_URL = "http://localhost:5001"
NUM_CONCURRENT_USERS = 5
REQUESTS_PER_USER = 10

print("🚀 SUM Platform Production Readiness Test")
print("=" * 50)

def check_server():
    """Check if server is running"""
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        return response.status_code == 200
    except:
        return False

def test_file_upload(user_id, request_id):
    """Test file upload with validation"""
    try:
        # Create test file
        test_content = f"User {user_id} - Request {request_id}\n" * 100
        test_content += "This is a test document for the SUM platform. " * 50
        
        files = {
            'file': (f'test_{user_id}_{request_id}.txt', test_content, 'text/plain')
        }
        data = {
            'model': 'simple',
            'maxTokens': '100'
        }
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/api/analyze_file",
            files=files,
            data=data,
            timeout=30
        )
        duration = time.time() - start_time
        
        return {
            'user_id': user_id,
            'request_id': request_id,
            'status_code': response.status_code,
            'duration': duration,
            'response': response.json() if response.status_code == 200 else response.text
        }
    except Exception as e:
        return {
            'user_id': user_id,
            'request_id': request_id,
            'error': str(e)
        }

def test_text_summarization(user_id, request_id):
    """Test text summarization endpoint"""
    try:
        test_text = f"User {user_id} Request {request_id}. " + """
        The SUM platform is a comprehensive knowledge distillation system that uses advanced AI techniques
        to extract meaningful insights from large volumes of text. It employs multiple summarization models,
        topic analysis, and semantic understanding to provide users with concise, accurate summaries.
        """ * 10
        
        data = {
            'text': test_text,
            'model': 'simple',
            'maxLength': 100
        }
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/api/summarize",
            json=data,
            timeout=30
        )
        duration = time.time() - start_time
        
        return {
            'user_id': user_id,
            'request_id': request_id,
            'status_code': response.status_code,
            'duration': duration,
            'response': response.json() if response.status_code == 200 else response.text
        }
    except Exception as e:
        return {
            'user_id': user_id,
            'request_id': request_id,
            'error': str(e)
        }

def simulate_user_load():
    """Simulate multiple concurrent users"""
    print(f"\n📊 Simulating {NUM_CONCURRENT_USERS} concurrent users...")
    print(f"Each user will make {REQUESTS_PER_USER} requests")
    
    results = []
    with ThreadPoolExecutor(max_workers=NUM_CONCURRENT_USERS) as executor:
        futures = []
        
        # Submit all tasks
        for user_id in range(NUM_CONCURRENT_USERS):
            for request_id in range(REQUESTS_PER_USER):
                # Alternate between file upload and text summarization
                if request_id % 2 == 0:
                    future = executor.submit(test_file_upload, user_id, request_id)
                else:
                    future = executor.submit(test_text_summarization, user_id, request_id)
                futures.append(future)
        
        # Collect results
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            
            # Print progress
            if len(results) % 10 == 0:
                print(f"Progress: {len(results)}/{len(futures)} requests completed")
    
    return results

def analyze_results(results):
    """Analyze test results"""
    print("\n📈 Test Results Analysis")
    print("=" * 50)
    
    # Overall statistics
    total_requests = len(results)
    successful_requests = sum(1 for r in results if r.get('status_code') == 200)
    failed_requests = sum(1 for r in results if 'error' in r or r.get('status_code', 0) != 200)
    
    print(f"Total Requests: {total_requests}")
    print(f"Successful: {successful_requests} ({successful_requests/total_requests*100:.1f}%)")
    print(f"Failed: {failed_requests} ({failed_requests/total_requests*100:.1f}%)")
    
    # Response time analysis
    response_times = [r['duration'] for r in results if 'duration' in r]
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        print(f"\nResponse Times:")
        print(f"  Average: {avg_time:.2f}s")
        print(f"  Min: {min_time:.2f}s")
        print(f"  Max: {max_time:.2f}s")
    
    # Error analysis
    errors = [r for r in results if 'error' in r]
    if errors:
        print(f"\nErrors encountered:")
        error_types = {}
        for error in errors:
            error_type = error.get('error', 'Unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        for error_type, count in error_types.items():
            print(f"  {error_type}: {count}")
    
    # Status code analysis
    status_codes = {}
    for r in results:
        code = r.get('status_code', 'error')
        status_codes[code] = status_codes.get(code, 0) + 1
    
    print(f"\nStatus Code Distribution:")
    for code, count in sorted(status_codes.items()):
        print(f"  {code}: {count}")
    
    return {
        'success_rate': successful_requests / total_requests * 100,
        'avg_response_time': avg_time if response_times else 0
    }

def test_robustness_features():
    """Test specific robustness features"""
    print("\n🛡️ Testing Robustness Features")
    print("=" * 50)
    
    # Test 1: Large file handling
    print("\n1. Testing large file handling...")
    try:
        large_content = "x" * (5 * 1024 * 1024)  # 5MB file
        files = {'file': ('large.txt', large_content, 'text/plain')}
        response = requests.post(f"{BASE_URL}/api/analyze_file", files=files, timeout=60)
        
        if response.status_code == 202:
            print("✅ Large file queued for async processing")
        elif response.status_code == 200:
            print("✅ Large file processed successfully")
        else:
            print(f"❌ Large file handling failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Large file test error: {e}")
    
    # Test 2: Rate limiting
    print("\n2. Testing rate limiting...")
    rapid_requests = []
    for i in range(15):  # Try to exceed rate limit
        try:
            response = requests.get(f"{BASE_URL}/api/health", timeout=1)
            rapid_requests.append(response.status_code)
        except:
            rapid_requests.append('timeout')
    
    rate_limited = any(code == 429 for code in rapid_requests)
    if rate_limited:
        print("✅ Rate limiting is working")
    else:
        print("⚠️ Rate limiting may not be configured")
    
    # Test 3: Error recovery
    print("\n3. Testing error recovery...")
    try:
        # Send malformed request
        response = requests.post(
            f"{BASE_URL}/api/summarize",
            json={'invalid': 'data'},
            timeout=10
        )
        if response.status_code == 400:
            print("✅ Invalid request handled gracefully")
        else:
            print(f"⚠️ Unexpected response: {response.status_code}")
    except Exception as e:
        print(f"❌ Error recovery test failed: {e}")
    
    # Test 4: Health checks
    print("\n4. Testing health check endpoints...")
    try:
        health_response = requests.get(f"{BASE_URL}/health/live", timeout=5)
        ready_response = requests.get(f"{BASE_URL}/health/ready", timeout=5)
        
        if health_response.status_code == 200:
            print("✅ Liveness check passed")
        if ready_response.status_code == 200:
            print("✅ Readiness check passed")
            print(f"   Ready status: {ready_response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")

def main():
    """Main test execution"""
    # Check if server is running
    if not check_server():
        print("❌ Server is not running at", BASE_URL)
        print("Please start the server with: python main.py")
        return
    
    print("✅ Server is running")
    
    # Run load test
    print("\n🏃 Starting load test...")
    start_time = time.time()
    results = simulate_user_load()
    total_duration = time.time() - start_time
    
    # Analyze results
    metrics = analyze_results(results)
    
    print(f"\n⏱️ Total test duration: {total_duration:.2f}s")
    print(f"📊 Requests per second: {len(results)/total_duration:.2f}")
    
    # Test robustness features
    test_robustness_features()
    
    # Final verdict
    print("\n🏁 Production Readiness Assessment")
    print("=" * 50)
    
    if metrics['success_rate'] >= 95:
        print("✅ SUCCESS RATE: Excellent")
    elif metrics['success_rate'] >= 90:
        print("⚠️ SUCCESS RATE: Good, but needs improvement")
    else:
        print("❌ SUCCESS RATE: Poor, not production ready")
    
    if metrics['avg_response_time'] <= 2:
        print("✅ PERFORMANCE: Excellent")
    elif metrics['avg_response_time'] <= 5:
        print("⚠️ PERFORMANCE: Acceptable")
    else:
        print("❌ PERFORMANCE: Too slow")
    
    print(f"\nOverall: The system {'is' if metrics['success_rate'] >= 90 else 'is NOT'} production ready")

if __name__ == "__main__":
    main()