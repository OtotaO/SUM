#!/usr/bin/env python3
"""
quickstart_local.py - Test SUM without any external dependencies

This version works WITHOUT Redis or PostgreSQL.
Perfect for testing the concept immediately.
"""

import os
import time
import hashlib
from flask import Flask, request, jsonify
from transformers import pipeline

print("🚀 SUM v2 - Local Quick Start (No Dependencies)")
print("=" * 50)

# Simple in-memory cache (replaces Redis)
cache = {}

# Configuration
MAX_TEXT_LENGTH = 100000
RATE_LIMIT = 60

# Initialize Flask
app = Flask(__name__)

# Load model
print("Loading AI model (this takes ~30 seconds)...")
start_time = time.time()
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print(f"✓ Model loaded in {time.time() - start_time:.1f}s")

# Simple rate limiting (in-memory)
rate_limit_store = {}

def check_rate_limit_local(client_ip: str) -> bool:
    """Simple in-memory rate limiting"""
    current_time = time.time()
    
    if client_ip not in rate_limit_store:
        rate_limit_store[client_ip] = []
    
    # Clean old entries
    rate_limit_store[client_ip] = [
        t for t in rate_limit_store[client_ip] 
        if current_time - t < 60
    ]
    
    # Check limit
    if len(rate_limit_store[client_ip]) >= RATE_LIMIT:
        return False
    
    rate_limit_store[client_ip].append(current_time)
    return True

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'cache_size': len(cache),
        'version': 'v2-local'
    })

@app.route('/summarize', methods=['POST'])
def summarize():
    """Main summarization endpoint"""
    # Get client IP
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    
    # Rate limit check
    if not check_rate_limit_local(client_ip):
        return jsonify({'error': 'Rate limit exceeded'}), 429
    
    # Get JSON data
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text field'}), 400
    
    text = data['text']
    if not text or len(text) > MAX_TEXT_LENGTH:
        return jsonify({'error': f'Text must be 1-{MAX_TEXT_LENGTH} characters'}), 400
    
    # Generate cache key
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Check cache
    if text_hash in cache:
        return jsonify({
            'summary': cache[text_hash],
            'cached': True,
            'source': 'memory'
        })
    
    # Generate summary
    try:
        start_time = time.time()
        result = summarizer(text, max_length=130, min_length=30, do_sample=False)
        summary = result[0]['summary_text']
        
        # Cache it
        cache[text_hash] = summary
        
        return jsonify({
            'summary': summary,
            'cached': False,
            'processing_time': time.time() - start_time
        })
    
    except Exception as e:
        return jsonify({'error': 'Summarization failed', 'details': str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    """Simple statistics"""
    return jsonify({
        'cache_size': len(cache),
        'cache_memory_kb': sum(len(k) + len(v) for k, v in cache.items()) / 1024,
        'rate_limit_tracked_ips': len(rate_limit_store),
        'status': 'running_locally'
    })

if __name__ == '__main__':
    print("\n✅ SUM v2 is ready!")
    print("=" * 50)
    print("\n📝 Test it with:")
    print("curl -X POST localhost:3000/summarize \\")
    print("  -H \"Content-Type: application/json\" \\")
    print("  -d '{\"text\": \"Your long text here...\"}'")
    print("\n🔍 Check health:")
    print("curl localhost:3000/health")
    print("\n📊 View stats:")
    print("curl localhost:3000/stats")
    print("\n" + "=" * 50)
    print("Starting server on http://localhost:3000")
    print("Press Ctrl+C to stop")
    print("=" * 50 + "\n")
    
    app.run(host='0.0.0.0', port=3000, debug=False)