"""
Capture API Server - HTTP API for Zero-Friction Capture System

Provides HTTP endpoints for browser extensions, mobile apps, and other
clients to submit capture requests to the SUM engine.

Author: ototao (optimized with Claude Code)
License: Apache License 2.0
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import time
import json
import uuid
from typing import Dict, Any, Optional
import threading
from datetime import datetime, timedelta

# Import capture engine
from .capture_engine import capture_engine, CaptureSource, CaptureResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for browser extension

# Request tracking
active_requests: Dict[str, Dict[str, Any]] = {}
request_lock = threading.Lock()

# Statistics
server_stats = {
    'requests_received': 0,
    'requests_completed': 0,
    'requests_failed': 0,
    'total_processing_time': 0.0,
    'server_start_time': time.time(),
    'requests_by_source': {}
}


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'uptime': time.time() - server_stats['server_start_time'],
        'version': '1.0.0'
    })


@app.route('/api/capture', methods=['POST'])
def capture_text():
    """
    Main capture endpoint for text processing.
    
    Request body:
    {
        "text": "Text to capture and summarize",
        "source": "browser_extension",
        "context": {
            "url": "https://example.com",
            "title": "Page Title",
            "capture_type": "selection"
        }
    }
    """
    try:
        # Parse request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required fields
        if 'text' not in data or not data['text'].strip():
            return jsonify({'error': 'Text field is required and cannot be empty'}), 400
        
        text = data['text'].strip()
        source_str = data.get('source', 'api')
        context = data.get('context', {})
        
        # Convert source string to enum
        try:
            source = CaptureSource(source_str)
        except ValueError:
            source = CaptureSource.API_WEBHOOK
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Track request
        with request_lock:
            server_stats['requests_received'] += 1
            server_stats['requests_by_source'][source_str] = (
                server_stats['requests_by_source'].get(source_str, 0) + 1
            )
            
            active_requests[request_id] = {
                'timestamp': time.time(),
                'source': source_str,
                'text_length': len(text),
                'context': context
            }
        
        logger.info(f"Received capture request {request_id} from {source_str}")
        
        # Submit to capture engine
        capture_request_id = capture_engine.capture_text(
            text=text,
            source=source,
            context=context,
            callback=lambda result: _handle_capture_complete(request_id, result)
        )
        
        # For synchronous response, wait for completion (with timeout)
        result = _wait_for_result(capture_request_id, timeout=30.0)
        
        if result:
            # Clean up tracking
            with request_lock:
                active_requests.pop(request_id, None)
                server_stats['requests_completed'] += 1
                server_stats['total_processing_time'] += result.processing_time
            
            # Return result
            response_data = {
                'request_id': request_id,
                'capture_id': result.request_id,
                'summary': result.summary,
                'keywords': result.keywords,
                'concepts': result.concepts,
                'processing_time': result.processing_time,
                'algorithm_used': result.algorithm_used,
                'confidence_score': result.confidence_score,
                'timestamp': time.time()
            }
            
            logger.info(f"Capture request {request_id} completed in {result.processing_time:.3f}s")
            return jsonify(response_data)
        
        else:
            # Timeout or failure
            with request_lock:
                active_requests.pop(request_id, None)
                server_stats['requests_failed'] += 1
            
            logger.error(f"Capture request {request_id} timed out")
            return jsonify({'error': 'Request timed out'}), 408
    
    except Exception as e:
        logger.error(f"Capture request failed: {e}", exc_info=True)
        
        with request_lock:
            server_stats['requests_failed'] += 1
        
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@app.route('/api/capture/async', methods=['POST'])
def capture_text_async():
    """
    Asynchronous capture endpoint - returns immediately with request ID.
    Use /api/capture/status/<request_id> to check status.
    """
    try:
        # Parse request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required fields
        if 'text' not in data or not data['text'].strip():
            return jsonify({'error': 'Text field is required and cannot be empty'}), 400
        
        text = data['text'].strip()
        source_str = data.get('source', 'api')
        context = data.get('context', {})
        
        # Convert source string to enum
        try:
            source = CaptureSource(source_str)
        except ValueError:
            source = CaptureSource.API_WEBHOOK
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Track request
        with request_lock:
            server_stats['requests_received'] += 1
            server_stats['requests_by_source'][source_str] = (
                server_stats['requests_by_source'].get(source_str, 0) + 1
            )
            
            active_requests[request_id] = {
                'timestamp': time.time(),
                'source': source_str,
                'text_length': len(text),
                'context': context,
                'status': 'pending',
                'result': None
            }
        
        logger.info(f"Received async capture request {request_id} from {source_str}")
        
        # Submit to capture engine
        capture_request_id = capture_engine.capture_text(
            text=text,
            source=source,
            context=context,
            callback=lambda result: _handle_capture_complete(request_id, result)
        )
        
        # Store capture engine request ID for tracking
        with request_lock:
            active_requests[request_id]['capture_request_id'] = capture_request_id
        
        return jsonify({
            'request_id': request_id,
            'status': 'processing',
            'message': 'Request submitted for processing'
        })
    
    except Exception as e:
        logger.error(f"Async capture request failed: {e}", exc_info=True)
        
        with request_lock:
            server_stats['requests_failed'] += 1
        
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@app.route('/api/capture/status/<request_id>', methods=['GET'])
def get_capture_status(request_id: str):
    """Get the status of an async capture request."""
    with request_lock:
        request_data = active_requests.get(request_id)
    
    if not request_data:
        return jsonify({'error': 'Request not found'}), 404
    
    response = {
        'request_id': request_id,
        'status': request_data.get('status', 'pending'),
        'submitted_at': request_data['timestamp']
    }
    
    # Add result if completed
    if request_data.get('result'):
        result = request_data['result']
        response.update({
            'summary': result.summary,
            'keywords': result.keywords,
            'concepts': result.concepts,
            'processing_time': result.processing_time,
            'algorithm_used': result.algorithm_used,
            'confidence_score': result.confidence_score
        })
    
    return jsonify(response)


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get server and capture engine statistics."""
    # Get capture engine stats
    engine_stats = capture_engine.get_stats()
    
    # Combine with server stats
    combined_stats = {
        'server': {
            **server_stats,
            'active_requests': len(active_requests),
            'uptime': time.time() - server_stats['server_start_time']
        },
        'capture_engine': engine_stats,
        'timestamp': time.time()
    }
    
    return jsonify(combined_stats)


@app.route('/api/captures/recent', methods=['GET'])
def get_recent_captures():
    """Get recent capture results (last 50)."""
    # This would typically come from a database
    # For now, return empty list as this is a demo
    return jsonify({
        'captures': [],
        'message': 'Recent captures feature coming soon'
    })


def _wait_for_result(capture_request_id: str, timeout: float = 30.0) -> Optional[CaptureResult]:
    """Wait for capture result with timeout."""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        result = capture_engine.get_result(capture_request_id)
        if result:
            return result
        
        time.sleep(0.1)  # Check every 100ms
    
    return None


def _handle_capture_complete(request_id: str, result: CaptureResult):
    """Handle completion of async capture request."""
    with request_lock:
        if request_id in active_requests:
            active_requests[request_id]['status'] = 'completed'
            active_requests[request_id]['result'] = result
            active_requests[request_id]['completed_at'] = time.time()


def _cleanup_old_requests():
    """Clean up old requests (run periodically)."""
    cutoff_time = time.time() - 3600  # 1 hour ago
    
    with request_lock:
        expired_requests = [
            req_id for req_id, req_data in active_requests.items()
            if req_data['timestamp'] < cutoff_time
        ]
        
        for req_id in expired_requests:
            active_requests.pop(req_id, None)
    
    if expired_requests:
        logger.info(f"Cleaned up {len(expired_requests)} expired requests")


def start_cleanup_thread():
    """Start background thread for request cleanup."""
    def cleanup_loop():
        while True:
            time.sleep(300)  # Clean up every 5 minutes
            _cleanup_old_requests()
    
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()


if __name__ == '__main__':
    logger.info("Starting SUM Capture API Server...")
    
    # Start cleanup thread
    start_cleanup_thread()
    
    # Start Flask app
    app.run(
        host='127.0.0.1',
        port=8000,
        debug=False,
        threaded=True
    )