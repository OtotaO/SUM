"""
compression.py - Adaptive Compression API Endpoints

Clean compression API following Carmack's principles:
- Content-aware compression strategies
- Fast adaptive processing
- Clear compression metrics
- Minimal overhead

Author: ototao
License: Apache License 2.0
"""

import logging
from flask import Blueprint, request, jsonify

from web.middleware import rate_limit, validate_json_input
from application.service_registry import registry


logger = logging.getLogger(__name__)
compression_bp = Blueprint('compression', __name__)


def _check_adaptive_availability():
    """Check if adaptive compression is available."""
    engine = registry.get_service('adaptive_engine')
    if not engine:
        return jsonify({
            'error': 'Adaptive compression not available',
            'details': 'Please check dependencies'
        }), 503
    return engine


def _check_life_availability():
    """Check if life compression is available."""
    system = registry.get_service('life_system')
    if not system:
        return jsonify({
            'error': 'Life compression not available',
            'details': 'Please check dependencies'
        }), 503
    return system


@compression_bp.route('/adaptive_compress', methods=['POST'])
@rate_limit(20, 60)
@validate_json_input()
def adaptive_compress():
    """
    Compress text using content-aware compression strategies.
    
    Expected JSON input:
    {
        "text": "Text to compress...",
        "target_ratio": 0.3,
        "content_type": "auto|philosophical|technical|narrative|activity_log",
        "benchmark": false
    }
    """
    engine = _check_adaptive_availability()
    if isinstance(engine, tuple):  # Error response
        return engine
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        target_ratio = float(data.get('target_ratio', 0.2))
        content_type_str = data.get('content_type', 'auto')
        run_benchmark = data.get('benchmark', False)
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Determine content type
        force_type = None
        if content_type_str != 'auto':
            try:
                from adaptive_compression import ContentType
                force_type = ContentType[content_type_str.upper()]
            except KeyError:
                return jsonify({
                    'error': f'Invalid content type: {content_type_str}'
                }), 400
        
        # Compress text
        result = engine.compress(text, target_ratio, force_type)
        
        # Run benchmarks if requested
        benchmark_results = None
        if run_benchmark:
            benchmark_results = {}
            benchmarks = engine.benchmark_compression()
            for content_type, metrics in benchmarks.items():
                benchmark_results[content_type.value] = {
                    'compression_ratio': metrics.compression_ratio,
                    'information_retention': metrics.information_retention,
                    'semantic_coherence': metrics.semantic_coherence,
                    'readability_score': metrics.readability_score,
                    'processing_time': metrics.processing_time
                }
        
        response = {
            'compressed_text': result['compressed'],
            'content_type': result['content_type'],
            'information_density': result['information_density'],
            'compression_metrics': {
                'target_ratio': result['target_ratio'],
                'adjusted_ratio': result['adjusted_ratio'],
                'actual_ratio': result['actual_ratio'],
                'original_words': result['original_length'],
                'compressed_words': result['compressed_length']
            },
            'strategy_used': result['strategy'],
            'processing_time': result['processing_time']
        }
        
        if benchmark_results:
            response['benchmarks'] = benchmark_results
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in adaptive compression: {e}")
        return jsonify({
            'error': 'Compression failed',
            'details': str(e)
        }), 500


@compression_bp.route('/life_compression/start', methods=['POST'])
@rate_limit(5, 300)
def start_life_compression():
    """Start the life compression monitoring system."""
    system = _check_life_availability()
    if isinstance(system, tuple):
        return system
    
    try:
        if not hasattr(system, '_is_running') or not system._is_running:
            system.start()
            system._is_running = True
            return jsonify({
                'status': 'started',
                'message': 'Life compression system is now monitoring'
            })
        else:
            return jsonify({
                'status': 'already_running',
                'message': 'Life compression system is already running'
            })
            
    except Exception as e:
        logger.error(f"Error starting life compression: {e}")
        return jsonify({
            'error': 'Failed to start life compression',
            'details': str(e)
        }), 500


@compression_bp.route('/life_compression/search', methods=['POST'])
@rate_limit(20, 60)
@validate_json_input()
def search_life_history():
    """
    Search through compressed life history.
    
    Expected JSON input:
    {
        "query": "search terms...",
        "limit": 10
    }
    """
    system = _check_life_availability()
    if isinstance(system, tuple):
        return system
    
    try:
        data = request.get_json()
        query = data.get('query', '')
        limit = int(data.get('limit', 10))
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        results = system.search_life_history(query)[:limit]
        
        memories = []
        for memory in results:
            memories.append({
                'time_range': f"{memory.start_time} to {memory.end_time}",
                'time_scale': memory.time_scale,
                'compressed_text': memory.compressed_text,
                'key_concepts': memory.key_concepts,
                'highlights': memory.highlights,
                'compression_ratio': memory.compression_ratio,
                'original_events': memory.original_event_count
            })
        
        return jsonify({
            'query': query,
            'results': memories,
            'count': len(memories)
        })
        
    except Exception as e:
        logger.error(f"Error searching life history: {e}")
        return jsonify({
            'error': 'Search failed',
            'details': str(e)
        }), 500


@compression_bp.route('/progressive_summarization', methods=['GET'])
def progressive_summarization_info():
    """
    Get information about the Progressive Summarization WebSocket API.
    """
    return jsonify({
        'websocket_url': 'ws://localhost:8765',
        'description': 'Revolutionary real-time progressive summarization with live progress updates',
        'features': [
            'Real-time chunk processing visualization',
            'Live concept extraction display', 
            'Progressive summary building',
            'Memory usage and performance monitoring',
            'Interactive parameter adjustment',
            'Beautiful WebSocket interface with animations'
        ],
        'message_types': {
            'start_processing': {
                'description': 'Start progressive text processing',
                'required_fields': ['text'],
                'optional_fields': ['session_id', 'config'],
                'example': {
                    'type': 'start_processing',
                    'text': 'Your text to process...',
                    'session_id': 'unique_session_id',
                    'config': {
                        'chunk_size_words': 1000,
                        'overlap_ratio': 0.15,
                        'max_memory_mb': 512,
                        'max_concurrent_chunks': 4
                    }
                }
            },
            'get_status': {
                'description': 'Get current processing status',
                'example': {'type': 'get_status'}
            },
            'ping': {
                'description': 'Ping the server for connectivity test',
                'example': {'type': 'ping'}
            }
        },
        'usage_instructions': {
            'step_1': 'Start WebSocket server: python progressive_summarization.py',
            'step_2': 'Open progressive_client.html in browser',
            'step_3': 'Connect WebSocket client to ws://localhost:8765',
            'step_4': 'Send start_processing message with your text',
            'step_5': 'Receive real-time progress updates'
        },
        'html_client': 'A beautiful HTML client is auto-generated as progressive_client.html'
    })