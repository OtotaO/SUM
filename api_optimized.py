"""
Optimized SUM API - Carmack-Principles Applied

Key optimizations:
- Single file API (no complex module structure)
- Fast response times with intelligent caching
- Bulletproof error handling
- Memory efficient operations
- Clean, minimal interface surface

Author: ototao (optimized with Claude Code)
License: Apache License 2.0
"""

import time
import logging
import json
from typing import Dict, Any, Optional, List
from functools import wraps
from threading import Lock
import traceback

from flask import Flask, request, jsonify, abort
from werkzeug.exceptions import BadRequest, InternalServerError

# Core engine import
from core import SumEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global engine instance (singleton for performance)
_engine = None
_engine_lock = Lock()

# Rate limiting storage
_rate_limit_storage = {}
_rate_limit_lock = Lock()


def get_engine() -> SumEngine:
    """Get or create the global engine instance (thread-safe singleton)."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = SumEngine()
                logger.info("SumEngine initialized")
    return _engine


def rate_limit(calls_per_minute: int = 60):
    """
    Simple rate limiting decorator.
    
    Args:
        calls_per_minute: Maximum calls allowed per minute
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            client_ip = request.remote_addr
            current_time = time.time()
            
            with _rate_limit_lock:
                if client_ip not in _rate_limit_storage:
                    _rate_limit_storage[client_ip] = []
                
                # Clean old entries (older than 1 minute)
                _rate_limit_storage[client_ip] = [
                    timestamp for timestamp in _rate_limit_storage[client_ip]
                    if current_time - timestamp < 60
                ]
                
                # Check rate limit
                if len(_rate_limit_storage[client_ip]) >= calls_per_minute:
                    abort(429)  # Too Many Requests
                
                # Add current request
                _rate_limit_storage[client_ip].append(current_time)
            
            return f(*args, **kwargs)
        return wrapper
    return decorator


def validate_json(required_fields: List[str] = None):
    """
    Validate JSON input decorator.
    
    Args:
        required_fields: List of required field names
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not request.is_json:
                abort(400, description="Content-Type must be application/json")
            
            try:
                data = request.get_json()
                if data is None:
                    abort(400, description="Invalid JSON")
            except Exception:
                abort(400, description="Invalid JSON format")
            
            # Check required fields
            if required_fields:
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    abort(400, description=f"Missing required fields: {missing_fields}")
            
            return f(*args, **kwargs)
        return wrapper
    return decorator


def create_app() -> Flask:
    """Create optimized Flask application."""
    app = Flask(__name__)
    
    # App configuration
    app.config.update({
        'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB max
        'JSON_SORT_KEYS': False,
        'JSONIFY_PRETTYPRINT_REGULAR': False,  # Faster JSON responses
    })
    
    # Global error handlers
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            'error': 'Bad Request',
            'message': error.description or 'Invalid request'
        }), 400
    
    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        return jsonify({
            'error': 'Rate Limit Exceeded',
            'message': 'Too many requests. Please slow down.'
        }), 429
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred'
        }), 500
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        """Fast health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'version': '2.0.0-optimized',
            'engine': 'SumEngine-Carmack',
            'timestamp': time.time()
        })
    
    # Engine statistics endpoint
    @app.route('/stats', methods=['GET'])
    def get_stats():
        """Get engine performance statistics."""
        try:
            engine = get_engine()
            stats = engine.get_stats()
            return jsonify({
                'engine_stats': stats,
                'uptime': time.time() - app.start_time if hasattr(app, 'start_time') else 0
            })
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return jsonify({'error': 'Failed to get statistics'}), 500
    
    # Main summarization endpoint
    @app.route('/summarize', methods=['POST'])
    @rate_limit(calls_per_minute=30)  # Conservative rate limit
    @validate_json(required_fields=['text'])
    def summarize_text():
        """
        Main text summarization endpoint.
        
        Expected JSON:
        {
            "text": "Text to summarize...",
            "max_length": 100,
            "algorithm": "auto|fast|quality|hierarchical",
            "options": {
                "include_keywords": true,
                "include_concepts": true,
                "include_stats": true
            }
        }
        """
        start_time = time.time()
        
        try:
            data = request.get_json()
            
            # Extract parameters with validation
            text = data.get('text', '').strip()
            if not text:
                return jsonify({'error': 'Empty text provided'}), 400
            
            if len(text) > 100000:  # 100k character limit
                return jsonify({'error': 'Text too long (max 100k characters)'}), 400
            
            max_length = data.get('max_length', 100)
            if not isinstance(max_length, int) or not (5 <= max_length <= 1000):
                return jsonify({'error': 'max_length must be integer between 5 and 1000'}), 400
            
            algorithm = data.get('algorithm', 'auto')
            if algorithm not in ['auto', 'fast', 'quality', 'hierarchical']:
                return jsonify({'error': 'Invalid algorithm. Use: auto, fast, quality, hierarchical'}), 400
            
            options = data.get('options', {})
            
            # Get engine and process
            engine = get_engine()
            result = engine.summarize(
                text=text,
                max_length=max_length,
                algorithm=algorithm,
                **options
            )
            
            # Handle engine errors
            if 'error' in result:
                return jsonify(result), 500
            
            # Add API metadata
            result['api_metadata'] = {
                'request_time': time.time() - start_time,
                'text_length': len(text),
                'algorithm_requested': algorithm,
                'options_used': options
            }
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Summarization error: {e}\n{traceback.format_exc()}")
            return jsonify({
                'error': 'Summarization failed',
                'message': str(e)
            }), 500
    
    # Batch processing endpoint
    @app.route('/summarize/batch', methods=['POST'])
    @rate_limit(calls_per_minute=10)  # Lower limit for batch processing
    @validate_json(required_fields=['texts'])
    def batch_summarize():
        """
        Batch text summarization.
        
        Expected JSON:
        {
            "texts": ["Text 1", "Text 2", ...],
            "max_length": 100,
            "algorithm": "auto"
        }
        """
        try:
            data = request.get_json()
            texts = data.get('texts', [])
            
            if not isinstance(texts, list) or not texts:
                return jsonify({'error': 'texts must be non-empty list'}), 400
            
            if len(texts) > 50:  # Limit batch size
                return jsonify({'error': 'Maximum 50 texts per batch'}), 400
            
            max_length = data.get('max_length', 100)
            algorithm = data.get('algorithm', 'auto')
            
            # Process each text
            engine = get_engine()
            results = []
            
            for i, text in enumerate(texts):
                if not isinstance(text, str) or not text.strip():
                    results.append({'error': f'Invalid text at index {i}'})
                    continue
                
                result = engine.summarize(
                    text=text.strip(),
                    max_length=max_length,
                    algorithm=algorithm
                )
                results.append(result)
            
            return jsonify({
                'results': results,
                'batch_size': len(texts),
                'algorithm_used': algorithm
            })
            
        except Exception as e:
            logger.error(f"Batch summarization error: {e}")
            return jsonify({
                'error': 'Batch processing failed',
                'message': str(e)
            }), 500
    
    # Keywords extraction endpoint
    @app.route('/keywords', methods=['POST'])
    @rate_limit(calls_per_minute=60)
    @validate_json(required_fields=['text'])
    def extract_keywords():
        """
        Extract keywords from text.
        
        Expected JSON:
        {
            "text": "Text to analyze...",
            "count": 10
        }
        """
        try:
            data = request.get_json()
            text = data.get('text', '').strip()
            
            if not text:
                return jsonify({'error': 'Empty text provided'}), 400
            
            count = data.get('count', 10)
            if not isinstance(count, int) or not (1 <= count <= 50):
                return jsonify({'error': 'count must be integer between 1 and 50'}), 400
            
            # Get engine and extract keywords
            engine = get_engine()
            keywords = engine.analyzer.extract_keywords(text, count=count)
            concepts = engine.analyzer.extract_concepts(text, max_concepts=min(count, 10))
            
            return jsonify({
                'keywords': keywords,
                'concepts': concepts,
                'text_length': len(text.split())
            })
            
        except Exception as e:
            logger.error(f"Keyword extraction error: {e}")
            return jsonify({
                'error': 'Keyword extraction failed',
                'message': str(e)
            }), 500
    
    # Text analysis endpoint
    @app.route('/analyze', methods=['POST'])
    @rate_limit(calls_per_minute=30)
    @validate_json(required_fields=['text'])
    def analyze_text():
        """
        Comprehensive text analysis.
        
        Expected JSON:
        {
            "text": "Text to analyze...",
            "include_sentiment": true,
            "include_entities": true,
            "include_topics": true
        }
        """
        try:
            data = request.get_json()
            text = data.get('text', '').strip()
            
            if not text:
                return jsonify({'error': 'Empty text provided'}), 400
            
            engine = get_engine()
            analyzer = engine.analyzer
            processor = engine.processor
            
            # Base analysis
            result = {
                'text_stats': processor.get_text_stats(text),
                'keywords': analyzer.extract_keywords(text, count=10),
                'concepts': analyzer.extract_concepts(text),
                'complexity': analyzer.get_content_complexity(text)
            }
            
            # Optional analyses
            if data.get('include_sentiment', False):
                result['sentiment'] = analyzer.detect_sentiment(text)
            
            if data.get('include_entities', False):
                result['entities'] = analyzer.extract_named_entities(text)
            
            if data.get('include_topics', False):
                result['topics'] = analyzer.analyze_topic_distribution(text)
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Text analysis error: {e}")
            return jsonify({
                'error': 'Text analysis failed',
                'message': str(e)
            }), 500
    
    # Cache management endpoint
    @app.route('/cache/clear', methods=['POST'])
    def clear_cache():
        """Clear engine caches."""
        try:
            engine = get_engine()
            engine.clear_cache()
            return jsonify({'message': 'Caches cleared successfully'})
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return jsonify({'error': 'Failed to clear caches'}), 500
    
    # Configuration endpoint
    @app.route('/config', methods=['GET'])
    def get_config():
        """Get API configuration information."""
        return jsonify({
            'api_version': '2.0.0-optimized',
            'engine': 'SumEngine-Carmack',
            'algorithms': ['auto', 'fast', 'quality', 'hierarchical'],
            'limits': {
                'max_text_length': 100000,
                'max_summary_length': 1000,
                'max_batch_size': 50,
                'rate_limit_default': 30
            },
            'features': [
                'text_summarization',
                'keyword_extraction',
                'concept_analysis',
                'batch_processing',
                'sentiment_analysis',
                'entity_recognition',
                'topic_analysis'
            ]
        })
    
    return app


def main():
    """Main entry point for optimized SUM API."""
    app = create_app()
    app.start_time = time.time()
    
    # Development server
    logger.info("Starting optimized SUM API server...")
    logger.info("Carmack principles applied: Fast, Simple, Clear, Bulletproof")
    
    app.run(
        host='0.0.0.0',
        port=3000,
        debug=False,  # Production-ready
        threaded=True  # Enable threading for better performance
    )


if __name__ == '__main__':
    main()