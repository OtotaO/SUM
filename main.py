"""
main.py - SUM Knowledge Distillation Platform Web Service

This module provides a Flask-based web service for the SUM knowledge
distillation platform, offering both API endpoints and a web interface.

Design principles:
- Clean separation of concerns (Fowler architecture)
- Defensive programming (Schneier security)
- Comprehensive error handling (Stroustrup robustness)
- Efficient resource management (Knuth optimization)
- Clear code structure (Torvalds/van Rossum style)

Author: ototao
License: Apache License 2.0
"""

import os
import logging
import json
import time
from functools import wraps
from typing import Dict, List, Any, Callable, Optional, Tuple, Union
import traceback
import tempfile
from threading import Lock

from flask import Flask, request, jsonify, render_template, send_from_directory, abort
from werkzeug.utils import secure_filename

# Import SUM components
from SUM import SimpleSUM, MagnumOpusSUM, HierarchicalDensificationEngine
from StreamingEngine import StreamingHierarchicalEngine, StreamingConfig
from Utils.data_loader import DataLoader
from Models.topic_modeling import TopicModeler
from Models.summarizer import Summarizer
from config import active_config

# Import Adaptive Compression System
try:
    from adaptive_compression import AdaptiveCompressionEngine, ContentType
    from life_compression_system import LifeCompressionSystem
    ADAPTIVE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Adaptive compression not available: {e}")
    ADAPTIVE_AVAILABLE = False

# Import AI models
try:
    from ai_models import HybridAIEngine, SecureKeyManager, AVAILABLE_MODELS
    AI_AVAILABLE = True
except ImportError as e:
    logger.warning(f"AI models not available: {e}")
    AI_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=getattr(logging, active_config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(active_config.LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

# Configure application from active_config
app.config['SECRET_KEY'] = active_config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = active_config.MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = active_config.UPLOADS_DIR
app.config['ALLOWED_EXTENSIONS'] = active_config.ALLOWED_EXTENSIONS
app.config['DEBUG'] = active_config.DEBUG
app.config['TESTING'] = getattr(active_config, 'TESTING', False)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize SUM components with configuration
simple_summarizer = SimpleSUM()
advanced_summarizer = None  # Lazy-loaded
hierarchical_engine = None  # Lazy-loaded - Hierarchical Densification Engine
streaming_engine = None  # Lazy-loaded - Streaming Engine for unlimited text
topic_modeler = None  # Lazy-loaded
adaptive_engine = None  # Lazy-loaded - Adaptive Compression Engine
life_system = None  # Lazy-loaded - Life Compression System

# Concurrency control
summarizer_lock = Lock()
cache = {}
cache_lock = Lock()

# Add configuration endpoint
@app.route('/api/config', methods=['GET'])
def get_config():
    """Return safe configuration settings."""
    return jsonify(active_config.get_config_dict())


# Utility functions
def allowed_file(filename: str) -> bool:
    """Check if a filename has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def rate_limit(max_calls: int = 10, time_frame: int = 60) -> Callable:
    """
    Rate limiting decorator to prevent abuse.
    
    Args:
        max_calls: Maximum number of calls allowed in the time frame
        time_frame: Time frame in seconds
        
    Returns:
        Decorated function with rate limiting
    """
    calls = {}
    lock = Lock()
    
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapped(*args, **kwargs):
            # Get client identifier (IP)
            client_id = request.remote_addr
            
            current_time = time.time()
            with lock:
                # Clean old entries
                calls_to_remove = []
                for cid, call_history in list(calls.items()):
                    updated_history = [timestamp for timestamp in call_history 
                                      if current_time - timestamp < time_frame]
                    if updated_history:
                        calls[cid] = updated_history
                    else:
                        calls_to_remove.append(cid)
                
                for cid in calls_to_remove:
                    del calls[cid]
                
                # Check rate limit
                if client_id in calls and len(calls[client_id]) >= max_calls:
                    return jsonify({
                        'error': 'Rate limit exceeded',
                        'retry_after': time_frame
                    }), 429
                
                # Record this call
                if client_id not in calls:
                    calls[client_id] = []
                calls[client_id].append(current_time)
            
            # Call the original function
            return f(*args, **kwargs)
        return wrapped
    return decorator


def validate_json_input() -> Callable:
    """
    Decorator to validate JSON input for API endpoints.
    
    Returns:
        Decorated function with JSON validation
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapped(*args, **kwargs):
            if not request.is_json:
                return jsonify({'error': 'Request must be JSON'}), 400
            
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'Empty JSON provided'}), 400
            except Exception as e:
                return jsonify({'error': f'Invalid JSON: {str(e)}'}), 400
            
            return f(*args, **kwargs)
        return wrapped
    return decorator


def timed_lru_cache(max_size: int = 128, expiration: int = 3600) -> Callable:
    """
    Time-based LRU cache decorator with expiration.
    
    Args:
        max_size: Maximum number of items in cache
        expiration: Cache entry lifetime in seconds
        
    Returns:
        Decorated function with caching
    """
    cache_dict = {}
    insertion_times = {}
    access_times = {}
    lock = Lock()
    
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapped(*args, **kwargs):
            # Convert args/kwargs to a cache key
            key_parts = [str(arg) for arg in args]
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            key = f.__name__ + ":" + ":".join(key_parts)
            
            current_time = time.time()
            
            with lock:
                # Clean expired entries
                expired_keys = [k for k, t in insertion_times.items() 
                               if current_time - t > expiration]
                for k in expired_keys:
                    if k in cache_dict:
                        del cache_dict[k]
                    if k in insertion_times:
                        del insertion_times[k]
                    if k in access_times:
                        del access_times[k]
                
                # Check if result in cache
                if key in cache_dict:
                    # Update access time
                    access_times[key] = current_time
                    return cache_dict[key]
                
                # If cache full, remove least recently used
                if len(cache_dict) >= max_size:
                    # Find least recently accessed item
                    oldest_key = min(access_times.items(), key=lambda x: x[1])[0]
                    del cache_dict[oldest_key]
                    del insertion_times[oldest_key]
                    del access_times[oldest_key]
            
            # Compute the result
            result = f(*args, **kwargs)
            
            # Cache the result
            with lock:
                cache_dict[key] = result
                insertion_times[key] = current_time
                access_times[key] = current_time
                
            return result
        return wrapped
    return decorator


# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'uptime': time.time() - app.start_time
    })


@app.route('/api/process_text', methods=['POST'])
@rate_limit(20, 60)  # 20 calls per minute
@validate_json_input()
def process_text():
    """
    Process and summarize text using the appropriate summarization model.
    
    Expected JSON input:
    {
        "text": "Text to summarize...",
        "model": "simple|advanced|hierarchical|streaming",  # Optional, default: "simple"
        "config": {                          # Optional model configuration
            "maxTokens": 100,               # For simple/advanced
            "threshold": 0.3,               # For simple/advanced
            "include_analysis": false,      # For advanced
            
            # Hierarchical Engine specific options:
            "max_concepts": 7,              # Level 1: Max concept extraction
            "max_summary_tokens": 50,       # Level 2: Max summary tokens
            "complexity_threshold": 0.7,    # Level 3: Context expansion threshold
            "max_insights": 3,              # Insight Engine: Max insights
            "min_insight_score": 0.6,       # Insight Engine: Min insight quality
            
            # Streaming Engine specific options (for unlimited text length):
            "chunk_size_words": 1000,       # Words per chunk
            "overlap_ratio": 0.15,          # Overlap between chunks
            "max_memory_mb": 512,           # Memory limit in MB
            "max_concurrent_chunks": 4,     # Parallel processing limit
            "enable_progressive_refinement": True,  # Improve results progressively
            "cache_processed_chunks": True  # Cache chunks for efficiency
        }
    }
    
    Returns:
        JSON response with summary and metadata
        
        Hierarchical Engine additionally returns:
        {
            "hierarchical_summary": {
                "level_1_concepts": [...],  # Key concepts extracted
                "level_2_core": "...",      # Core summary
                "level_3_expanded": "..."   # Expanded context (if needed)
            },
            "key_insights": [               # Important insights
                {
                    "text": "...",
                    "score": 0.95,
                    "type": "truth|wisdom|purpose|existential|love|insight"
                }
            ],
            "metadata": {
                "concept_density": 0.066,   # Concept density
                "insight_count": 2          # Number of insights found
            }
        }
        
        Streaming Engine (for unlimited text length) returns similar hierarchical 
        structure plus additional streaming metadata:
        {
            "processing_stats": {
                "total_chunks": 150,        # Number of text chunks processed
                "successful_chunks": 148,   # Successfully processed chunks
                "success_rate": 0.987       # Processing success rate
            },
            "streaming_metadata": {
                "chunks_processed": 150,    # Total chunks
                "memory_efficiency": 0.85,  # Memory usage efficiency
                "processing_speed": "4500 words/sec"  # Processing speed
            }
        }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        if 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
            
        text = data['text']
        model_type = data.get('model', 'simple').lower()
        config = data.get('config', {})
        
        # Process with appropriate model
        start_time = time.time()
        
        with summarizer_lock:
            if model_type == 'simple':
                result = simple_summarizer.process_text(text, config)
            elif model_type == 'advanced':
                # Lazy-load advanced summarizer
                global advanced_summarizer
                if advanced_summarizer is None:
                    try:
                        advanced_summarizer = MagnumOpusSUM()
                        logger.info("Initialized MagnumOpusSUM")
                    except Exception as e:
                        logger.error(f"Failed to initialize MagnumOpusSUM: {e}")
                        return jsonify({
                            'error': 'Advanced summarizer unavailable',
                            'details': str(e)
                        }), 500
                
                result = advanced_summarizer.process_text(text, config)
            elif model_type == 'hierarchical':
                # Lazy-load Hierarchical Densification Engine
                global hierarchical_engine
                if hierarchical_engine is None:
                    try:
                        hierarchical_engine = HierarchicalDensificationEngine()
                        logger.info("Hierarchical Densification Engine initialized")
                    except Exception as e:
                        logger.error(f"Failed to initialize Hierarchical Densification Engine: {e}")
                        return jsonify({
                            'error': 'Hierarchical Densification Engine unavailable',
                            'details': str(e)
                        }), 500
                
                result = hierarchical_engine.process_text(text, config)
            elif model_type == 'adaptive':
                # Use Adaptive Compression Engine
                global adaptive_engine
                if adaptive_engine is None:
                    try:
                        adaptive_engine = AdaptiveCompressionEngine()
                        logger.info("Initialized Adaptive Compression Engine")
                    except Exception as e:
                        logger.error(f"Failed to initialize Adaptive Compression Engine: {e}")
                        return jsonify({
                            'error': 'Adaptive Compression Engine unavailable',
                            'details': str(e)
                        }), 500
                
                # Get target ratio from config
                target_ratio = config.get('target_ratio', 0.2)
                
                # Compress using adaptive strategy
                compression_result = adaptive_engine.compress(text, target_ratio)
                
                # Format result to match hierarchical structure
                result = {
                    'hierarchical_summary': {
                        'level_1_concepts': compression_result.get('key_concepts', []),
                        'level_2_core': compression_result['compressed'],
                        'level_3_expanded': None  # Adaptive doesn't use level 3
                    },
                    'compression_metrics': {
                        'content_type': compression_result['content_type'],
                        'information_density': compression_result['information_density'],
                        'strategy_used': compression_result['strategy'],
                        'target_ratio': compression_result['target_ratio'],
                        'actual_ratio': compression_result['actual_ratio'],
                        'original_words': compression_result['original_length'],
                        'compressed_words': compression_result['compressed_length']
                    }
                }
            elif model_type == 'streaming':
                # Lazy-load Streaming Engine for unlimited text length
                global streaming_engine
                if streaming_engine is None:
                    try:
                        # Create streaming config from request config
                        streaming_config = StreamingConfig(
                            chunk_size_words=config.get('chunk_size_words', 1000),
                            overlap_ratio=config.get('overlap_ratio', 0.15),
                            max_memory_mb=config.get('max_memory_mb', 512),
                            max_concurrent_chunks=config.get('max_concurrent_chunks', 4),
                            enable_progressive_refinement=config.get('enable_progressive_refinement', True),
                            cache_processed_chunks=config.get('cache_processed_chunks', True)
                        )
                        streaming_engine = StreamingHierarchicalEngine(streaming_config)
                        logger.info("Streaming Hierarchical Engine initialized for unlimited text processing")
                    except Exception as e:
                        logger.error(f"Failed to initialize Streaming Engine: {e}")
                        return jsonify({
                            'error': 'Streaming Engine unavailable',
                            'details': str(e)
                        }), 500
                
                result = streaming_engine.process_streaming_text(text)
            else:
                return jsonify({'error': f'Unknown model type: {model_type}. Available: simple, advanced, hierarchical, adaptive, streaming'}), 400
        
        # Add processing metadata
        result['processing_time'] = time.time() - start_time
        result['model'] = model_type
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        return jsonify({
            'error': 'Error processing text',
            'details': str(e)
        }), 500


@app.route('/api/analyze_topics', methods=['POST'])
@rate_limit(10, 60)  # 10 calls per minute
@validate_json_input()
def analyze_topics():
    """
    Perform topic modeling on a collection of documents.
    
    Expected JSON input:
    {
        "documents": ["Document 1", "Document 2", ...],
        "num_topics": 5,  # Optional, default: 5
        "algorithm": "lda|nmf|lsa",  # Optional, default: "lda"
        "top_n_words": 10  # Optional, default: 10
    }
    
    Returns:
        JSON response with topic model results
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        if 'documents' not in data or not data['documents']:
            return jsonify({'error': 'No documents provided'}), 400
            
        documents = data['documents']
        num_topics = int(data.get('num_topics', 5))
        algorithm = data.get('algorithm', 'lda').lower()
        top_n_words = int(data.get('top_n_words', 10))
        
        # Validate parameters
        if not all(isinstance(doc, str) for doc in documents):
            return jsonify({'error': 'All documents must be strings'}), 400
            
        if num_topics < 1 or num_topics > 100:
            return jsonify({'error': 'num_topics must be between 1 and 100'}), 400
            
        if algorithm not in ['lda', 'nmf', 'lsa']:
            return jsonify({'error': f'Unsupported algorithm: {algorithm}'}), 400
            
        if top_n_words < 1 or top_n_words > 50:
            return jsonify({'error': 'top_n_words must be between 1 and 50'}), 400
        
        # Initialize topic modeler
        global topic_modeler
        topic_modeler = TopicModeler(
            n_topics=num_topics,
            algorithm=algorithm,
            n_top_words=top_n_words
        )
        
        # Fit the model
        start_time = time.time()
        topic_modeler.fit(documents)
        
        # Extract topics
        topics_summary = topic_modeler.get_topics_summary()
        
        # Transform documents to topic space
        doc_topics = topic_modeler.transform(documents)
        
        # Prepare response
        result = {
            'topics': topics_summary,
            'document_topics': doc_topics.tolist(),
            'processing_time': time.time() - start_time,
            'model_info': {
                'algorithm': algorithm,
                'num_topics': num_topics,
                'coherence': topic_modeler.coherence_scores
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error analyzing topics: {e}\n{traceback.format_exc()}")
        return jsonify({
            'error': 'Error analyzing topics',
            'details': str(e)
        }), 500


@app.route('/api/analyze_file', methods=['POST'])
@rate_limit(5, 300)  # 5 calls per 5 minutes
def analyze_file():
    """
    Analyze text file and generate summary and topic information.
    
    Expected form data:
    - file: File upload
    - model: "simple|advanced" (Optional, default: "simple")
    - num_topics: Number of topics (Optional, default: 5)
    
    Returns:
        JSON response with analysis results
    """
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save file securely
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Get model type and configuration
            model_type = request.form.get('model', 'simple').lower()
            include_topics = request.form.get('include_topics', 'true').lower() == 'true'
            include_analysis = request.form.get('include_analysis', 'false').lower() == 'true'
            
            # Configuration for the summarizer
            config = {
                'maxTokens': int(request.form.get('maxTokens', str(active_config.MAX_SUMMARY_LENGTH))),
                'include_analysis': include_analysis
            }
            
            # Create summarizer with the file
            summarizer = Summarizer(
                data_file=filepath,
                num_topics=int(request.form.get('num_topics', str(active_config.NUM_TOPICS))),
                algorithm=request.form.get('algorithm', active_config.DEFAULT_ALGORITHM),
                advanced=(model_type == 'advanced')
            )
            
            # Analyze the file
            result = summarizer.analyze(
                max_tokens=config['maxTokens'],
                include_topics=include_topics,
                include_analysis=include_analysis
            )
            
            # Add filename to result
            result['filename'] = filename
            
            return jsonify(result)
            
        finally:
            # Clean up - delete uploaded file
            try:
                os.remove(filepath)
            except Exception as e:
                logger.warning(f"Failed to delete uploaded file {filepath}: {e}")
                
    except Exception as e:
        logger.error(f"Error analyzing file: {e}\n{traceback.format_exc()}")
        return jsonify({
            'error': 'Error analyzing file',
            'details': str(e)
        }), 500


@app.route('/api/adaptive_compress', methods=['POST'])
@rate_limit(20, 60)
@validate_json_input()
def adaptive_compress():
    """
    Compress text using content-aware compression strategies.
    
    Expected JSON input:
    {
        "text": "Text to compress...",
        "target_ratio": 0.3,  # Optional, default: 0.2 (20% of original)
        "content_type": "auto|philosophical|technical|narrative|activity_log",  # Optional
        "benchmark": false  # Optional, run benchmarks
    }
    
    Returns:
        JSON response with compressed text and quality metrics
    """
    if not ADAPTIVE_AVAILABLE:
        return jsonify({
            'error': 'Adaptive compression not available',
            'details': 'Please check dependencies'
        }), 503
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        target_ratio = float(data.get('target_ratio', 0.2))
        content_type_str = data.get('content_type', 'auto')
        run_benchmark = data.get('benchmark', False)
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Initialize adaptive engine if needed
        global adaptive_engine
        if adaptive_engine is None:
            adaptive_engine = AdaptiveCompressionEngine()
            logger.info("Initialized Adaptive Compression Engine")
        
        # Determine content type
        force_type = None
        if content_type_str != 'auto':
            try:
                force_type = ContentType[content_type_str.upper()]
            except KeyError:
                return jsonify({
                    'error': f'Invalid content type: {content_type_str}'
                }), 400
        
        # Compress text
        result = adaptive_engine.compress(text, target_ratio, force_type)
        
        # Run benchmarks if requested
        benchmark_results = None
        if run_benchmark:
            benchmark_results = {}
            benchmarks = adaptive_engine.benchmark_compression()
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


@app.route('/api/life_compression/start', methods=['POST'])
@rate_limit(5, 300)
def start_life_compression():
    """Start the life compression monitoring system."""
    if not ADAPTIVE_AVAILABLE:
        return jsonify({
            'error': 'Life compression not available',
            'details': 'Please check dependencies'
        }), 503
    
    try:
        global life_system
        if life_system is None:
            life_system = LifeCompressionSystem()
        
        if not hasattr(life_system, '_is_running') or not life_system._is_running:
            life_system.start()
            life_system._is_running = True
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


@app.route('/api/chat_export/process', methods=['POST'])
@rate_limit(10, 60)
def process_chat_export():
    """
    Process chat export files to extract training insights.
    
    Expected form data:
    - file: Chat export file (JSON)
    - extract_training: bool (optional, default: false)
    """
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w+b', suffix='.json', delete=False) as tmp:
            file.save(tmp.name)
            
            try:
                from chat_intelligence_engine import ChatIntelligenceEngine
                engine = ChatIntelligenceEngine()
                
                # Process the chat export
                result = engine.process_chat_export(tmp.name)
                
                # Clean up
                os.unlink(tmp.name)
                
                return jsonify({
                    'status': 'success',
                    'filename': file.filename,
                    'processing_result': result
                })
                
            except Exception as e:
                os.unlink(tmp.name)
                raise e
                
    except Exception as e:
        logger.error(f"Error processing chat export: {e}")
        return jsonify({
            'error': 'Chat export processing failed',
            'details': str(e)  
        }), 500


@app.route('/api/life_compression/search', methods=['POST'])
@rate_limit(20, 60)
@validate_json_input()
def search_life_history():
    """
    Search through compressed life history.
    
    Expected JSON input:
    {
        "query": "search terms...",
        "limit": 10  # Optional
    }
    """
    if not ADAPTIVE_AVAILABLE:
        return jsonify({
            'error': 'Life compression not available'
        }), 503
    
    try:
        data = request.get_json()
        query = data.get('query', '')
        limit = int(data.get('limit', 10))
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        global life_system
        if life_system is None:
            life_system = LifeCompressionSystem()
        
        results = life_system.search_life_history(query)[:limit]
        
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


@app.route('/api/progressive_summarization', methods=['GET'])
def progressive_summarization_info():
    """
    Get information about the Progressive Summarization WebSocket API.
    
    Returns:
        JSON response with WebSocket connection details and usage instructions
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
        'progress_events': {
            'processing_start': 'Processing session initiated',
            'chunking_start': 'Text chunking phase started',
            'chunking_complete': 'Semantic chunks created',
            'chunk_start': 'Individual chunk processing started',
            'chunk_complete': 'Chunk processing completed with results',
            'summary_update': 'Progressive summary updated',
            'complete': 'All processing completed',
            'error': 'Processing error occurred'
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


@app.route('/api/knowledge_graph', methods=['POST'])
@rate_limit(5, 300)  # 5 calls per 5 minutes
@validate_json_input()
def generate_knowledge_graph():
    """
    Generate a knowledge graph from input text.
    
    Expected JSON input:
    {
        "text": "Text to analyze...",
        "max_nodes": 20,  # Optional, default: 20
        "min_weight": 0.1  # Optional, default: 0.1
    }
    
    Returns:
        JSON response with knowledge graph data
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        if 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
            
        text = data['text']
        max_nodes = int(data.get('max_nodes', 20))
        min_weight = float(data.get('min_weight', 0.1))
        
        # For this endpoint, we need the advanced summarizer for entity extraction
        global advanced_summarizer
        if advanced_summarizer is None:
            advanced_summarizer = MagnumOpusSUM()
            
        # Extract entities and their relationships
        entities = advanced_summarizer.identify_entities(text)
        
        # Create a simple knowledge graph using entities
        # In a production system, this would use more sophisticated NLP
        nodes = []
        edges = []
        
        # Convert entities to nodes
        for i, (entity, entity_type) in enumerate(entities[:max_nodes]):
            nodes.append({
                'id': i,
                'label': entity,
                'type': entity_type
            })
        
        # Create edges between related entities
        # This is a simplified approach - real systems would use dependency parsing
        sentences = text.split('.')
        for sentence in sentences:
            sentence_entities = []
            for i, (entity, _) in enumerate(entities):
                if entity.lower() in sentence.lower():
                    sentence_entities.append(i)
            
            # Create edges between co-occurring entities
            for i in range(len(sentence_entities)):
                for j in range(i+1, len(sentence_entities)):
                    source = sentence_entities[i]
                    target = sentence_entities[j]
                    
                    # Check if this edge already exists
                    edge_exists = False
                    for edge in edges:
                        if (edge['source'] == source and edge['target'] == target) or \
                           (edge['source'] == target and edge['target'] == source):
                            edge['weight'] += 1
                            edge_exists = True
                            break
                            
                    if not edge_exists:
                        edges.append({
                            'source': source,
                            'target': target,
                            'weight': 1
                        })
        
        # Filter edges by minimum weight
        edges = [edge for edge in edges if edge['weight'] >= min_weight]
        
        # Normalize edge weights
        max_weight = max([edge['weight'] for edge in edges]) if edges else 1
        for edge in edges:
            edge['weight'] = edge['weight'] / max_weight
        
        # Prepare response
        result = {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'node_count': len(nodes),
                'edge_count': len(edges)
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error generating knowledge graph: {e}")
        return jsonify({
            'error': 'Error generating knowledge graph',
            'details': str(e)
        }), 500


# Web interface routes

@app.route('/')
def index():
    """Render the main application page."""
    return send_from_directory('static', 'index.html')


@app.route('/dashboard')
def dashboard():
    """Render the analytics dashboard."""
    return render_template('dashboard.html')


@app.route('/api/docs')
def api_docs():
    """Render the API documentation."""
    return render_template('api_docs.html')


@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('static', path)


# AI Model Integration Endpoints
if AI_AVAILABLE:
    # Initialize AI engine
    ai_engine = HybridAIEngine()
    
    @app.route('/api/ai/models', methods=['GET'])
    def get_ai_models():
        """Get available AI models."""
        try:
            models = ai_engine.get_available_models()
            return jsonify({
                'models': models,
                'status': 'success'
            })
        except Exception as e:
            logger.error(f"Error getting AI models: {e}")
            return jsonify({
                'error': 'Failed to get models',
                'details': str(e)
            }), 500
    
    @app.route('/api/ai/keys', methods=['GET'])
    def get_api_keys():
        """Get saved API keys (masked for security)."""
        try:
            key_manager = ai_engine.key_manager
            keys = key_manager.load_api_keys()
            # Mask keys for security
            masked_keys = {provider: '••••••••' if key else '' for provider, key in keys.items()}
            return jsonify({
                'keys': masked_keys,
                'status': 'success'
            })
        except Exception as e:
            logger.error(f"Error getting API keys: {e}")
            return jsonify({
                'error': 'Failed to get API keys',
                'details': str(e)
            }), 500
    
    @app.route('/api/ai/keys', methods=['POST'])
    @rate_limit(max_calls=5, time_frame=60)
    def save_api_key():
        """Save API key for a provider."""
        try:
            data = request.get_json()
            if not data or 'provider' not in data or 'api_key' not in data:
                return jsonify({
                    'error': 'Missing provider or api_key'
                }), 400
            
            provider = data['provider']
            api_key = data['api_key']
            
            # Validate provider
            valid_providers = ['openai', 'anthropic']
            if provider not in valid_providers:
                return jsonify({
                    'error': f'Invalid provider. Must be one of: {valid_providers}'
                }), 400
            
            # Basic API key validation
            if provider == 'openai' and not api_key.startswith('sk-'):
                return jsonify({
                    'error': 'Invalid OpenAI API key format'
                }), 400
            elif provider == 'anthropic' and not api_key.startswith('sk-ant-'):
                return jsonify({
                    'error': 'Invalid Anthropic API key format'
                }), 400
            
            # Save key
            key_manager = ai_engine.key_manager
            key_manager.save_api_key(provider, api_key)
            
            # Refresh available models
            ai_engine._model_cache.clear()
            
            return jsonify({
                'status': 'success',
                'message': f'API key saved for {provider}'
            })
            
        except Exception as e:
            logger.error(f"Error saving API key: {e}")
            return jsonify({
                'error': 'Failed to save API key',
                'details': str(e)
            }), 500
    
    @app.route('/api/ai/process', methods=['POST'])
    @rate_limit(max_calls=20, time_frame=60)
    def process_with_ai():
        """Process text using AI models."""
        try:
            data = request.get_json()
            if not data or 'text' not in data:
                return jsonify({
                    'error': 'Missing text in request'
                }), 400
            
            text = data['text']
            model_id = data.get('model_id', 'traditional')
            config = data.get('config', {})
            
            # Validate text length
            if len(text) > 50000:  # Reasonable limit
                return jsonify({
                    'error': 'Text too long (max 50,000 characters)'
                }), 400
            
            # Process text
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    ai_engine.process_text(text, model_id, config)
                )
            finally:
                loop.close()
            
            return jsonify({
                'result': result,
                'status': 'success'
            })
            
        except Exception as e:
            logger.error(f"Error processing with AI: {e}")
            return jsonify({
                'error': 'Processing failed',
                'details': str(e)
            }), 500
    
    @app.route('/api/ai/compare', methods=['POST'])
    @rate_limit(max_calls=10, time_frame=60)
    def compare_ai_models():
        """Compare outputs from multiple AI models."""
        try:
            data = request.get_json()
            if not data or 'text' not in data or 'model_ids' not in data:
                return jsonify({
                    'error': 'Missing text or model_ids in request'
                }), 400
            
            text = data['text']
            model_ids = data['model_ids']
            config = data.get('config', {})
            
            # Validate inputs
            if len(text) > 30000:  # Smaller limit for comparison
                return jsonify({
                    'error': 'Text too long for comparison (max 30,000 characters)'
                }), 400
            
            if len(model_ids) > 4:  # Limit number of models to compare
                return jsonify({
                    'error': 'Too many models for comparison (max 4)'
                }), 400
            
            # Compare models
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    ai_engine.compare_models(text, model_ids, config)
                )
            finally:
                loop.close()
            
            return jsonify({
                'comparison_results': result['comparison_results'],
                'models_compared': result['models_compared'],
                'timestamp': result['timestamp'],
                'status': 'success'
            })
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return jsonify({
                'error': 'Comparison failed',
                'details': str(e)
            }), 500
    
    @app.route('/api/ai/estimate_cost', methods=['POST'])
    def estimate_cost():
        """Estimate cost for processing text with a model."""
        try:
            data = request.get_json()
            if not data or 'text' not in data or 'model_id' not in data:
                return jsonify({
                    'error': 'Missing text or model_id in request'
                }), 400
            
            text = data['text']
            model_id = data['model_id']
            
            # Estimate cost
            cost_info = ai_engine.estimate_processing_cost(text, model_id)
            
            return jsonify({
                'cost_estimate': cost_info,
                'status': 'success'
            })
            
        except Exception as e:
            logger.error(f"Error estimating cost: {e}")
            return jsonify({
                'error': 'Cost estimation failed',
                'details': str(e)
            }), 500

else:
    # Fallback endpoints when AI is not available
    @app.route('/api/ai/<path:path>', methods=['GET', 'POST'])
    def ai_not_available(path):
        """Handle AI endpoints when not available."""
        return jsonify({
            'error': 'AI models not available',
            'details': 'Install required packages: pip install openai anthropic cryptography'
        }), 503


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    logger.error(f"Server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


# Main entry point
if __name__ == '__main__':
    # Record start time for uptime calculations
    app.start_time = time.time()
    
    # Get configuration from active_config
    host = active_config.HOST
    port = active_config.PORT
    debug = active_config.DEBUG
    
    # Start server
    logger.info(f"Starting SUM server on {host}:{port} (debug={debug})...")
    app.run(host=host, port=port, debug=debug)
