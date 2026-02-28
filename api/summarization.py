"""
summarization.py - Core Summarization API Endpoints

Clean API implementation following Carmack's principles:
- Each endpoint has ONE clear purpose
- Fast response times with efficient processing
- Clear error handling and responses
- Minimal dependencies

Author: ototao
License: Apache License 2.0
"""

import time
import logging
import traceback
from flask import Blueprint, request, jsonify, Response, stream_with_context
from threading import Lock

from web.middleware import rate_limit, validate_json_input
from config import active_config
from summarization_engine import BasicSummarizationEngine, AdvancedSummarizationEngine, HierarchicalDensificationEngine
from unlimited_text_processor import process_unlimited_text, get_unlimited_processor
from api.auth import optional_api_key, require_api_key


logger = logging.getLogger(__name__)
summarization_bp = Blueprint('summarization', __name__)

# Thread-safe processing lock
_processing_lock = Lock()


@summarization_bp.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'service': 'summarization'
    })


@summarization_bp.route('/config', methods=['GET'])
def get_config():
    """Return safe configuration settings."""
    return jsonify(active_config.get_config_dict())


@summarization_bp.route('/process_text', methods=['POST'])
@optional_api_key()  # Better limits with API key
@rate_limit(20, 60)  # 20 calls per minute (public)
@validate_json_input()
def process_text():
    """
    Process and summarize text using the appropriate summarization model.

    Expected JSON input:
    {
        "text": "Text to summarize...",
        "model": "simple|advanced|hierarchical|streaming|unlimited",
        "config": {...},
        "file_path": "Optional path to large file"
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
        file_path = data.get('file_path')

        # Check for unlimited processing
        if model_type == 'unlimited' or file_path:
            # Use unlimited processor for file paths or explicit unlimited mode
            if file_path:
                result = process_unlimited_text(file_path, config)
            else:
                result = process_unlimited_text(text, config)
            result['processing_time'] = 0.0
            result['model'] = 'unlimited'
            return jsonify(result)

        # Process with appropriate model
        start_time = time.time()

        with _processing_lock:
            result = _process_with_model(text, model_type, config)
            if 'error' in result:
                return jsonify(result), result.get('status_code', 500)

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


@summarization_bp.route('/ultimate', methods=['POST'])
@optional_api_key()
@rate_limit(30, 60)
def ultimate_summarization():
    """
    The ultimate summarization endpoint that handles everything.
    Adapts to text length, requested density, and input format.

    Supported input: JSON or Multipart Form (files)
    """
    try:
        start_time = time.time()

        # 1. Parse Input
        text = None
        file_path = None
        density = 'medium'

        if request.is_json:
            data = request.get_json()
            text = data.get('text')
            density = data.get('density', 'medium')
        else:
            # File upload
            if 'file' in request.files:
                file = request.files['file']
                # Save temp file logic would go here...
                # For now assuming text for simplicity in this example
                text = file.read().decode('utf-8', errors='ignore')

            density = request.form.get('density', 'medium')

        if not text:
            return jsonify({'error': 'No content provided'}), 400

        # 2. Determine Strategy
        word_count = len(text.split())

        # Map density to engine config
        # API densities: tags, minimal, short, medium, detailed, all

        if density == 'tags':
             # Use Basic Engine just for tags/keywords
             engine = BasicSummarizationEngine()
             result = {
                 'tags': engine.generate_tag_summary(text),
                 'original_words': word_count
             }
             return jsonify(result)

        # Use Hierarchical Engine for most cases as it's the most powerful
        # But configure it based on density

        # Config for HierarchicalDensificationEngine
        config = {
            'target_density': 0.15, # Default
            'use_cache': True
        }

        if density == 'minimal':
            config['target_density'] = 0.05
            config['max_summary_tokens'] = 50
        elif density == 'short':
            config['target_density'] = 0.10
            config['max_summary_tokens'] = 150
        elif density == 'medium':
            config['target_density'] = 0.20
            config['max_summary_tokens'] = 400
        elif density == 'detailed':
            config['target_density'] = 0.35
            config['max_summary_tokens'] = 1000

        engine = HierarchicalDensificationEngine()
        result = engine.process_text(text, config)

        # Add metadata
        result['original_words'] = word_count
        result['processing_time'] = time.time() - start_time

        return jsonify(result)

    except Exception as e:
        logger.error(f"Ultimate summarization error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@summarization_bp.route('/analyze_topics', methods=['POST'])
@rate_limit(10, 60)  # 10 calls per minute
@validate_json_input()
def analyze_topics():
    """
    Perform topic modeling on a collection of documents.

    Expected JSON input:
    {
        "documents": ["Document 1", "Document 2", ...],
        "num_topics": 5,
        "algorithm": "lda|nmf|lsa",
        "top_n_words": 10
    }
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

        if not 1 <= num_topics <= 100:
            return jsonify({'error': 'num_topics must be between 1 and 100'}), 400

        if algorithm not in ['lda', 'nmf', 'lsa']:
            return jsonify({'error': f'Unsupported algorithm: {algorithm}'}), 400

        if not 1 <= top_n_words <= 50:
            return jsonify({'error': 'top_n_words must be between 1 and 50'}), 400

        # Get or create topic modeler
        from Models.topic_modeling import TopicModeler
        topic_modeler = TopicModeler(
            n_topics=num_topics,
            algorithm=algorithm,
            n_top_words=top_n_words
        )

        # Process documents
        start_time = time.time()
        topic_modeler.fit(documents)

        # Extract results
        topics_summary = topic_modeler.get_topics_summary()
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


@summarization_bp.route('/cache/stats', methods=['GET'])
def get_cache_stats():
    """Get cache statistics."""
    try:
        from smart_cache import get_cache
        cache = get_cache()
        stats = cache.get_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        return jsonify({'error': 'Failed to get cache stats'}), 500


@summarization_bp.route('/cache/clear', methods=['POST'])
@rate_limit(1, 300)  # 1 call per 5 minutes
def clear_cache():
    """Clear the cache."""
    try:
        from smart_cache import get_cache
        cache = get_cache()

        # Get optional parameters
        data = request.get_json() or {}
        text = data.get('text')
        pattern = data.get('pattern')

        cache.invalidate(text=text, pattern=pattern)

        return jsonify({
            'status': 'success',
            'message': 'Cache cleared'
        })
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({'error': 'Failed to clear cache'}), 500


@summarization_bp.route('/process_unlimited', methods=['POST'])
@require_api_key(['summarize'])  # Requires API key for large files
@rate_limit(10, 60)  # 10 calls per minute with API key
def process_unlimited():
    """
    Process text of unlimited length.

    Supports:
    - Direct text in JSON
    - File paths in JSON
    - File upload as multipart/form-data
    """
    try:
        import tempfile
        import os

        file_path = None
        text_input = None
        config = {}
        cleanup_file = False

        # Handle different input types
        if request.is_json:
            data = request.get_json()
            text_input = data.get('text')
            file_path = data.get('file_path')
            config = data.get('config', {})
        else:
            # File upload
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400

            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
                file.save(tmp.name)
                file_path = tmp.name
                cleanup_file = True

            # Get config from form data
            if 'config' in request.form:
                import json
                try:
                    config = json.loads(request.form['config'])
                except:
                    config = {}

        # Validate input
        if not text_input and not file_path:
            return jsonify({'error': 'No text or file path provided'}), 400

        # Process
        start_time = time.time()

        if file_path:
            result = process_unlimited_text(file_path, config)
        else:
            result = process_unlimited_text(text_input, config)

        # Add metadata
        result['processing_time'] = time.time() - start_time
        result['endpoint'] = 'process_unlimited'

        # Cleanup temporary file if needed
        if cleanup_file and os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except:
                pass

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error processing unlimited text: {str(e)}", exc_info=True)
        # Cleanup on error
        if 'cleanup_file' in locals() and cleanup_file and 'file_path' in locals() and os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except:
                pass
        return jsonify({
            'error': 'Error processing unlimited text',
            'details': str(e)
        }), 500


def _process_with_model(text: str, model_type: str, config: dict) -> dict:
    """
    Process text with the specified model.

    Args:
        text: Input text
        model_type: Model to use
        config: Model configuration

    Returns:
        Processing result or error dict
    """
    if model_type == 'simple':
        summarizer = BasicSummarizationEngine()
        if not summarizer:
            return {'error': 'Simple summarizer unavailable', 'status_code': 500}
        return summarizer.process_text(text, config)

    elif model_type == 'advanced':
        summarizer = AdvancedSummarizationEngine()
        if not summarizer:
            return {
                'error': 'Advanced summarizer unavailable',
                'details': 'Failed to initialize MagnumOpusSUM',
                'status_code': 500
            }
        return summarizer.process_text(text, config)

    elif model_type == 'hierarchical':
        engine = HierarchicalDensificationEngine()
        if not engine:
            return {
                'error': 'Hierarchical Densification Engine unavailable',
                'status_code': 500
            }
        return engine.process_text(text, config)

    elif model_type == 'streaming':
        # Create streaming config from request
        from streaming_engine import StreamingConfig
        streaming_config = StreamingConfig(
            chunk_size_words=config.get('chunk_size_words', 1000),
            overlap_ratio=config.get('overlap_ratio', 0.15),
            max_memory_mb=config.get('max_memory_mb', 512),
            max_concurrent_chunks=config.get('max_concurrent_chunks', 4),
            enable_progressive_refinement=config.get('enable_progressive_refinement', True),
            cache_processed_chunks=config.get('cache_processed_chunks', True)
        )

        # Get or create streaming engine with config
        from streaming_engine import StreamingHierarchicalEngine
        engine = StreamingHierarchicalEngine(streaming_config)
        return engine.process_streaming_text(text)

    else:
        return {
            'error': f'Unknown model type: {model_type}',
            'available_models': ['simple', 'advanced', 'hierarchical', 'streaming', 'unlimited'],
            'status_code': 400
        }

@summarization_bp.route('/stream_unlimited', methods=['POST'])
@rate_limit(10, 60)
def stream_unlimited():
    """
    Stream processing of text of unlimited length.

    Supports:
    - Direct text in JSON
    - File paths in JSON
    - File upload as multipart/form-data
    """
    try:
        import tempfile
        import os
        import json

        file_path = None
        text_input = None
        config = {}
        cleanup_file = False

        # Handle different input types
        if request.is_json:
            data = request.get_json()
            text_input = data.get('text')
            file_path = data.get('file_path')
            config = data.get('config', {})
        else:
            # File upload
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400

            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
                file.save(tmp.name)
                file_path = tmp.name
                cleanup_file = True

            # Get config from form data
            if 'config' in request.form:
                try:
                    config = json.loads(request.form['config'])
                except:
                    config = {}

        # Validate input
        if not text_input and not file_path:
            return jsonify({'error': 'No text or file path provided'}), 400

        def generate():
            try:
                processor = get_unlimited_processor()
                # Determine input to pass to processor
                input_data = file_path if file_path else text_input

                yield f"data: {json.dumps({'status': 'initializing'})}\n\n"

                for event in processor.process_text_stream(input_data, config):
                    yield f"data: {json.dumps(event)}\n\n"

                yield f"data: {json.dumps({'status': 'complete'})}\n\n"

            except Exception as e:
                logger.error(f"Error in stream processing: {str(e)}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            finally:
                # Cleanup temporary file if needed
                if cleanup_file and file_path and os.path.exists(file_path):
                    try:
                        os.unlink(file_path)
                    except:
                        pass

        return Response(stream_with_context(generate()), mimetype='text/event-stream')

    except Exception as e:
        logger.error(f"Error initializing stream: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Error initializing stream',
            'details': str(e)
        }), 500
