#!/usr/bin/env python3
"""
main_enhanced.py - Enhanced SUM Platform with Multi-Modal Processing

Enhanced version of the SUM web service with comprehensive multi-modal processing
capabilities, local AI integration, and advanced document handling.

New Features:
- Multi-modal document processing (PDF, DOCX, images, HTML, Markdown)
- Local AI model integration via Ollama
- Vision-language model support for image analysis
- Advanced OCR capabilities
- Real-time processing status
- Model performance monitoring
- Privacy-focused local processing options

Author: ototao
License: Apache License 2.0
"""

import os
import sys
import logging
import json
import time
from functools import wraps
from typing import Dict, List, Any, Optional, Tuple, Union
import traceback
import tempfile
from threading import Lock
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_from_directory, abort, flash, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

# Import existing SUM components
from summarization_engine import SimpleSUM, MagnumOpusSUM, HierarchicalDensificationEngine
from streaming_engine import StreamingHierarchicalEngine, StreamingConfig
from Utils.data_loader import DataLoader
from Models.topic_modeling import TopicModeler
from Models.summarizer import Summarizer
from config import active_config

# Import new multi-modal components
try:
    from multimodal_processor import MultiModalProcessor, ContentType, ProcessingResult
    from ollama_manager import OllamaManager, ProcessingRequest, ModelType
    MULTIMODAL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Multi-modal processing not available: {e}")
    MULTIMODAL_AVAILABLE = False

# Import existing optional components
try:
    from adaptive_compression import AdaptiveCompressionEngine
    from life_compression_system import LifeCompressionSystem
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False

try:
    from ai_models import HybridAIEngine, SecureKeyManager, AVAILABLE_MODELS
    AI_AVAILABLE = True
except ImportError:
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

# Configure application
app.config['SECRET_KEY'] = active_config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = active_config.MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = active_config.UPLOADS_DIR
app.config['ALLOWED_EXTENSIONS'] = active_config.ALLOWED_EXTENSIONS
app.config['DEBUG'] = active_config.DEBUG

# Extend allowed extensions for multi-modal processing
if MULTIMODAL_AVAILABLE:
    app.config['ALLOWED_EXTENSIONS'].update({
        'pdf', 'docx', 'doc', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff',
        'html', 'htm', 'md', 'markdown'
    })

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('temp', exist_ok=True)

# Initialize components
simple_summarizer = SimpleSUM()
hierarchical_engine = None
multimodal_processor = None
ollama_manager = None

# Concurrency control
processing_lock = Lock()
cache = {}
cache_lock = Lock()

# Processing statistics
processing_stats = {
    'total_requests': 0,
    'successful_requests': 0,
    'failed_requests': 0,
    'processing_time_total': 0.0,
    'average_processing_time': 0.0,
    'files_processed': 0,
    'content_types_processed': {},
    'models_used': {}
}
stats_lock = Lock()


def init_components():
    """Initialize components lazily when needed."""
    global hierarchical_engine, multimodal_processor, ollama_manager
    
    if hierarchical_engine is None:
        hierarchical_engine = HierarchicalDensificationEngine()
        logger.info("Hierarchical engine initialized")
    
    if MULTIMODAL_AVAILABLE and multimodal_processor is None:
        multimodal_processor = MultiModalProcessor()
        logger.info("Multi-modal processor initialized")
    
    if MULTIMODAL_AVAILABLE and ollama_manager is None:
        ollama_manager = OllamaManager()
        logger.info("Ollama manager initialized")


def update_stats(processing_time: float, content_type: str = None, 
                model_used: str = None, success: bool = True):
    """Update processing statistics."""
    with stats_lock:
        processing_stats['total_requests'] += 1
        if success:
            processing_stats['successful_requests'] += 1
        else:
            processing_stats['failed_requests'] += 1
        
        processing_stats['processing_time_total'] += processing_time
        processing_stats['average_processing_time'] = (
            processing_stats['processing_time_total'] / processing_stats['total_requests']
        )
        
        if content_type:
            processing_stats['content_types_processed'][content_type] = (
                processing_stats['content_types_processed'].get(content_type, 0) + 1
            )
        
        if model_used:
            processing_stats['models_used'][model_used] = (
                processing_stats['models_used'].get(model_used, 0) + 1
            )


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def generate_cache_key(content: str, config: Dict[str, Any]) -> str:
    """Generate cache key for results."""
    import hashlib
    content_hash = hashlib.md5(content.encode()).hexdigest()
    config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return f"{content_hash}_{config_hash}"


@app.route('/')
def index():
    """Main interface with multi-modal capabilities."""
    init_components()
    
    # Get system capabilities
    capabilities = {
        'multimodal_available': MULTIMODAL_AVAILABLE,
        'ai_available': AI_AVAILABLE,
        'adaptive_available': ADAPTIVE_AVAILABLE,
        'supported_formats': list(app.config['ALLOWED_EXTENSIONS']) if MULTIMODAL_AVAILABLE else ['txt'],
        'local_models': list(ollama_manager.available_models.keys()) if ollama_manager else [],
        'processing_stats': processing_stats
    }
    
    return render_template('index_enhanced.html', capabilities=capabilities)


@app.route('/api/system/status')
def system_status():
    """Get comprehensive system status."""
    init_components()
    
    status = {
        'system': {
            'multimodal_available': MULTIMODAL_AVAILABLE,
            'ai_available': AI_AVAILABLE,
            'adaptive_available': ADAPTIVE_AVAILABLE,
            'hierarchical_engine': hierarchical_engine is not None,
        },
        'processing_stats': processing_stats,
        'capabilities': {}
    }
    
    if MULTIMODAL_AVAILABLE and multimodal_processor:
        status['capabilities']['multimodal'] = multimodal_processor.get_processing_stats()
    
    if ollama_manager:
        status['capabilities']['local_ai'] = ollama_manager.get_model_status()
    
    return jsonify(status)


@app.route('/api/process/text', methods=['POST'])
def process_text():
    """Enhanced text processing endpoint."""
    start_time = time.time()
    init_components()
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        processing_config = data.get('config', {})
        use_local_ai = data.get('use_local_ai', False)
        model_preference = data.get('model', None)
        
        # Check cache
        cache_key = generate_cache_key(text, processing_config)
        with cache_lock:
            if cache_key in cache:
                cached_result = cache[cache_key]
                cached_result['cached'] = True
                return jsonify(cached_result)
        
        # Process with hierarchical engine
        with processing_lock:
            hierarchical_result = hierarchical_engine.process_text(text, processing_config)
        
        # Enhance with local AI if requested and available
        if use_local_ai and ollama_manager and ollama_manager.available_models:
            try:
                local_request = ProcessingRequest(
                    text=text,
                    model_name=model_preference,
                    task_type=processing_config.get('task_type', 'summarization'),
                    max_tokens=processing_config.get('max_tokens', 200),
                    temperature=processing_config.get('temperature', 0.3)
                )
                
                local_response = ollama_manager.process_text(local_request)
                hierarchical_result['local_ai_analysis'] = {
                    'summary': local_response.response,
                    'model_used': local_response.model_used,
                    'processing_time': local_response.processing_time,
                    'confidence': local_response.confidence_score
                }
                
                update_stats(
                    processing_time=time.time() - start_time,
                    content_type='text',
                    model_used=local_response.model_used,
                    success=True
                )
                
            except Exception as e:
                logger.warning(f"Local AI processing failed: {e}")
                hierarchical_result['local_ai_error'] = str(e)
        
        # Add metadata
        hierarchical_result.update({
            'processing_time': time.time() - start_time,
            'timestamp': time.time(),
            'cached': False,
            'system_info': {
                'hierarchical_engine': True,
                'local_ai_used': use_local_ai and 'local_ai_analysis' in hierarchical_result
            }
        })
        
        # Cache result
        with cache_lock:
            cache[cache_key] = hierarchical_result
            
        update_stats(
            processing_time=time.time() - start_time,
            content_type='text',
            success=True
        )
        
        return jsonify(hierarchical_result)
        
    except Exception as e:
        logger.error(f"Text processing error: {e}")
        update_stats(processing_time=time.time() - start_time, success=False)
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/file', methods=['POST'])
def process_file():
    """Enhanced file processing endpoint with multi-modal support."""
    start_time = time.time()
    init_components()
    
    if not MULTIMODAL_AVAILABLE:
        return jsonify({'error': 'Multi-modal processing not available'}), 503
    
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not supported. Allowed: {list(app.config["ALLOWED_EXTENSIONS"])}'}), 400
        
        # Get processing options
        use_local_ai = request.form.get('use_local_ai', 'false').lower() == 'true'
        use_vision = request.form.get('use_vision', 'true').lower() == 'true'
        hierarchical_config = json.loads(request.form.get('hierarchical_config', '{}'))
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{int(time.time())}_{filename}")
        file.save(file_path)
        
        try:
            # Process with multi-modal processor
            processing_result = multimodal_processor.process_file(
                file_path,
                use_vision=use_vision,
                hierarchical_config=hierarchical_config
            )
            
            # Enhance with local AI if requested
            if (use_local_ai and ollama_manager and 
                processing_result.extracted_text and 
                len(processing_result.extracted_text.strip()) > 20):
                
                try:
                    local_request = ProcessingRequest(
                        text=processing_result.extracted_text,
                        task_type='summarization',
                        max_tokens=300
                    )
                    
                    local_response = ollama_manager.process_text(local_request)
                    processing_result.metadata['local_ai_analysis'] = {
                        'summary': local_response.response,
                        'model_used': local_response.model_used,
                        'processing_time': local_response.processing_time,
                        'confidence': local_response.confidence_score
                    }
                    
                except Exception as e:
                    logger.warning(f"Local AI enhancement failed: {e}")
                    processing_result.metadata['local_ai_error'] = str(e)
            
            # Prepare response
            response_data = {
                'content_type': processing_result.content_type.value,
                'extracted_text': processing_result.extracted_text,
                'metadata': processing_result.metadata,
                'processing_time': processing_result.processing_time,
                'confidence_score': processing_result.confidence_score,
                'error_message': processing_result.error_message,
                'filename': filename,
                'file_size': os.path.getsize(file_path),
                'timestamp': time.time()
            }
            
            update_stats(
                processing_time=time.time() - start_time,
                content_type=processing_result.content_type.value,
                success=processing_result.error_message is None
            )
            
            return jsonify(response_data)
            
        finally:
            # Clean up uploaded file
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Could not remove temporary file: {e}")
        
    except Exception as e:
        logger.error(f"File processing error: {e}")
        update_stats(processing_time=time.time() - start_time, success=False)
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/batch', methods=['POST'])
def process_batch():
    """Process multiple files in batch."""
    start_time = time.time()
    init_components()
    
    if not MULTIMODAL_AVAILABLE:
        return jsonify({'error': 'Multi-modal processing not available'}), 503
    
    try:
        files = request.files.getlist('files')
        if not files or len(files) == 0:
            return jsonify({'error': 'No files uploaded'}), 400
        
        use_local_ai = request.form.get('use_local_ai', 'false').lower() == 'true'
        hierarchical_config = json.loads(request.form.get('hierarchical_config', '{}'))
        
        results = []
        temp_files = []
        
        try:
            # Save all files temporarily
            for file in files:
                if file.filename and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(
                        app.config['UPLOAD_FOLDER'], 
                        f"batch_{int(time.time())}_{filename}"
                    )
                    file.save(file_path)
                    temp_files.append((file_path, filename))
            
            # Process all files
            file_paths = [path for path, _ in temp_files]
            processing_results = multimodal_processor.process_batch(
                file_paths,
                hierarchical_config=hierarchical_config
            )
            
            # Format results
            for i, (result, (file_path, original_filename)) in enumerate(zip(processing_results, temp_files)):
                file_result = {
                    'filename': original_filename,
                    'content_type': result.content_type.value,
                    'extracted_text': result.extracted_text,
                    'metadata': result.metadata,
                    'processing_time': result.processing_time,
                    'confidence_score': result.confidence_score,
                    'error_message': result.error_message,
                    'file_size': os.path.getsize(file_path)
                }
                
                # Add local AI analysis if requested
                if (use_local_ai and ollama_manager and 
                    result.extracted_text and len(result.extracted_text.strip()) > 20):
                    
                    try:
                        local_request = ProcessingRequest(
                            text=result.extracted_text[:2000],  # Limit for batch processing
                            task_type='quick_summary',
                            max_tokens=100
                        )
                        
                        local_response = ollama_manager.process_text(local_request)
                        file_result['local_ai_summary'] = local_response.response
                        file_result['local_ai_model'] = local_response.model_used
                        
                    except Exception as e:
                        logger.warning(f"Local AI failed for {original_filename}: {e}")
                
                results.append(file_result)
            
            batch_result = {
                'results': results,
                'total_files': len(results),
                'successful_files': len([r for r in results if not r['error_message']]),
                'total_processing_time': time.time() - start_time,
                'timestamp': time.time()
            }
            
            update_stats(
                processing_time=time.time() - start_time,
                content_type='batch',
                success=True
            )
            
            return jsonify(batch_result)
            
        finally:
            # Clean up temporary files
            for file_path, _ in temp_files:
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Could not remove {file_path}: {e}")
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        update_stats(processing_time=time.time() - start_time, success=False)
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/local')
def list_local_models():
    """List available local AI models."""
    init_components()
    
    if not ollama_manager:
        return jsonify({'error': 'Local AI not available'}), 503
    
    models_info = []
    for name, info in ollama_manager.available_models.items():
        models_info.append({
            'name': info.name,
            'type': info.type.value,
            'size': info.size,
            'capabilities': info.capabilities,
            'parameters': info.parameters,
            'performance_score': info.performance_score,
            'memory_usage': info.memory_usage,
            'response_time': info.response_time
        })
    
    return jsonify({
        'models': models_info,
        'total_models': len(models_info),
        'ollama_status': ollama_manager.get_model_status()
    })


@app.route('/api/models/benchmark', methods=['POST'])
def benchmark_models():
    """Benchmark local models performance."""
    init_components()
    
    if not ollama_manager:
        return jsonify({'error': 'Local AI not available'}), 503
    
    try:
        data = request.get_json() or {}
        sample_text = data.get('sample_text')
        
        results = ollama_manager.benchmark_models(sample_text)
        
        return jsonify({
            'benchmark_results': results,
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def get_stats():
    """Get detailed processing statistics."""
    return jsonify({
        'processing_stats': processing_stats,
        'cache_size': len(cache),
        'system_capabilities': {
            'multimodal_available': MULTIMODAL_AVAILABLE,
            'ai_available': AI_AVAILABLE,
            'adaptive_available': ADAPTIVE_AVAILABLE,
            'local_models': len(ollama_manager.available_models) if ollama_manager else 0
        },
        'timestamp': time.time()
    })


@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear processing cache."""
    with cache_lock:
        cache_size = len(cache)
        cache.clear()
    
    logger.info(f"Cleared {cache_size} cached results")
    return jsonify({'message': f'Cleared {cache_size} cached results'})


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large'}), 413


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Initialize components on startup
    init_components()
    
    # Print startup information
    print("🚀 Enhanced SUM Platform Starting Up")
    print("=" * 50)
    print(f"Multi-modal Processing: {'✅' if MULTIMODAL_AVAILABLE else '❌'}")
    print(f"Local AI (Ollama): {'✅' if ollama_manager and ollama_manager.ollama_client else '❌'}")
    print(f"Advanced AI Models: {'✅' if AI_AVAILABLE else '❌'}")
    print(f"Adaptive Compression: {'✅' if ADAPTIVE_AVAILABLE else '❌'}")
    
    if MULTIMODAL_AVAILABLE and multimodal_processor:
        formats = multimodal_processor.get_supported_formats()
        print(f"Supported Formats: {', '.join(formats)}")
    
    if ollama_manager and ollama_manager.available_models:
        print(f"Local Models: {len(ollama_manager.available_models)} available")
        for name, info in list(ollama_manager.available_models.items())[:3]:
            print(f"  - {name} ({info.type.value}, {info.size})")
        if len(ollama_manager.available_models) > 3:
            print(f"  ... and {len(ollama_manager.available_models) - 3} more")
    
    print("=" * 50)
    print(f"🌐 Server starting on http://localhost:{os.getenv('FLASK_PORT', 5000)}")
    
    # Start the application
    port = int(os.getenv('FLASK_PORT', 5000))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=active_config.DEBUG,
        threaded=True
    )