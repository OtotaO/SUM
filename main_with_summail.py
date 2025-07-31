#!/usr/bin/env python3
"""
main_with_summail.py - Unified SUM Platform with Email Processing

Enhanced version of the SUM platform that includes all existing capabilities
plus the new SumMail email processing mode as an integrated component.

Modes:
- Text Processing: Hierarchical text summarization
- Multi-Modal: Process PDFs, images, documents
- Email Mode (SumMail): Intelligent email compression and management
- Streaming: Handle unlimited text sizes
- Local AI: Privacy-focused processing with Ollama

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
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, request, jsonify, render_template, send_from_directory, abort, flash, redirect, url_for, session, Response
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

# Import existing SUM components
from SUM import SimpleSUM, MagnumOpusSUM, HierarchicalDensificationEngine
from StreamingEngine import StreamingHierarchicalEngine, StreamingConfig
from Utils.data_loader import DataLoader
from Models.topic_modeling import TopicModeler
from Models.summarizer import Summarizer
from config import active_config

# Import new components
try:
    from multimodal_processor import MultiModalProcessor, ContentType, ProcessingResult
    from ollama_manager import OllamaManager, ProcessingRequest, ModelType
    MULTIMODAL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Multi-modal processing not available: {e}")
    MULTIMODAL_AVAILABLE = False

# Import SumMail components
try:
    from summail_engine import (
        SumMailEngine, EmailCategory, Priority, EmailDigest,
        CompressedEmail, SoftwareInfo
    )
    SUMMAIL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"SumMail not available: {e}")
    SUMMAIL_AVAILABLE = False

# Import Knowledge OS components
try:
    from knowledge_os import (
        KnowledgeOperatingSystem, Thought, KnowledgeCluster,
        CaptureSession
    )
    from knowledge_os_interface import init_knowledge_os
    KNOWLEDGE_OS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Knowledge OS not available: {e}")
    KNOWLEDGE_OS_AVAILABLE = False

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
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', active_config.SECRET_KEY)
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
os.makedirs('templates', exist_ok=True)

# Initialize components
simple_summarizer = SimpleSUM()
hierarchical_engine = None
multimodal_processor = None
ollama_manager = None
user_summail_engines = {}  # Store SumMail engines per user session

# Concurrency control
processing_lock = Lock()
cache = {}
cache_lock = Lock()

# Background processing
executor = ThreadPoolExecutor(max_workers=4)

# Processing statistics
processing_stats = {
    'total_requests': 0,
    'successful_requests': 0,
    'failed_requests': 0,
    'processing_time_total': 0.0,
    'average_processing_time': 0.0,
    'files_processed': 0,
    'emails_processed': 0,
    'content_types_processed': {},
    'models_used': {},
    'modes_used': {
        'text': 0,
        'multimodal': 0,
        'email': 0,
        'streaming': 0
    }
}
stats_lock = Lock()

# Email processing status tracking
email_processing_status = {}  # session_id -> status


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


def update_stats(processing_time: float, mode: str = 'text', 
                content_type: str = None, model_used: str = None, 
                success: bool = True):
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
        
        # Mode tracking
        if mode in processing_stats['modes_used']:
            processing_stats['modes_used'][mode] += 1
        
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


def get_session_id() -> str:
    """Get or create session ID."""
    if 'session_id' not in session:
        session['session_id'] = f"session_{int(time.time())}_{os.urandom(8).hex()}"
    return session['session_id']


@app.route('/')
def index():
    """Main unified interface."""
    init_components()
    
    # Get system capabilities
    capabilities = {
        'multimodal_available': MULTIMODAL_AVAILABLE,
        'ai_available': AI_AVAILABLE,
        'adaptive_available': ADAPTIVE_AVAILABLE,
        'summail_available': SUMMAIL_AVAILABLE,
        'knowledge_os_available': KNOWLEDGE_OS_AVAILABLE,
        'supported_formats': list(app.config['ALLOWED_EXTENSIONS']) if MULTIMODAL_AVAILABLE else ['txt'],
        'local_models': list(ollama_manager.available_models.keys()) if ollama_manager else [],
        'processing_stats': processing_stats,
        'available_modes': {
            'text': True,
            'multimodal': MULTIMODAL_AVAILABLE,
            'email': SUMMAIL_AVAILABLE,
            'knowledge': KNOWLEDGE_OS_AVAILABLE,
            'streaming': True,
            'local_ai': ollama_manager is not None and len(ollama_manager.available_models) > 0
        }
    }
    
    return render_template('index_unified.html', capabilities=capabilities)


@app.route('/api/system/status')
def system_status():
    """Get comprehensive system status."""
    init_components()
    
    status = {
        'system': {
            'multimodal_available': MULTIMODAL_AVAILABLE,
            'ai_available': AI_AVAILABLE,
            'adaptive_available': ADAPTIVE_AVAILABLE,
            'summail_available': SUMMAIL_AVAILABLE,
            'knowledge_os_available': KNOWLEDGE_OS_AVAILABLE,
            'hierarchical_engine': hierarchical_engine is not None,
        },
        'processing_stats': processing_stats,
        'capabilities': {},
        'active_modes': []
    }
    
    if MULTIMODAL_AVAILABLE and multimodal_processor:
        status['capabilities']['multimodal'] = multimodal_processor.get_processing_stats()
        status['active_modes'].append('multimodal')
    
    if ollama_manager:
        status['capabilities']['local_ai'] = ollama_manager.get_model_status()
        if ollama_manager.available_models:
            status['active_modes'].append('local_ai')
    
    if SUMMAIL_AVAILABLE:
        status['active_modes'].append('email')
        status['capabilities']['summail'] = {
            'engine_available': True,
            'categories': [cat.value for cat in EmailCategory],
            'priorities': [pri.value for pri in Priority]
        }
    
    if KNOWLEDGE_OS_AVAILABLE:
        status['active_modes'].append('knowledge')
        status['capabilities']['knowledge_os'] = {
            'engine_available': True,
            'features': ['thought_capture', 'background_intelligence', 'auto_densification', 'insight_generation']
        }
    
    return jsonify(status)


# ========== EMAIL PROCESSING ROUTES (SumMail) ==========

@app.route('/api/summail/connect', methods=['POST'])
def summail_connect():
    """Connect to email account for SumMail processing."""
    if not SUMMAIL_AVAILABLE:
        return jsonify({'error': 'SumMail not available'}), 503
    
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        imap_server = data.get('imap_server')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        # Create SumMail engine for this session
        session_id = get_session_id()
        engine = SumMailEngine({'use_local_ai': True})
        
        # Connect to email account
        if engine.connect_email_account(email, password, imap_server):
            user_summail_engines[session_id] = engine
            session['email_connected'] = True
            session['email_address'] = email
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'email': email,
                'message': 'Successfully connected to email account'
            })
        else:
            return jsonify({'error': 'Failed to connect to email account'}), 401
            
    except Exception as e:
        logger.error(f"Email connection error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/summail/process', methods=['POST'])
def summail_process():
    """Process emails with SumMail engine."""
    session_id = get_session_id()
    
    if session_id not in user_summail_engines:
        return jsonify({'error': 'Email not connected'}), 401
    
    engine = user_summail_engines[session_id]
    
    try:
        data = request.get_json()
        folder = data.get('folder', 'INBOX')
        limit = data.get('limit', 100)
        start_date = data.get('start_date')
        categories = data.get('categories', [])
        
        # Start background processing
        email_processing_status[session_id] = {
            'status': 'processing',
            'processed': 0,
            'total': 0,
            'start_time': time.time()
        }
        
        # Submit processing task
        future = executor.submit(
            _process_emails_background,
            engine, session_id, folder, limit, start_date, categories
        )
        
        update_stats(0, mode='email', success=True)
        
        return jsonify({
            'status': 'processing',
            'session_id': session_id,
            'message': 'Email processing started'
        })
        
    except Exception as e:
        logger.error(f"Email processing error: {e}")
        update_stats(0, mode='email', success=False)
        return jsonify({'error': str(e)}), 500


def _process_emails_background(engine: SumMailEngine, session_id: str, 
                              folder: str, limit: int, 
                              start_date: Optional[str], 
                              categories: List[str]):
    """Background email processing task."""
    try:
        # Select folder
        engine.imap.select(folder)
        
        # Build search criteria
        criteria = []
        if start_date:
            date_obj = datetime.fromisoformat(start_date)
            criteria.append(f'SINCE {date_obj.strftime("%d-%b-%Y")}')
        
        search_string = ' '.join(criteria) if criteria else 'ALL'
        
        # Search emails
        _, message_ids = engine.imap.search(None, search_string)
        message_ids = message_ids[0].split()[-limit:]  # Get last N emails
        
        email_processing_status[session_id]['total'] = len(message_ids)
        
        # Process each email
        processed_count = 0
        for msg_id in message_ids:
            try:
                # Fetch email
                _, msg_data = engine.imap.fetch(msg_id, '(RFC822)')
                email_data = msg_data[0][1]
                
                # Process email
                compressed = engine.process_email(email_data)
                
                # Update progress
                processed_count += 1
                email_processing_status[session_id]['processed'] = processed_count
                
                # Update global stats
                with stats_lock:
                    processing_stats['emails_processed'] += 1
                
            except Exception as e:
                logger.error(f"Error processing email {msg_id}: {e}")
        
        # Mark as complete
        email_processing_status[session_id]['status'] = 'completed'
        email_processing_status[session_id]['end_time'] = time.time()
        email_processing_status[session_id]['processing_time'] = (
            email_processing_status[session_id]['end_time'] - 
            email_processing_status[session_id]['start_time']
        )
        
    except Exception as e:
        logger.error(f"Background email processing error: {e}")
        email_processing_status[session_id]['status'] = 'error'
        email_processing_status[session_id]['error'] = str(e)


@app.route('/api/summail/status')
def summail_status():
    """Get email processing status."""
    session_id = get_session_id()
    status = email_processing_status.get(session_id, {'status': 'idle'})
    
    # Calculate progress percentage
    if status.get('total', 0) > 0:
        status['progress'] = (status.get('processed', 0) / status['total']) * 100
    else:
        status['progress'] = 0
    
    return jsonify(status)


@app.route('/api/summail/digest', methods=['POST'])
def summail_digest():
    """Generate email digest."""
    session_id = get_session_id()
    
    if session_id not in user_summail_engines:
        return jsonify({'error': 'Email not connected'}), 401
    
    engine = user_summail_engines[session_id]
    
    try:
        data = request.get_json()
        
        # Parse date range
        start_date = datetime.fromisoformat(data.get('start_date', 
                                                    (datetime.now() - timedelta(days=7)).isoformat()))
        end_date = datetime.fromisoformat(data.get('end_date', 
                                                  datetime.now().isoformat()))
        
        # Parse categories
        categories = None
        if data.get('categories'):
            categories = [EmailCategory[cat.upper()] for cat in data['categories']]
        
        # Generate digest
        digest = engine.generate_digest(start_date, end_date, categories)
        
        # Convert to JSON-serializable format
        digest_data = {
            'period_start': digest.period_start.isoformat(),
            'period_end': digest.period_end.isoformat(),
            'total_emails': digest.total_emails,
            'categories_breakdown': {
                cat.value: count for cat, count in digest.categories_breakdown.items()
            },
            'software_updates': [
                {
                    'name': update.name,
                    'current_version': update.current_version,
                    'latest_version': update.latest_version,
                    'release_date': update.release_date.isoformat() if update.release_date else None
                }
                for update in digest.software_updates
            ],
            'action_items': digest.action_items,
            'compressed_newsletters': digest.compressed_newsletters,
            'important_threads': digest.important_threads,
            'financial_summary': digest.financial_summary,
            'compression_stats': {
                'total_original_size': sum(e.metadata.size for e in engine.processed_emails.values()),
                'total_compressed_size': sum(
                    e.metadata.size * (1 - e.compression_ratio) 
                    for e in engine.processed_emails.values()
                ),
                'average_compression': sum(
                    e.compression_ratio for e in engine.processed_emails.values()
                ) / max(len(engine.processed_emails), 1)
            }
        }
        
        return jsonify(digest_data)
        
    except Exception as e:
        logger.error(f"Digest generation error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/summail/search', methods=['POST'])
def summail_search():
    """Search processed emails."""
    session_id = get_session_id()
    
    if session_id not in user_summail_engines:
        return jsonify({'error': 'Email not connected'}), 401
    
    engine = user_summail_engines[session_id]
    
    try:
        data = request.get_json()
        query = data.get('query', '')
        category = data.get('category')
        priority = data.get('priority')
        sender = data.get('sender')
        
        results = []
        
        for msg_id, email in engine.processed_emails.items():
            # Apply filters
            if category and email.metadata.category.value != category:
                continue
            if priority and email.metadata.priority.value != priority:
                continue
            if sender and sender.lower() not in email.metadata.sender.lower():
                continue
            
            # Search in content
            if query:
                query_lower = query.lower()
                if (query_lower not in email.metadata.subject.lower() and
                    query_lower not in email.summary.lower() and
                    not any(query_lower in point.lower() for point in email.key_points)):
                    continue
            
            # Add to results
            results.append({
                'message_id': msg_id,
                'subject': email.metadata.subject,
                'sender': email.metadata.sender,
                'date': email.metadata.date.isoformat(),
                'category': email.metadata.category.value,
                'priority': email.metadata.priority.value,
                'summary': email.summary,
                'key_points': email.key_points,
                'compression_ratio': email.compression_ratio,
                'has_attachments': email.metadata.has_attachments,
                'action_items': email.metadata.action_items
            })
        
        # Sort by date (newest first)
        results.sort(key=lambda x: x['date'], reverse=True)
        
        return jsonify({
            'results': results[:50],  # Limit to 50 results
            'total': len(results)
        })
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/summail/disconnect', methods=['POST'])
def summail_disconnect():
    """Disconnect from email account."""
    session_id = get_session_id()
    
    if session_id in user_summail_engines:
        try:
            engine = user_summail_engines[session_id]
            if hasattr(engine, 'imap') and engine.imap:
                engine.imap.logout()
        except:
            pass
        
        del user_summail_engines[session_id]
    
    if session_id in email_processing_status:
        del email_processing_status[session_id]
    
    session.pop('email_connected', None)
    session.pop('email_address', None)
    
    return jsonify({'success': True, 'message': 'Disconnected from email'})


# ========== KNOWLEDGE OS ROUTES ==========

@app.route('/knowledge')
def knowledge_interface():
    """Knowledge Operating System main interface."""
    if not KNOWLEDGE_OS_AVAILABLE:
        return render_template('error.html', 
                             error_title="Knowledge OS Not Available",
                             error_message="The Knowledge Operating System is not installed or configured properly.")
    
    # Initialize Knowledge OS if needed
    init_knowledge_os()
    return render_template('knowledge_os.html')

@app.route('/api/knowledge/capture', methods=['POST'])
def api_knowledge_capture():
    """API endpoint to capture thoughts."""
    if not KNOWLEDGE_OS_AVAILABLE:
        return jsonify({'error': 'Knowledge OS not available'}), 503
    
    try:
        init_knowledge_os()
        from knowledge_os_interface import knowledge_os
        
        data = request.get_json()
        content = data.get('content', '').strip()
        
        if not content:
            return jsonify({'success': False, 'error': 'No content provided'})
        
        # Capture the thought
        thought = knowledge_os.capture_thought(content)
        
        if thought:
            update_stats(0.1, 'knowledge', 'thought', 'knowledge_os')
            return jsonify({
                'success': True,
                'thought_id': thought.id,
                'message': 'Thought captured successfully'
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to capture thought'})
            
    except Exception as e:
        logger.error(f"Error capturing thought: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/knowledge/prompt')
def api_knowledge_prompt():
    """Get a contextual capture prompt."""
    if not KNOWLEDGE_OS_AVAILABLE:
        return jsonify({'error': 'Knowledge OS not available'}), 503
    
    try:
        init_knowledge_os()
        from knowledge_os_interface import knowledge_os
        
        prompt = knowledge_os.get_capture_prompt()
        return jsonify({'success': True, 'prompt': prompt})
    except Exception as e:
        logger.error(f"Error getting prompt: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/knowledge/recent-thoughts')
def api_knowledge_recent():
    """Get recent thoughts."""
    if not KNOWLEDGE_OS_AVAILABLE:
        return jsonify({'error': 'Knowledge OS not available'}), 503
    
    try:
        init_knowledge_os()
        from knowledge_os_interface import knowledge_os
        
        limit = int(request.args.get('limit', 5))
        thoughts = knowledge_os.get_recent_thoughts(limit=limit)
        
        thoughts_data = []
        for thought in thoughts:
            thoughts_data.append({
                'id': thought.id,
                'content': thought.content,
                'timestamp': thought.timestamp.isoformat(),
                'tags': thought.tags,
                'concepts': thought.concepts,
                'importance': thought.importance,
                'word_count': thought.word_count,
                'connections': len(thought.connections)
            })
        
        return jsonify({'success': True, 'thoughts': thoughts_data})
    except Exception as e:
        logger.error(f"Error getting recent thoughts: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/knowledge/insights')
def api_knowledge_insights():
    """Get system insights and analytics."""
    if not KNOWLEDGE_OS_AVAILABLE:
        return jsonify({'error': 'Knowledge OS not available'}), 503
    
    try:
        init_knowledge_os()
        from knowledge_os_interface import knowledge_os
        
        insights = knowledge_os.get_system_insights()
        return jsonify({'success': True, 'insights': insights})
    except Exception as e:
        logger.error(f"Error getting insights: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/knowledge/search')
def api_knowledge_search():
    """Search thoughts."""
    if not KNOWLEDGE_OS_AVAILABLE:
        return jsonify({'error': 'Knowledge OS not available'}), 503
    
    try:
        init_knowledge_os()
        from knowledge_os_interface import knowledge_os
        
        query = request.args.get('q', '').strip()
        
        if not query:
            return jsonify({'success': False, 'error': 'No search query provided'})
        
        results = knowledge_os.search_thoughts(query)
        
        search_results = []
        for thought in results[:10]:  # Limit to 10 results
            search_results.append({
                'id': thought.id,
                'content': thought.content,
                'timestamp': thought.timestamp.isoformat(),
                'importance': thought.importance,
                'tags': thought.tags,
                'concepts': thought.concepts
            })
        
        return jsonify({
            'success': True, 
            'results': search_results,
            'count': len(search_results)
        })
    except Exception as e:
        logger.error(f"Error searching thoughts: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/knowledge/densify')
def api_knowledge_densify_check():
    """Check for densification opportunities."""
    if not KNOWLEDGE_OS_AVAILABLE:
        return jsonify({'error': 'Knowledge OS not available'}), 503
    
    try:
        init_knowledge_os()
        from knowledge_os_interface import knowledge_os
        
        opportunities = knowledge_os.check_densification_opportunities()
        
        opportunities_data = []
        for opp in opportunities:
            opportunities_data.append({
                'concept': opp['concept'],
                'thought_count': len(opp['thoughts']),
                'analysis': opp['analysis'],
                'suggestion': opp['analysis']['suggestion']
            })
        
        return jsonify({
            'success': True, 
            'opportunities': opportunities_data,
            'count': len(opportunities_data)
        })
    except Exception as e:
        logger.error(f"Error checking densification: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/knowledge/densify/<concept>', methods=['POST'])
def api_knowledge_densify_concept(concept):
    """Densify thoughts for a specific concept."""
    if not KNOWLEDGE_OS_AVAILABLE:
        return jsonify({'error': 'Knowledge OS not available'}), 503
    
    try:
        init_knowledge_os()
        from knowledge_os_interface import knowledge_os
        
        cluster = knowledge_os.densify_concept(concept)
        
        if cluster:
            update_stats(0.5, 'knowledge', 'densification', 'knowledge_os')
            return jsonify({
                'success': True,
                'cluster': {
                    'id': cluster.id,
                    'name': cluster.name,
                    'summary': cluster.summary,
                    'key_insights': cluster.key_insights,
                    'compression_ratio': cluster.compression_ratio,
                    'original_word_count': cluster.original_word_count,
                    'compressed_word_count': cluster.compressed_word_count
                }
            })
        else:
            return jsonify({'success': False, 'error': 'Unable to densify concept'})
            
    except Exception as e:
        logger.error(f"Error densifying concept: {e}")
        return jsonify({'success': False, 'error': str(e)})


# ========== EXISTING ROUTES (Text, Multi-modal, etc.) ==========

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
                    mode='text',
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
            mode='text',
            content_type='text',
            success=True
        )
        
        return jsonify(hierarchical_result)
        
    except Exception as e:
        logger.error(f"Text processing error: {e}")
        update_stats(processing_time=time.time() - start_time, mode='text', success=False)
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
                mode='multimodal',
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
        update_stats(processing_time=time.time() - start_time, mode='multimodal', success=False)
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def get_stats():
    """Get detailed processing statistics."""
    session_id = get_session_id()
    
    # Add SumMail-specific stats
    summail_stats = {}
    if session_id in user_summail_engines:
        engine = user_summail_engines[session_id]
        summail_stats = {
            'emails_processed': len(engine.processed_emails),
            'software_tracked': len(engine.software_tracker),
            'average_compression': sum(
                e.compression_ratio for e in engine.processed_emails.values()
            ) / max(len(engine.processed_emails), 1) if engine.processed_emails else 0,
            'total_action_items': sum(
                len(e.metadata.action_items) for e in engine.processed_emails.values()
            )
        }
    
    return jsonify({
        'processing_stats': processing_stats,
        'cache_size': len(cache),
        'summail_stats': summail_stats,
        'system_capabilities': {
            'multimodal_available': MULTIMODAL_AVAILABLE,
            'ai_available': AI_AVAILABLE,
            'adaptive_available': ADAPTIVE_AVAILABLE,
            'summail_available': SUMMAIL_AVAILABLE,
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


# Create unified template
UNIFIED_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SUM Platform - Unified Knowledge Processing</title>
    <style>
        :root {
            --primary: #6366f1;
            --secondary: #8b5cf6;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --dark: #1f2937;
            --light: #f3f4f6;
            --white: #ffffff;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: var(--dark);
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        
        .header h1 {
            font-size: 2.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        .mode-selector {
            display: flex;
            gap: 15px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        
        .mode-btn {
            padding: 15px 25px;
            background: white;
            border: 2px solid #e5e7eb;
            border-radius: 15px;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 1rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .mode-btn:hover {
            border-color: var(--primary);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(99, 102, 241, 0.2);
        }
        
        .mode-btn.active {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }
        
        .mode-content {
            display: none;
            background: white;
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        
        .mode-content.active {
            display: block;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: var(--dark);
        }
        
        .form-group input, .form-group textarea {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e5e7eb;
            border-radius: 10px;
            font-size: 1rem;
            transition: all 0.3s;
        }
        
        .form-group input:focus, .form-group textarea:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn-primary {
            background: var(--primary);
            color: white;
        }
        
        .btn-primary:hover {
            background: #5558d9;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(99, 102, 241, 0.3);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        
        .stat-card {
            background: #f9fafb;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
        }
        
        .stat-card h3 {
            font-size: 1.5rem;
            color: var(--primary);
            margin-bottom: 5px;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #e5e7eb;
            border-radius: 4px;
            overflow: hidden;
            margin: 20px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            transition: width 0.3s ease;
        }
        
        .email-digest {
            background: #f9fafb;
            padding: 25px;
            border-radius: 15px;
            margin-top: 20px;
        }
        
        .category-badges {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin: 15px 0;
        }
        
        .badge {
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        
        .badge-newsletter {
            background: #e0e7ff;
            color: #4338ca;
        }
        
        .badge-software {
            background: #d1fae5;
            color: #047857;
        }
        
        .badge-financial {
            background: #fee2e2;
            color: #dc2626;
        }
        
        .action-item {
            background: #fef3c7;
            border-left: 4px solid var(--warning);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .fade-in {
            animation: fadeIn 0.5s ease;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header fade-in">
            <h1>🚀 SUM Platform</h1>
            <p>Unified Knowledge Processing - Text, Documents, Images, and Emails</p>
            
            <div class="mode-selector">
                <button class="mode-btn active" onclick="switchMode('text')">
                    📝 Text Processing
                </button>
                <button class="mode-btn" onclick="switchMode('file')">
                    📄 Multi-Modal Files
                </button>
                <button class="mode-btn" onclick="switchMode('email')">
                    📧 Email Processing
                </button>
                <button class="mode-btn" onclick="switchMode('stats')">
                    📊 Statistics
                </button>
            </div>
        </div>
        
        <!-- Text Processing Mode -->
        <div id="textMode" class="mode-content active fade-in">
            <h2>Hierarchical Text Processing</h2>
            <form id="textForm">
                <div class="form-group">
                    <label for="textInput">Enter Text</label>
                    <textarea id="textInput" rows="8" placeholder="Paste your text here for hierarchical processing..."></textarea>
                </div>
                <div class="form-group">
                    <label>
                        <input type="checkbox" id="useLocalAI"> Use Local AI Enhancement
                    </label>
                </div>
                <button type="submit" class="btn btn-primary">
                    Process Text
                </button>
            </form>
            <div id="textResults"></div>
        </div>
        
        <!-- Multi-Modal File Mode -->
        <div id="fileMode" class="mode-content fade-in">
            <h2>Multi-Modal File Processing</h2>
            <form id="fileForm">
                <div class="form-group">
                    <label for="fileInput">Select File</label>
                    <input type="file" id="fileInput" accept=".txt,.pdf,.docx,.png,.jpg,.jpeg,.html,.md">
                </div>
                <div class="form-group">
                    <label>
                        <input type="checkbox" id="useVision" checked> Use Vision Models (for images)
                    </label>
                </div>
                <button type="submit" class="btn btn-primary">
                    Process File
                </button>
            </form>
            <div id="fileResults"></div>
        </div>
        
        <!-- Email Processing Mode -->
        <div id="emailMode" class="mode-content fade-in">
            <h2>SumMail - Intelligent Email Processing</h2>
            
            <div id="emailLogin">
                <form id="emailForm">
                    <div class="form-group">
                        <label for="emailAddress">Email Address</label>
                        <input type="email" id="emailAddress" required placeholder="your@email.com">
                    </div>
                    <div class="form-group">
                        <label for="emailPassword">Password</label>
                        <input type="password" id="emailPassword" required>
                    </div>
                    <div class="form-group">
                        <label for="imapServer">IMAP Server (optional)</label>
                        <input type="text" id="imapServer" placeholder="Auto-detected">
                    </div>
                    <button type="submit" class="btn btn-primary">
                        Connect & Process
                    </button>
                </form>
            </div>
            
            <div id="emailDashboard" style="display: none;">
                <div class="progress-bar" id="emailProgress" style="display: none;">
                    <div class="progress-fill" id="emailProgressFill" style="width: 0%;"></div>
                </div>
                
                <div class="email-digest">
                    <h3>Email Digest</h3>
                    <div class="category-badges" id="categoryBadges"></div>
                    <div id="digestContent"></div>
                    <button class="btn btn-primary" onclick="generateDigest()">
                        Generate New Digest
                    </button>
                </div>
            </div>
        </div>
        
        <!-- Statistics Mode -->
        <div id="statsMode" class="mode-content fade-in">
            <h2>Processing Statistics</h2>
            <div class="stats-grid" id="statsGrid"></div>
        </div>
    </div>
    
    <script>
        let currentMode = 'text';
        let emailConnected = false;
        let eventSource;
        
        function switchMode(mode) {
            currentMode = mode;
            
            // Update buttons
            document.querySelectorAll('.mode-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
            
            // Update content
            document.querySelectorAll('.mode-content').forEach(content => {
                content.classList.remove('active');
            });
            document.getElementById(mode + 'Mode').classList.add('active');
            
            // Load stats if switching to stats mode
            if (mode === 'stats') {
                loadStats();
            }
        }
        
        // Text processing
        document.getElementById('textForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const text = document.getElementById('textInput').value;
            const useLocalAI = document.getElementById('useLocalAI').checked;
            
            if (!text) {
                alert('Please enter some text');
                return;
            }
            
            try {
                const response = await fetch('/api/process/text', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        text: text,
                        use_local_ai: useLocalAI,
                        config: {
                            max_concepts: 7,
                            max_summary_tokens: 150,
                            max_insights: 5
                        }
                    })
                });
                
                const result = await response.json();
                displayTextResults(result);
                
            } catch (error) {
                alert('Error: ' + error.message);
            }
        });
        
        // File processing
        document.getElementById('fileForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const fileInput = document.getElementById('fileInput');
            const useVision = document.getElementById('useVision').checked;
            
            if (!fileInput.files[0]) {
                alert('Please select a file');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('use_vision', useVision);
            formData.append('hierarchical_config', JSON.stringify({
                max_concepts: 10,
                max_summary_tokens: 200
            }));
            
            try {
                const response = await fetch('/api/process/file', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                displayFileResults(result);
                
            } catch (error) {
                alert('Error: ' + error.message);
            }
        });
        
        // Email processing
        document.getElementById('emailForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const email = document.getElementById('emailAddress').value;
            const password = document.getElementById('emailPassword').value;
            const imapServer = document.getElementById('imapServer').value;
            
            try {
                const response = await fetch('/api/summail/connect', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        email: email,
                        password: password,
                        imap_server: imapServer
                    })
                });
                
                if (response.ok) {
                    emailConnected = true;
                    document.getElementById('emailLogin').style.display = 'none';
                    document.getElementById('emailDashboard').style.display = 'block';
                    
                    // Start processing
                    processEmails();
                } else {
                    alert('Failed to connect to email');
                }
                
            } catch (error) {
                alert('Error: ' + error.message);
            }
        });
        
        async function processEmails() {
            const response = await fetch('/api/summail/process', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    folder: 'INBOX',
                    limit: 100
                })
            });
            
            if (response.ok) {
                monitorEmailProgress();
            }
        }
        
        async function monitorEmailProgress() {
            const progressBar = document.getElementById('emailProgress');
            const progressFill = document.getElementById('emailProgressFill');
            
            progressBar.style.display = 'block';
            
            const checkProgress = async () => {
                const response = await fetch('/api/summail/status');
                const status = await response.json();
                
                if (status.progress) {
                    progressFill.style.width = status.progress + '%';
                }
                
                if (status.status === 'completed') {
                    progressBar.style.display = 'none';
                    generateDigest();
                } else if (status.status === 'processing') {
                    setTimeout(checkProgress, 1000);
                }
            };
            
            checkProgress();
        }
        
        async function generateDigest() {
            const response = await fetch('/api/summail/digest', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({})
            });
            
            if (response.ok) {
                const digest = await response.json();
                displayEmailDigest(digest);
            }
        }
        
        function displayTextResults(result) {
            const resultsDiv = document.getElementById('textResults');
            
            let html = '<h3>Processing Results</h3>';
            
            if (result.hierarchical_summary) {
                const summary = result.hierarchical_summary;
                html += '<div class="stat-card">';
                html += '<h4>Key Concepts</h4>';
                html += '<p>' + (summary.level_1_concepts || []).join(', ') + '</p>';
                html += '</div>';
                
                html += '<div class="stat-card">';
                html += '<h4>Core Summary</h4>';
                html += '<p>' + (summary.level_2_core || '') + '</p>';
                html += '</div>';
            }
            
            if (result.local_ai_analysis) {
                html += '<div class="stat-card">';
                html += '<h4>AI Enhanced Summary</h4>';
                html += '<p>' + result.local_ai_analysis.summary + '</p>';
                html += '<small>Model: ' + result.local_ai_analysis.model_used + '</small>';
                html += '</div>';
            }
            
            resultsDiv.innerHTML = html;
        }
        
        function displayFileResults(result) {
            const resultsDiv = document.getElementById('fileResults');
            
            let html = '<h3>File Processing Results</h3>';
            html += '<div class="stat-card">';
            html += '<p><strong>File Type:</strong> ' + result.content_type + '</p>';
            html += '<p><strong>Confidence:</strong> ' + Math.round(result.confidence_score * 100) + '%</p>';
            html += '<p><strong>Processing Time:</strong> ' + result.processing_time.toFixed(2) + 's</p>';
            html += '</div>';
            
            if (result.metadata.hierarchical_analysis) {
                const analysis = result.metadata.hierarchical_analysis;
                html += '<div class="stat-card">';
                html += '<h4>Content Analysis</h4>';
                html += '<p>' + (analysis.hierarchical_summary?.level_2_core || 'No summary available') + '</p>';
                html += '</div>';
            }
            
            resultsDiv.innerHTML = html;
        }
        
        function displayEmailDigest(digest) {
            const badgesDiv = document.getElementById('categoryBadges');
            const contentDiv = document.getElementById('digestContent');
            
            // Update badges
            badgesDiv.innerHTML = '';
            for (const [category, count] of Object.entries(digest.categories_breakdown)) {
                const badge = document.createElement('span');
                badge.className = 'badge badge-' + category;
                badge.textContent = category + ': ' + count;
                badgesDiv.appendChild(badge);
            }
            
            // Update content
            let html = '';
            
            // Action items
            if (digest.action_items.length > 0) {
                html += '<h4>Action Items</h4>';
                digest.action_items.forEach(item => {
                    html += '<div class="action-item">';
                    html += '<strong>' + item.action + '</strong><br>';
                    html += '<small>From: ' + item.source + '</small>';
                    html += '</div>';
                });
            }
            
            // Software updates
            if (digest.software_updates.length > 0) {
                html += '<h4>Software Updates</h4>';
                digest.software_updates.forEach(update => {
                    html += '<div class="stat-card">';
                    html += '<strong>' + update.name + '</strong><br>';
                    html += 'Current: ' + update.current_version;
                    if (update.latest_version) {
                        html += ' → Latest: ' + update.latest_version;
                    }
                    html += '</div>';
                });
            }
            
            // Compression stats
            if (digest.compression_stats) {
                html += '<h4>Compression Statistics</h4>';
                html += '<div class="stat-card">';
                html += '<p>Average Compression: ' + Math.round(digest.compression_stats.average_compression * 100) + '%</p>';
                html += '<p>Space Saved: ' + Math.round(digest.compression_stats.total_compressed_size / 1024) + ' KB</p>';
                html += '</div>';
            }
            
            contentDiv.innerHTML = html;
        }
        
        async function loadStats() {
            const response = await fetch('/api/stats');
            const stats = await response.json();
            
            const gridDiv = document.getElementById('statsGrid');
            gridDiv.innerHTML = '';
            
            // Processing stats
            const processingCard = createStatCard('Total Requests', stats.processing_stats.total_requests);
            gridDiv.appendChild(processingCard);
            
            const successCard = createStatCard('Success Rate', 
                Math.round((stats.processing_stats.successful_requests / stats.processing_stats.total_requests) * 100) + '%');
            gridDiv.appendChild(successCard);
            
            // Mode usage
            for (const [mode, count] of Object.entries(stats.processing_stats.modes_used)) {
                const card = createStatCard(mode.charAt(0).toUpperCase() + mode.slice(1) + ' Mode', count);
                gridDiv.appendChild(card);
            }
            
            // Email stats if available
            if (stats.summail_stats.emails_processed) {
                const emailCard = createStatCard('Emails Processed', stats.summail_stats.emails_processed);
                gridDiv.appendChild(emailCard);
                
                const compressionCard = createStatCard('Email Compression', 
                    Math.round(stats.summail_stats.average_compression * 100) + '%');
                gridDiv.appendChild(compressionCard);
            }
        }
        
        function createStatCard(title, value) {
            const card = document.createElement('div');
            card.className = 'stat-card';
            card.innerHTML = '<h3>' + value + '</h3><p>' + title + '</p>';
            return card;
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    # Save template
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    with open(templates_dir / "index_unified.html", "w") as f:
        f.write(UNIFIED_TEMPLATE)
    
    # Initialize components on startup
    init_components()
    
    # Print startup information
    print("🚀 Unified SUM Platform Starting Up")
    print("=" * 50)
    print(f"✅ Text Processing: Hierarchical Engine")
    print(f"{'✅' if MULTIMODAL_AVAILABLE else '❌'} Multi-Modal Processing: PDF, DOCX, Images")
    print(f"{'✅' if SUMMAIL_AVAILABLE else '❌'} Email Processing: SumMail Engine")
    print(f"{'✅' if ollama_manager and ollama_manager.ollama_client else '❌'} Local AI: Ollama Integration")
    print(f"{'✅' if AI_AVAILABLE else '❌'} Cloud AI: OpenAI/Anthropic")
    print(f"{'✅' if ADAPTIVE_AVAILABLE else '❌'} Adaptive Compression")
    
    if MULTIMODAL_AVAILABLE and multimodal_processor:
        formats = multimodal_processor.get_supported_formats()
        print(f"\nSupported Formats: {', '.join(formats)}")
    
    if ollama_manager and ollama_manager.available_models:
        print(f"\nLocal Models: {len(ollama_manager.available_models)} available")
    
    print("=" * 50)
    print(f"🌐 Server starting on http://localhost:{os.getenv('FLASK_PORT', 5000)}")
    print("\nModes available:")
    print("  📝 Text Processing - Hierarchical summarization")
    print("  📄 Multi-Modal - Process any document or image")
    print("  📧 Email Mode - Intelligent email compression")
    print("  📊 Statistics - Real-time usage analytics")
    
    # Start the application
    port = int(os.getenv('FLASK_PORT', 5000))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=active_config.DEBUG,
        threaded=True
    )