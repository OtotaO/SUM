"""
file_processing.py - File Processing API Endpoints

Clean file processing API following Carmack's principles:
- Secure file handling with automatic cleanup
- Fast processing with minimal memory usage
- Clear error handling and validation
- Single responsibility per endpoint

Author: ototao
License: Apache License 2.0
"""

import os
import tempfile
import logging
import traceback
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from web.middleware import rate_limit, allowed_file
from config import active_config
from summarization_engine import (
    SummarizationEngine, 
    BasicSummarizationEngine, 
    AdvancedSummarizationEngine, 
    HierarchicalDensificationEngine
)
from multimodal_processor import MultiModalProcessor

# Try to import ChatIntelligenceEngine, but don't fail if missing
try:
    from chat_intelligence_engine import ChatIntelligenceEngine
except ImportError:
    ChatIntelligenceEngine = None

# Import OnePunchBridge
try:
    from onepunch_bridge import OnePunchBridge
    onepunch_bridge = OnePunchBridge()
except ImportError:
    onepunch_bridge = None

logger = logging.getLogger(__name__)
file_processing_bp = Blueprint('file_processing', __name__)


@file_processing_bp.route('/process/file', methods=['POST'])
@file_processing_bp.route('/analyze_file', methods=['POST'])
@rate_limit(5, 300)  # 5 calls per 5 minutes
def analyze_file():
    """
    Analyze file (Text, PDF, Image) and generate summary.
    Optionally generate OnePunch content if 'generate_social' is true.
    """
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save file securely
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # 1. Extract content using MultiModalProcessor
            processor = MultiModalProcessor()
            processed_data = processor.process_file(filepath)
            
            if 'error' in processed_data:
                return jsonify({'error': processed_data['error']}), 400
            
            text_content = processed_data.get('content', '')
            if not text_content:
                return jsonify({'error': 'No text extracted from file'}), 400

            # 2. Process with SummarizationEngine
            model_type = request.form.get('model', 'hierarchical').lower()
            max_tokens = int(request.form.get('maxTokens', str(active_config.MAX_SUMMARY_LENGTH)))
            
            # Select appropriate engine
            if model_type == 'hierarchical':
                engine = HierarchicalDensificationEngine()
            elif model_type == 'advanced':
                engine = AdvancedSummarizationEngine()
            else:
                engine = BasicSummarizationEngine()
            
            # Use appropriate method based on request or default to process_text which handles logic
            result = engine.process_text(text_content, {'maxTokens': max_tokens})
            
            # 3. Optional: OnePunch Integration
            generate_social = request.form.get('generate_social', 'false').lower() == 'true'
            social_content = None
            
            if generate_social and onepunch_bridge:
                try:
                    title = filename.rsplit('.', 1)[0].replace('_', ' ').title()
                    social_content = onepunch_bridge.process_content(text_content, title=title)
                except Exception as e:
                    logger.warning(f"OnePunch bridge failed: {e}")
                    social_content = {"error": "Bridge generation failed", "details": str(e)}

            # Format output
            response = {
                'filename': filename,
                'file_type': processed_data.get('type', 'unknown'),
                'summary': result.get('summary', ''),
                'hierarchical_summary': result.get('hierarchical_summary', {}),
                'topics': result.get('tags', []),
                'original_length': len(text_content),
                'pages': processed_data.get('pages', 1)
            }
            
            if social_content:
                response['social_content'] = social_content
            
            return jsonify(response)
            
        finally:
            # Clean up - delete uploaded file
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                logger.warning(f"Failed to delete uploaded file {filepath}: {e}")
    
    except Exception as e:
        logger.error(f"Error analyzing file: {e}\n{traceback.format_exc()}")
        return jsonify({
            'error': 'Error analyzing file',
            'details': str(e)
        }), 500


@file_processing_bp.route('/chat_export/process', methods=['POST'])
@rate_limit(10, 60)
def process_chat_export():
    """
    Process chat export files.
    """
    if ChatIntelligenceEngine is None:
         return jsonify({'error': 'ChatIntelligenceEngine not available'}), 501

    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w+b', suffix='.json', delete=False) as tmp:
            file.save(tmp.name)
            temp_path = tmp.name
        
        try:
            # Process the chat export
            engine = ChatIntelligenceEngine()
            result = engine.process_chat_export(temp_path)
            
            return jsonify({
                'status': 'success',
                'filename': file.filename,
                'processing_result': result
            })
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_path}: {e}")
    
    except Exception as e:
        logger.error(f"Error processing chat export: {e}")
        return jsonify({
            'error': 'Chat export processing failed',
            'details': str(e)
        }), 500


@file_processing_bp.route('/knowledge_graph', methods=['POST'])
@rate_limit(5, 300)
def generate_knowledge_graph():
    """
    Generate a knowledge graph from input text.
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        if 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        max_nodes = int(data.get('max_nodes', 20))
        min_weight = float(data.get('min_weight', 0.1))
        
        # Use SummarizationEngine for extraction logic
        # Default to Basic engine for knowledge graph if not specified
        engine = BasicSummarizationEngine() 
        result = engine.process_text(text)
        
        # In a real implementation, we would extract entities and relations.
        # For now, we use the tags as nodes and co-occurrence as edges.
        tags = result.get('tags', [])
        
        nodes = []
        edges = []
        
        for i, tag in enumerate(tags[:max_nodes]):
            nodes.append({
                'id': i,
                'label': tag,
                'type': 'concept'
            })
        
        # Simple co-occurrence mock for robustness
        import itertools
        for i, j in itertools.combinations(range(len(nodes)), 2):
            edges.append({
                'source': i,
                'target': j,
                'weight': 0.5 # Default weight
            })
            
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
