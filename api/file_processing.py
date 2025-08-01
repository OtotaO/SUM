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


logger = logging.getLogger(__name__)
file_processing_bp = Blueprint('file_processing', __name__)


@file_processing_bp.route('/analyze_file', methods=['POST'])
@rate_limit(5, 300)  # 5 calls per 5 minutes
def analyze_file():
    """
    Analyze text file and generate summary and topic information.
    
    Expected form data:
    - file: File upload
    - model: "simple|advanced" (Optional)
    - num_topics: Number of topics (Optional)
    """
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save file securely
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Get configuration from form data
            model_type = request.form.get('model', 'simple').lower()
            include_topics = request.form.get('include_topics', 'true').lower() == 'true'
            include_analysis = request.form.get('include_analysis', 'false').lower() == 'true'
            
            config = {
                'maxTokens': int(request.form.get('maxTokens', str(active_config.MAX_SUMMARY_LENGTH))),
                'include_analysis': include_analysis
            }
            
            # Process file with summarizer
            from Models.summarizer import Summarizer
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


@file_processing_bp.route('/chat_export/process', methods=['POST'])
@rate_limit(10, 60)
def process_chat_export():
    """
    Process chat export files to extract training insights.
    
    Expected form data:
    - file: Chat export file (JSON)
    - extract_training: bool (optional)
    """
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
            from chat_intelligence_engine import ChatIntelligenceEngine
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
    
    Expected JSON input:
    {
        "text": "Text to analyze...",
        "max_nodes": 20,
        "min_weight": 0.1
    }
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        # Validate required fields
        if 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        max_nodes = int(data.get('max_nodes', 20))
        min_weight = float(data.get('min_weight', 0.1))
        
        # Get advanced summarizer for entity extraction
        from application.service_registry import registry
        advanced_summarizer = registry.get_service('advanced_summarizer')
        if not advanced_summarizer:
            return jsonify({
                'error': 'Advanced summarizer required for knowledge graph generation'
            }), 500
        
        # Extract entities and relationships
        entities = advanced_summarizer.identify_entities(text)
        
        # Create knowledge graph
        nodes = []
        edges = []
        
        # Convert entities to nodes
        for i, (entity, entity_type) in enumerate(entities[:max_nodes]):
            nodes.append({
                'id': i,
                'label': entity,
                'type': entity_type
            })
        
        # Create edges between co-occurring entities
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
                    
                    # Check if edge exists, increment weight
                    edge_exists = False
                    for edge in edges:
                        if ((edge['source'] == source and edge['target'] == target) or
                           (edge['source'] == target and edge['target'] == source)):
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
        if edges:
            max_weight = max(edge['weight'] for edge in edges)
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