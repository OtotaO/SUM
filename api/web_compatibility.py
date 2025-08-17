"""
Web compatibility routes to bridge frontend and backend

This module provides the endpoints that the web interface expects.
"""

from flask import Blueprint, request, jsonify
from api.summarization import _process_with_model
from utils.universal_file_processor import UniversalFileProcessor
import tempfile
import os

web_compat_bp = Blueprint('web_compat', __name__)


@web_compat_bp.route('/summarize', methods=['POST'])
def summarize_simple():
    """Simple summarization endpoint for web UI compatibility."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        density = data.get('density', 'medium')
        
        # Map density to appropriate config
        config = {
            'maxTokens': {
                'minimal': 50,
                'short': 100,
                'medium': 200,
                'detailed': 500
            }.get(density, 200)
        }
        
        # Process using the simple model
        result = _process_with_model(text, 'simple', config)
        
        if 'error' in result:
            return jsonify(result), result.get('status_code', 500)
        
        # Format response for web UI
        return jsonify({
            'summary': result.get('summary', ''),
            'original_words': len(text.split()),
            'compression_ratio': len(text) / max(1, len(result.get('summary', ''))) if result.get('summary') else 1.0,
            'model': 'simple'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@web_compat_bp.route('/summarize/ultimate', methods=['POST'])
def summarize_ultimate():
    """Ultimate summarization endpoint with multi-density output."""
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json()
            text = data.get('text', '')
            density = data.get('density', 'all')
        else:
            # File upload
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            density = request.form.get('density', 'all')
            
            # Process file
            tmp_path = None
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
                    file.save(tmp_file.name)
                    tmp_path = tmp_file.name
                
                # Extract text
                processor = UniversalFileProcessor()
                result = processor.process_file(tmp_path)
                
                if not result['success']:
                    return jsonify({'error': result.get('error', 'Could not extract text from file')}), 400
                
                text = result['text']
                if not text:
                    return jsonify({'error': 'No text extracted from file'}), 400
                    
            except Exception as e:
                return jsonify({'error': f'File processing failed: {str(e)}'}), 500
            finally:
                # Always clean up temporary file
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
        
        if not text:
            return jsonify({'error': 'No text to summarize'}), 400
        
        # Check text size for unlimited processing
        text_size = len(text.encode('utf-8'))
        if text_size > 100 * 1024:  # > 100KB
            # Use unlimited processor
            from unlimited_text_processor import process_unlimited_text
            result = process_unlimited_text(text, {
                'max_summary_tokens': 500,
                'enable_semantic_clustering': True
            })
        else:
            # Process with hierarchical engine
            config = {
                'max_concepts': 10,
                'max_summary_tokens': 500,
                'enable_semantic_clustering': True
            }
            
            result = _process_with_model(text, 'hierarchical', config)
        
        if 'error' in result:
            return jsonify(result), result.get('status_code', 500)
        
        # Format multi-density response
        response = {
            'result': {
                'original_words': len(text.split()),
                'compression_ratio': len(text) / max(1, len(result.get('summary', ''))) if result.get('summary') else 1.0,
            }
        }
        
        # Extract hierarchical summaries if available
        if 'hierarchical_summary' in result:
            hs = result['hierarchical_summary']
            response['result'].update({
                'tags': result.get('tags', result.get('key_concepts', []))[:10],
                'minimal': ', '.join(hs.get('level_1_concepts', [])) if hs.get('level_1_concepts') else '',
                'short': hs.get('level_2_core', ''),
                'medium': hs.get('level_3_expanded', '') or hs.get('level_2_core', ''),
                'detailed': result.get('summary', hs.get('level_3_expanded', '') or hs.get('level_2_core', ''))
            })
        else:
            # Fallback to simple summary
            summary = result.get('summary', '')
            response['result'].update({
                'tags': result.get('tags', []),
                'minimal': summary[:100] + '...' if len(summary) > 100 else summary,
                'short': summary[:200] + '...' if len(summary) > 200 else summary,
                'medium': summary[:500] + '...' if len(summary) > 500 else summary,
                'detailed': summary
            })
        
        # Add cached flag if requested by frontend
        response['cached'] = False
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': 'Processing failed',
            'details': str(e)
        }), 500


@web_compat_bp.route('/api/stream/summarize', methods=['GET', 'POST'])
def stream_summarize():
    """Streaming summarization endpoint (placeholder)."""
    # For now, return a non-streaming response
    # TODO: Implement true streaming with Server-Sent Events
    
    if request.method == 'GET':
        text = request.args.get('text', '')
    else:
        data = request.get_json()
        text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Process normally and return as complete
    config = {'maxTokens': 200}
    result = _process_with_model(text, 'simple', config)
    
    return jsonify({
        'type': 'complete',
        'result': {
            'summary': result.get('summary', ''),
            'original_words': len(text.split())
        }
    })