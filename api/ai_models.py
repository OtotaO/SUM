"""
ai_models.py - AI Model Integration API Endpoints

Clean AI model API following Carmack's principles:
- Fast model switching and processing
- Clear API contracts
- Efficient async handling
- Minimal external dependencies

Author: ototao
License: Apache License 2.0
"""

import logging
import asyncio
from flask import Blueprint, request, jsonify

from web.middleware import rate_limit, validate_json_input
from application.service_registry import registry


logger = logging.getLogger(__name__)
ai_models_bp = Blueprint('ai_models', __name__)


def _check_ai_availability():
    """Check if AI models are available."""
    ai_engine = registry.get_service('ai_engine')
    if not ai_engine:
        return jsonify({
            'error': 'AI models not available',
            'details': 'Install required packages: pip install openai anthropic cryptography'
        }), 503
    return ai_engine


@ai_models_bp.route('/models', methods=['GET'])
def get_ai_models():
    """Get available AI models."""
    ai_engine = _check_ai_availability()
    if isinstance(ai_engine, tuple):  # Error response
        return ai_engine
    
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


@ai_models_bp.route('/keys', methods=['GET'])
def get_api_keys():
    """Get saved API keys (masked for security)."""
    ai_engine = _check_ai_availability()
    if isinstance(ai_engine, tuple):
        return ai_engine
    
    try:
        key_manager = ai_engine.key_manager
        keys = key_manager.load_api_keys()
        # Mask keys for security
        masked_keys = {
            provider: '••••••••' if key else '' 
            for provider, key in keys.items()
        }
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


@ai_models_bp.route('/keys', methods=['POST'])
@rate_limit(5, 60)
@validate_json_input()
def save_api_key():
    """Save API key for a provider."""
    ai_engine = _check_ai_availability()
    if isinstance(ai_engine, tuple):
        return ai_engine
    
    try:
        data = request.get_json()
        
        # Validate input
        if 'provider' not in data or 'api_key' not in data:
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
        
        # Clear model cache
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


@ai_models_bp.route('/process', methods=['POST'])
@rate_limit(20, 60)
@validate_json_input()
def process_with_ai():
    """Process text using AI models."""
    ai_engine = _check_ai_availability()
    if isinstance(ai_engine, tuple):
        return ai_engine
    
    try:
        data = request.get_json()
        
        # Validate input
        if 'text' not in data:
            return jsonify({
                'error': 'Missing text in request'
            }), 400
        
        text = data['text']
        model_id = data.get('model_id', 'traditional')
        config = data.get('config', {})
        
        # Validate text length
        if len(text) > 50000:
            return jsonify({
                'error': 'Text too long (max 50,000 characters)'
            }), 400
        
        # Process text asynchronously
        result = _run_async(ai_engine.process_text(text, model_id, config))
        
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


@ai_models_bp.route('/compare', methods=['POST'])
@rate_limit(10, 60)
@validate_json_input()
def compare_ai_models():
    """Compare outputs from multiple AI models."""
    ai_engine = _check_ai_availability()
    if isinstance(ai_engine, tuple):
        return ai_engine
    
    try:
        data = request.get_json()
        
        # Validate input
        if 'text' not in data or 'model_ids' not in data:
            return jsonify({
                'error': 'Missing text or model_ids in request'
            }), 400
        
        text = data['text']
        model_ids = data['model_ids']
        config = data.get('config', {})
        
        # Validate inputs
        if len(text) > 30000:
            return jsonify({
                'error': 'Text too long for comparison (max 30,000 characters)'
            }), 400
        
        if len(model_ids) > 4:
            return jsonify({
                'error': 'Too many models for comparison (max 4)'
            }), 400
        
        # Compare models asynchronously
        result = _run_async(ai_engine.compare_models(text, model_ids, config))
        
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


@ai_models_bp.route('/estimate_cost', methods=['POST'])
@validate_json_input()
def estimate_cost():
    """Estimate cost for processing text with a model."""
    ai_engine = _check_ai_availability()
    if isinstance(ai_engine, tuple):
        return ai_engine
    
    try:
        data = request.get_json()
        
        # Validate input
        if 'text' not in data or 'model_id' not in data:
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


def _run_async(coro):
    """Run async coroutine in sync context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()