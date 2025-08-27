"""
Advanced Crystallization API - The Future of Summarization
"""

from flask import Blueprint, request, jsonify, stream_with_context, Response
from typing import Dict, Any, Optional
import json
import time
from functools import wraps
from knowledge_crystallizer import (
    KnowledgeCrystallizer, 
    CrystallizationConfig,
    DensityLevel,
    StylePersona
)
import logging

logger = logging.getLogger(__name__)

crystallization_bp = Blueprint('crystallization', __name__)
crystallizer = KnowledgeCrystallizer()

# User preference storage (in production, use database)
user_preferences = {}


def track_usage(f):
    """Decorator to track API usage for learning"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        
        # Track usage metrics
        duration = time.time() - start_time
        
        # Log for analytics
        logger.info(f"API call: {f.__name__}, duration: {duration:.2f}s")
        
        return result
    return decorated_function


@crystallization_bp.route('/api/crystallize', methods=['POST'])
@track_usage
def crystallize_text():
    """
    The ultimate summarization endpoint with full control
    
    Request body:
    {
        "text": "Your text here",
        "density": "standard",  # essence|tweet|elevator|executive|brief|standard|detailed|comprehensive
        "style": "neutral",     # hemingway|academic|storyteller|analyst|poet|executive|teacher|journalist|developer|neutral
        "options": {
            "preserve_entities": true,
            "preserve_numbers": true,
            "preserve_quotes": false,
            "interactive": false,
            "progressive": false
        },
        "user_id": "optional_user_id"  # For preference learning
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        # Parse configuration
        config = CrystallizationConfig()
        
        # Set density level
        if 'density' in data:
            try:
                config.density = DensityLevel[data['density'].upper()]
            except KeyError:
                return jsonify({'error': f"Invalid density level: {data['density']}"}), 400
        
        # Set style persona
        if 'style' in data:
            try:
                config.style = StylePersona[data['style'].upper()]
            except KeyError:
                return jsonify({'error': f"Invalid style: {data['style']}"}), 400
        
        # Set options
        if 'options' in data:
            options = data['options']
            config.preserve_entities = options.get('preserve_entities', True)
            config.preserve_numbers = options.get('preserve_numbers', True)
            config.preserve_quotes = options.get('preserve_quotes', False)
            config.interactive = options.get('interactive', False)
            config.progressive = options.get('progressive', False)
        
        # Set user preferences for learning
        if 'user_id' in data:
            config.user_preferences = {
                'user_id': data['user_id'],
                'session_id': request.headers.get('X-Session-ID'),
                'timestamp': time.time()
            }
        
        # Crystallize the knowledge
        result = crystallizer.crystallize(text, config)
        
        # Prepare response
        response = {
            'essence': result.essence,
            'summary': result.levels.get(config.density.name.lower(), ''),
            'all_levels': result.levels,
            'metadata': result.metadata,
            'quality_score': result.quality_score,
            'interactive': result.interactive_elements if config.interactive else None
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Crystallization error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@crystallization_bp.route('/api/crystallize/progressive', methods=['POST'])
@track_usage
def progressive_crystallization():
    """
    Stream progressive levels of crystallization
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        def generate():
            """Generate progressive summaries"""
            config = CrystallizationConfig()
            
            # Stream each density level
            for density in DensityLevel:
                config.density = density
                
                # Crystallize at this density
                result = crystallizer.crystallize(text, config)
                
                # Stream the result
                data = {
                    'level': density.name.lower(),
                    'density': density.value,
                    'summary': result.levels.get(density.name.lower(), ''),
                    'quality_score': result.quality_score
                }
                
                yield f"data: {json.dumps(data)}\n\n"
                
                # Small delay for demonstration
                time.sleep(0.1)
            
            # Send completion signal
            yield f"data: {json.dumps({'complete': True})}\n\n"
        
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )
        
    except Exception as e:
        logger.error(f"Progressive crystallization error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@crystallization_bp.route('/api/crystallize/adaptive', methods=['POST'])
@track_usage
def adaptive_crystallization():
    """
    Automatically adapt to user preferences and content type
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        user_id = data.get('user_id')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Create adaptive configuration
        config = CrystallizationConfig()
        
        # Let the system decide the best settings
        if user_id:
            config.user_preferences = {'user_id': user_id}
        
        # Auto-detect optimal density based on text length
        text_length = len(text)
        if text_length < 500:
            config.density = DensityLevel.STANDARD
        elif text_length < 2000:
            config.density = DensityLevel.BRIEF
        elif text_length < 10000:
            config.density = DensityLevel.EXECUTIVE
        else:
            config.density = DensityLevel.ESSENCE
        
        # Auto-detect optimal style based on content
        if any(term in text.lower() for term in ['research', 'study', 'methodology']):
            config.style = StylePersona.ACADEMIC
        elif any(term in text.lower() for term in ['meeting', 'action', 'decision']):
            config.style = StylePersona.EXECUTIVE
        elif any(term in text.lower() for term in ['api', 'function', 'code']):
            config.style = StylePersona.DEVELOPER
        else:
            config.style = StylePersona.NEUTRAL
        
        # Crystallize with adaptive settings
        result = crystallizer.crystallize(text, config)
        
        response = {
            'summary': result.levels.get(config.density.name.lower(), ''),
            'essence': result.essence,
            'adapted_settings': {
                'density': config.density.name.lower(),
                'style': config.style.value,
                'reason': 'Auto-adapted based on content analysis'
            },
            'metadata': result.metadata,
            'quality_score': result.quality_score
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Adaptive crystallization error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@crystallization_bp.route('/api/crystallize/feedback', methods=['POST'])
def submit_feedback():
    """
    Submit feedback to improve crystallization
    """
    try:
        data = request.get_json()
        
        user_id = data.get('user_id')
        text_type = data.get('text_type', 'general')
        rating = data.get('rating', 0)
        density = data.get('density')
        style = data.get('style')
        
        if not user_id:
            return jsonify({'error': 'User ID required for feedback'}), 400
        
        # Create config from feedback
        config = CrystallizationConfig()
        if density:
            config.density = DensityLevel[density.upper()]
        if style:
            config.style = StylePersona[style.upper()]
        
        # Learn from feedback
        crystallizer.preference_learner.learn_from_feedback(
            user_id, text_type, config, rating
        )
        
        return jsonify({
            'status': 'success',
            'message': 'Feedback recorded and preferences updated'
        })
        
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@crystallization_bp.route('/api/crystallize/styles', methods=['GET'])
def get_available_styles():
    """
    Get all available summarization styles with descriptions
    """
    styles = {
        'hemingway': {
            'name': 'Hemingway',
            'description': 'Terse. Direct. No fluff.',
            'best_for': 'Quick facts and decisive communication'
        },
        'academic': {
            'name': 'Academic',
            'description': 'Rigorous, cited, methodical',
            'best_for': 'Research papers and scholarly content'
        },
        'storyteller': {
            'name': 'Storyteller',
            'description': 'Narrative flow, engaging',
            'best_for': 'Case studies and experiential content'
        },
        'analyst': {
            'name': 'Analyst',
            'description': 'Data-driven, quantitative',
            'best_for': 'Reports with metrics and numbers'
        },
        'poet': {
            'name': 'Poet',
            'description': 'Metaphorical, evocative',
            'best_for': 'Creative and inspirational content'
        },
        'executive': {
            'name': 'Executive',
            'description': 'Action-oriented, strategic',
            'best_for': 'Business decisions and briefings'
        },
        'teacher': {
            'name': 'Teacher',
            'description': 'Educational, scaffolded',
            'best_for': 'Learning materials and tutorials'
        },
        'journalist': {
            'name': 'Journalist',
            'description': 'Who, what, when, where, why',
            'best_for': 'News and current events'
        },
        'developer': {
            'name': 'Developer',
            'description': 'Technical, precise, code-aware',
            'best_for': 'Technical documentation and code'
        },
        'neutral': {
            'name': 'Neutral',
            'description': 'Balanced, objective',
            'best_for': 'General purpose summarization'
        }
    }
    
    return jsonify(styles)


@crystallization_bp.route('/api/crystallize/densities', methods=['GET'])
def get_density_levels():
    """
    Get all available density levels with descriptions
    """
    densities = {
        'essence': {
            'value': 0.01,
            'description': 'Single most important insight',
            'typical_length': '1 sentence'
        },
        'tweet': {
            'value': 0.02,
            'description': '280 characters worth',
            'typical_length': '1-2 sentences'
        },
        'elevator': {
            'value': 0.05,
            'description': '30-second pitch',
            'typical_length': '3-4 sentences'
        },
        'executive': {
            'value': 0.10,
            'description': 'C-suite briefing',
            'typical_length': '1 paragraph'
        },
        'brief': {
            'value': 0.20,
            'description': 'Quick read',
            'typical_length': '2-3 paragraphs'
        },
        'standard': {
            'value': 0.30,
            'description': 'Balanced summary',
            'typical_length': '3-5 paragraphs'
        },
        'detailed': {
            'value': 0.50,
            'description': 'Thorough coverage',
            'typical_length': 'Half page'
        },
        'comprehensive': {
            'value': 0.70,
            'description': 'Near-complete retention',
            'typical_length': 'Full page'
        }
    }
    
    return jsonify(densities)


@crystallization_bp.route('/api/crystallize/compare', methods=['POST'])
@track_usage
def compare_styles():
    """
    Compare the same text in different styles side by side
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        styles = data.get('styles', ['neutral', 'hemingway', 'executive'])
        density = data.get('density', 'brief')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Limit to 5 styles for performance
        styles = styles[:5]
        
        comparisons = {}
        
        for style_name in styles:
            try:
                config = CrystallizationConfig()
                config.density = DensityLevel[density.upper()]
                config.style = StylePersona[style_name.upper()]
                
                result = crystallizer.crystallize(text, config)
                
                comparisons[style_name] = {
                    'summary': result.levels.get(config.density.name.lower(), ''),
                    'essence': result.essence,
                    'quality_score': result.quality_score
                }
            except KeyError:
                comparisons[style_name] = {'error': f'Invalid style: {style_name}'}
        
        return jsonify({
            'comparisons': comparisons,
            'density_used': density
        })
        
    except Exception as e:
        logger.error(f"Comparison error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@crystallization_bp.route('/api/crystallize/batch', methods=['POST'])
@track_usage
def batch_crystallize():
    """
    Process multiple texts with the same settings
    """
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        density = data.get('density', 'standard')
        style = data.get('style', 'neutral')
        
        if not texts:
            return jsonify({'error': 'No texts provided'}), 400
        
        # Limit batch size
        texts = texts[:100]
        
        config = CrystallizationConfig()
        config.density = DensityLevel[density.upper()]
        config.style = StylePersona[style.upper()]
        
        results = []
        
        for text in texts:
            if isinstance(text, dict):
                text_content = text.get('content', '')
                text_id = text.get('id', None)
            else:
                text_content = text
                text_id = None
            
            result = crystallizer.crystallize(text_content, config)
            
            results.append({
                'id': text_id,
                'summary': result.levels.get(config.density.name.lower(), ''),
                'essence': result.essence,
                'quality_score': result.quality_score
            })
        
        return jsonify({
            'results': results,
            'total_processed': len(results),
            'settings': {
                'density': density,
                'style': style
            }
        })
        
    except Exception as e:
        logger.error(f"Batch crystallization error: {str(e)}")
        return jsonify({'error': str(e)}), 500