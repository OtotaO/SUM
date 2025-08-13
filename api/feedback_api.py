"""
feedback_api.py - API Endpoints for Feedback System

Simple, practical feedback API that:
- Collects user ratings
- Returns personalized recommendations
- Provides usage insights

Author: SUM Development Team
License: Apache License 2.0
"""

import logging
from flask import Blueprint, request, jsonify
from typing import Dict, Any

from web.middleware import rate_limit, validate_json_input
from application.feedback_system import get_feedback_system, apply_feedback_preferences

logger = logging.getLogger(__name__)
feedback_bp = Blueprint('feedback', __name__)


@feedback_bp.route('/feedback/submit', methods=['POST'])
@rate_limit(100, 60)  # Allow many feedback submissions
@validate_json_input()
def submit_feedback():
    """
    Submit feedback for a summary.
    
    Expected JSON:
    {
        "content_hash": "hash_of_original_content",
        "summary_type": "medium",
        "rating": 4,
        "helpful": true,
        "comment": "Good summary but could be shorter"
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        if 'content_hash' not in data or 'rating' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Get feedback system
        feedback_system = get_feedback_system()
        
        # Add feedback
        feedback_id = feedback_system.add_feedback(
            content_hash=data['content_hash'],
            summary_type=data.get('summary_type', 'unknown'),
            rating=data['rating'],
            helpful=data.get('helpful'),
            comment=data.get('comment'),
            tags=data.get('tags', [])
        )
        
        # Return updated preferences
        preferences = feedback_system.get_preferences()
        
        return jsonify({
            'success': True,
            'feedback_id': feedback_id,
            'updated_preferences': preferences,
            'message': 'Thank you for your feedback!'
        })
        
    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        return jsonify({'error': str(e)}), 500


@feedback_bp.route('/feedback/preferences', methods=['GET'])
def get_preferences():
    """Get current learned preferences."""
    try:
        feedback_system = get_feedback_system()
        preferences = feedback_system.get_preferences()
        stats = feedback_system.get_feedback_stats()
        
        return jsonify({
            'preferences': preferences,
            'stats': stats,
            'confidence': feedback_system._calculate_confidence()
        })
        
    except Exception as e:
        logger.error(f"Error getting preferences: {e}")
        return jsonify({'error': str(e)}), 500


@feedback_bp.route('/feedback/recommendations', methods=['POST'])
@validate_json_input()
def get_recommendations():
    """
    Get recommended settings for summarization.
    
    Optional JSON:
    {
        "content_type": "technical" | "news" | "academic"
    }
    """
    try:
        data = request.get_json() or {}
        
        feedback_system = get_feedback_system()
        recommendations = feedback_system.get_recommended_settings(
            content_type=data.get('content_type')
        )
        
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return jsonify({'error': str(e)}), 500


@feedback_bp.route('/feedback/insights', methods=['GET'])
@rate_limit(20, 60)
def get_insights():
    """Get insights from collected feedback."""
    try:
        feedback_system = get_feedback_system()
        insights = feedback_system.export_insights()
        
        return jsonify({
            'success': True,
            'insights': insights
        })
        
    except Exception as e:
        logger.error(f"Error getting insights: {e}")
        return jsonify({'error': str(e)}), 500


@feedback_bp.route('/feedback/rate-summary', methods=['POST'])
@rate_limit(100, 60)
@validate_json_input()
def quick_rate():
    """
    Quick rating endpoint for immediate feedback.
    
    Expected JSON:
    {
        "summary_id": "unique_summary_id",
        "rating": 1-5,
        "summary_type": "medium"
    }
    """
    try:
        data = request.get_json()
        
        if 'summary_id' not in data or 'rating' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Use summary_id as content_hash for simplicity
        feedback_system = get_feedback_system()
        
        feedback_id = feedback_system.add_feedback(
            content_hash=data['summary_id'],
            summary_type=data.get('summary_type', 'unknown'),
            rating=data['rating'],
            helpful=data['rating'] >= 4  # Assume 4+ stars means helpful
        )
        
        return jsonify({
            'success': True,
            'feedback_id': feedback_id,
            'message': 'Rating recorded'
        })
        
    except Exception as e:
        logger.error(f"Quick rate error: {e}")
        return jsonify({'error': str(e)}), 500


# Export blueprint
__all__ = ['feedback_bp']