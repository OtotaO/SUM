"""
API Authentication Routes

Endpoints for managing API keys and authentication.

Author: SUM Team
License: Apache License 2.0
"""

import logging
from flask import Blueprint, request, jsonify
from api.auth import get_auth_manager, require_api_key

logger = logging.getLogger(__name__)
auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/api/auth/keys', methods=['GET'])
@require_api_key(['admin'])
def list_api_keys():
    """List all API keys (admin only)."""
    try:
        auth_manager = get_auth_manager()
        keys = auth_manager.list_keys()
        
        return jsonify({
            'keys': keys,
            'total': len(keys)
        })
    except Exception as e:
        logger.error(f"Error listing API keys: {e}")
        return jsonify({'error': 'Failed to list keys'}), 500


@auth_bp.route('/api/auth/keys', methods=['POST'])
@require_api_key(['admin'])
def create_api_key():
    """Create a new API key (admin only)."""
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'name' not in data:
            return jsonify({'error': 'Name is required'}), 400
            
        name = data['name']
        permissions = data.get('permissions', ['read', 'summarize'])
        rate_limit = data.get('rate_limit')
        daily_limit = data.get('daily_limit')
        metadata = data.get('metadata', {})
        
        # Create key
        auth_manager = get_auth_manager()
        key_id, api_key = auth_manager.generate_api_key(
            name=name,
            permissions=permissions,
            rate_limit=rate_limit,
            daily_limit=daily_limit,
            metadata=metadata
        )
        
        return jsonify({
            'key_id': key_id,
            'api_key': api_key,
            'name': name,
            'permissions': permissions,
            'message': 'API key created successfully. Save the key securely - it cannot be retrieved again.'
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating API key: {e}")
        return jsonify({'error': 'Failed to create key'}), 500


@auth_bp.route('/api/auth/keys/<key_id>', methods=['DELETE'])
@require_api_key(['admin'])
def revoke_api_key(key_id):
    """Revoke an API key (admin only)."""
    try:
        auth_manager = get_auth_manager()
        auth_manager.revoke_key(key_id)
        
        return jsonify({
            'message': f'API key {key_id} revoked successfully'
        })
    except Exception as e:
        logger.error(f"Error revoking API key: {e}")
        return jsonify({'error': 'Failed to revoke key'}), 500


@auth_bp.route('/api/auth/keys/<key_id>/stats', methods=['GET'])
@require_api_key(['admin'])
def get_key_stats(key_id):
    """Get usage statistics for an API key (admin only)."""
    try:
        days = int(request.args.get('days', 7))
        
        auth_manager = get_auth_manager()
        stats = auth_manager.get_usage_stats(key_id, days)
        
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting key stats: {e}")
        return jsonify({'error': 'Failed to get stats'}), 500


@auth_bp.route('/api/auth/validate', methods=['GET'])
def validate_key():
    """Validate an API key and return its info."""
    try:
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        
        if not api_key:
            return jsonify({'error': 'No API key provided'}), 400
            
        auth_manager = get_auth_manager()
        key_info = auth_manager.validate_api_key(api_key)
        
        if not key_info:
            return jsonify({'valid': False}), 401
            
        # Don't return the actual key hash
        return jsonify({
            'valid': True,
            'key_id': key_info.key_id,
            'name': key_info.name,
            'permissions': key_info.permissions,
            'rate_limit': key_info.rate_limit,
            'daily_limit': key_info.daily_limit
        })
        
    except Exception as e:
        logger.error(f"Error validating API key: {e}")
        return jsonify({'error': 'Validation failed'}), 500


@auth_bp.route('/api/auth/usage', methods=['GET'])
@require_api_key()
def get_my_usage():
    """Get usage statistics for the current API key."""
    try:
        # Get current key info from request context
        key_info = request.api_key_info
        days = int(request.args.get('days', 7))
        
        auth_manager = get_auth_manager()
        stats = auth_manager.get_usage_stats(key_info.key_id, days)
        
        # Add current limits
        stats['limits'] = {
            'rate_limit': key_info.rate_limit,
            'daily_limit': key_info.daily_limit
        }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting usage stats: {e}")
        return jsonify({'error': 'Failed to get usage'}), 500