#!/usr/bin/env python3
"""
Collaborative Intelligence API Endpoints
=======================================

Real-time collaborative intelligence API with WebSocket support for
live co-thinking sessions and shared knowledge spaces.

Features:
- RESTful API for collaborative intelligence operations
- WebSocket support for real-time updates
- Live collaboration events and notifications
- Secure multi-user session management

Author: SUM Revolutionary Team
License: Apache License 2.0
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Blueprint, request, jsonify, session
from flask_socketio import SocketIO, emit, join_room, leave_room, rooms
import uuid

# Import collaborative intelligence engine
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collaborative_intelligence_engine import (
    CollaborativeIntelligenceEngine, 
    CollaborationPermission,
    SessionState
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global collaborative intelligence engine
collab_engine = CollaborativeIntelligenceEngine()

# Create Blueprint for REST API
collab_api = Blueprint('collaborative_intelligence', __name__, url_prefix='/api/collaborative')

# WebSocket events will be handled by the main app


@collab_api.route('/clusters', methods=['POST'])
def create_cluster():
    """Create a new knowledge cluster."""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'description', 'creator_id', 'creator_name']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Create cluster
        cluster = collab_engine.create_knowledge_cluster(
            name=data['name'],
            description=data['description'],
            creator_id=data['creator_id'],
            creator_name=data['creator_name'],
            privacy_level=data.get('privacy_level', 'team')
        )
        
        return jsonify({
            'success': True,
            'cluster_id': cluster.id,
            'cluster': cluster.to_dict()
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating cluster: {e}")
        return jsonify({'error': 'Failed to create cluster'}), 500


@collab_api.route('/clusters/<cluster_id>/join', methods=['POST'])
def join_cluster(cluster_id: str):
    """Join an existing knowledge cluster."""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['user_id', 'user_name']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Parse permission level
        permission_str = data.get('permission', 'contribute')
        try:
            permission = CollaborationPermission(permission_str)
        except ValueError:
            return jsonify({'error': f'Invalid permission level: {permission_str}'}), 400
        
        # Join cluster
        success = collab_engine.join_knowledge_cluster(
            cluster_id=cluster_id,
            user_id=data['user_id'],
            user_name=data['user_name'],
            permission=permission
        )
        
        if not success:
            return jsonify({'error': 'Failed to join cluster. Cluster may not exist.'}), 404
        
        return jsonify({
            'success': True,
            'message': f"Successfully joined cluster {cluster_id}"
        }), 200
        
    except Exception as e:
        logger.error(f"Error joining cluster: {e}")
        return jsonify({'error': 'Failed to join cluster'}), 500


@collab_api.route('/clusters/<cluster_id>/contribute', methods=['POST'])
def add_contribution(cluster_id: str):
    """Add a contribution to a collaborative knowledge cluster."""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['participant_id', 'content']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Add contribution asynchronously
        async def _add_contribution():
            return await collab_engine.add_contribution(
                cluster_id=cluster_id,
                participant_id=data['participant_id'],
                content=data['content'],
                content_type=data.get('content_type', 'text')
            )
        
        # Run async function
        contribution = asyncio.run(_add_contribution())
        
        if not contribution:
            return jsonify({'error': 'Failed to add contribution. Check cluster ID and participant ID.'}), 404
        
        return jsonify({
            'success': True,
            'contribution_id': contribution.id,
            'contribution': contribution.to_dict()
        }), 201
        
    except Exception as e:
        logger.error(f"Error adding contribution: {e}")
        return jsonify({'error': 'Failed to add contribution'}), 500


@collab_api.route('/clusters/<cluster_id>/session', methods=['POST'])
def start_live_session(cluster_id: str):
    """Start a live co-thinking session."""
    try:
        data = request.get_json()
        session_name = data.get('session_name', f'Live Session {datetime.now().strftime("%Y-%m-%d %H:%M")}')
        
        # Start session asynchronously
        async def _start_session():
            return await collab_engine.start_live_session(cluster_id, session_name)
        
        success = asyncio.run(_start_session())
        
        if not success:
            return jsonify({'error': 'Failed to start session. Cluster may not exist.'}), 404
        
        return jsonify({
            'success': True,
            'message': f"Live session '{session_name}' started successfully",
            'session_name': session_name
        }), 200
        
    except Exception as e:
        logger.error(f"Error starting live session: {e}")
        return jsonify({'error': 'Failed to start live session'}), 500


@collab_api.route('/clusters/<cluster_id>/insights', methods=['GET'])
def get_collaborative_insights(cluster_id: str):
    """Get collaborative insights for a cluster."""
    try:
        insights = collab_engine.get_collaborative_insights(cluster_id)
        
        if not insights:
            return jsonify({'error': 'Cluster not found'}), 404
        
        return jsonify({
            'success': True,
            'cluster_id': cluster_id,
            'insights': insights
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting insights: {e}")
        return jsonify({'error': 'Failed to get insights'}), 500


@collab_api.route('/clusters', methods=['GET'])
def list_clusters():
    """List all accessible knowledge clusters."""
    try:
        user_id = request.args.get('user_id')
        
        clusters = []
        for cluster in collab_engine.knowledge_clusters.values():
            # Basic access control - show public or user's clusters
            if (cluster.privacy_level == 'public' or 
                (user_id and any(p.user_id == user_id for p in cluster.participants))):
                
                cluster_info = {
                    'id': cluster.id,
                    'name': cluster.name,
                    'description': cluster.description,
                    'created_at': cluster.created_at.isoformat(),
                    'state': cluster.state.value,
                    'privacy_level': cluster.privacy_level,
                    'participants': len(cluster.participants),
                    'active_participants': len(cluster.get_active_participants()),
                    'recent_contributions': len(cluster.get_recent_contributions(hours=24))
                }
                clusters.append(cluster_info)
        
        return jsonify({
            'success': True,
            'clusters': clusters,
            'total_clusters': len(clusters)
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing clusters: {e}")
        return jsonify({'error': 'Failed to list clusters'}), 500


@collab_api.route('/clusters/<cluster_id>', methods=['GET'])
def get_cluster_details(cluster_id: str):
    """Get detailed information about a specific cluster."""
    try:
        if cluster_id not in collab_engine.knowledge_clusters:
            return jsonify({'error': 'Cluster not found'}), 404
        
        cluster = collab_engine.knowledge_clusters[cluster_id]
        
        return jsonify({
            'success': True,
            'cluster': cluster.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting cluster details: {e}")
        return jsonify({'error': 'Failed to get cluster details'}), 500


@collab_api.route('/metrics', methods=['GET'])
def get_collaboration_metrics():
    """Get overall collaboration metrics."""
    try:
        metrics = collab_engine.get_collaboration_metrics()
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return jsonify({'error': 'Failed to get metrics'}), 500


@collab_api.route('/status', methods=['GET'])
def get_system_status():
    """Get comprehensive system status."""
    try:
        status = collab_engine.get_system_status()
        
        return jsonify({
            'success': True,
            'status': status
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({'error': 'Failed to get system status'}), 500


# WebSocket Event Handlers
def init_websocket_handlers(socketio: SocketIO):
    """Initialize WebSocket event handlers for real-time collaboration."""
    
    @socketio.on('join_collaboration')
    def handle_join_collaboration(data):
        """Handle user joining a collaborative space."""
        try:
            cluster_id = data.get('cluster_id')
            user_id = data.get('user_id')
            user_name = data.get('user_name')
            
            if not all([cluster_id, user_id, user_name]):
                emit('error', {'message': 'Missing required fields for collaboration'})
                return
            
            # Join the cluster room
            join_room(f"cluster_{cluster_id}")
            
            # Update user session
            session['user_id'] = user_id
            session['user_name'] = user_name
            session['cluster_id'] = cluster_id
            
            # Notify others in the room
            emit('user_joined', {
                'user_id': user_id,
                'user_name': user_name,
                'timestamp': datetime.now().isoformat()
            }, room=f"cluster_{cluster_id}", include_self=False)
            
            # Send current cluster state to the user
            if cluster_id in collab_engine.knowledge_clusters:
                cluster = collab_engine.knowledge_clusters[cluster_id]
                emit('cluster_state', {
                    'cluster': cluster.to_dict(),
                    'active_participants': len(cluster.get_active_participants())
                })
            
            logger.info(f"User {user_name} joined collaboration room for cluster {cluster_id}")
            
        except Exception as e:
            logger.error(f"Error in join_collaboration: {e}")
            emit('error', {'message': 'Failed to join collaboration'})
    
    @socketio.on('leave_collaboration')
    def handle_leave_collaboration():
        """Handle user leaving a collaborative space."""
        try:
            user_id = session.get('user_id')
            user_name = session.get('user_name')
            cluster_id = session.get('cluster_id')
            
            if cluster_id:
                # Leave the cluster room
                leave_room(f"cluster_{cluster_id}")
                
                # Notify others in the room
                if user_id and user_name:
                    emit('user_left', {
                        'user_id': user_id,
                        'user_name': user_name,
                        'timestamp': datetime.now().isoformat()
                    }, room=f"cluster_{cluster_id}")
                
                logger.info(f"User {user_name} left collaboration room for cluster {cluster_id}")
            
            # Clear session
            session.clear()
            
        except Exception as e:
            logger.error(f"Error in leave_collaboration: {e}")
    
    @socketio.on('live_contribution')
    def handle_live_contribution(data):
        """Handle real-time contribution to collaborative space."""
        try:
            cluster_id = session.get('cluster_id')
            user_id = session.get('user_id')
            user_name = session.get('user_name')
            content = data.get('content')
            content_type = data.get('content_type', 'text')
            
            if not all([cluster_id, user_id, content]):
                emit('error', {'message': 'Missing required fields for contribution'})
                return
            
            # Add contribution asynchronously
            async def _add_live_contribution():
                return await collab_engine.add_contribution(
                    cluster_id=cluster_id,
                    participant_id=user_id,
                    content=content,
                    content_type=content_type
                )
            
            contribution = asyncio.run(_add_live_contribution())
            
            if contribution:
                # Broadcast to all users in the cluster
                emit('new_contribution', {
                    'contribution': contribution.to_dict(),
                    'contributor_name': user_name,
                    'timestamp': datetime.now().isoformat()
                }, room=f"cluster_{cluster_id}")
                
                # If insights were generated, broadcast them too
                if contribution.insights:
                    emit('new_insights', {
                        'insights': contribution.insights,
                        'contribution_id': contribution.id,
                        'contributor_name': user_name
                    }, room=f"cluster_{cluster_id}")
                
                logger.info(f"Live contribution added by {user_name} to cluster {cluster_id}")
            else:
                emit('error', {'message': 'Failed to add contribution'})
                
        except Exception as e:
            logger.error(f"Error in live_contribution: {e}")
            emit('error', {'message': 'Failed to process contribution'})
    
    @socketio.on('request_insights')
    def handle_request_insights():
        """Handle request for collaborative insights."""
        try:
            cluster_id = session.get('cluster_id')
            
            if not cluster_id:
                emit('error', {'message': 'No active collaboration session'})
                return
            
            insights = collab_engine.get_collaborative_insights(cluster_id)
            
            emit('insights_update', {
                'insights': insights,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in request_insights: {e}")
            emit('error', {'message': 'Failed to get insights'})
    
    @socketio.on('typing_indicator')
    def handle_typing_indicator(data):
        """Handle typing indicators for live collaboration."""
        try:
            cluster_id = session.get('cluster_id')
            user_id = session.get('user_id')
            user_name = session.get('user_name')
            is_typing = data.get('is_typing', False)
            
            if cluster_id and user_id and user_name:
                emit('user_typing', {
                    'user_id': user_id,
                    'user_name': user_name,
                    'is_typing': is_typing,
                    'timestamp': datetime.now().isoformat()
                }, room=f"cluster_{cluster_id}", include_self=False)
                
        except Exception as e:
            logger.error(f"Error in typing_indicator: {e}")
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle user disconnection."""
        try:
            user_id = session.get('user_id')
            user_name = session.get('user_name')
            cluster_id = session.get('cluster_id')
            
            if cluster_id and user_id and user_name:
                emit('user_disconnected', {
                    'user_id': user_id,
                    'user_name': user_name,
                    'timestamp': datetime.now().isoformat()
                }, room=f"cluster_{cluster_id}")
                
                logger.info(f"User {user_name} disconnected from cluster {cluster_id}")
                
        except Exception as e:
            logger.error(f"Error in disconnect: {e}")
    
    # Register event handlers with the collaborative intelligence engine
    def on_cluster_created(event):
        """Handle cluster creation events."""
        socketio.emit('cluster_created', event['data'], room='public')
    
    def on_contribution_added(event):
        """Handle contribution events."""
        cluster_id = event['data']['cluster_id']
        socketio.emit('contribution_added', event['data'], room=f"cluster_{cluster_id}")
    
    def on_live_session_started(event):
        """Handle live session start events."""
        cluster_id = event['data']['cluster_id']
        socketio.emit('live_session_started', event['data'], room=f"cluster_{cluster_id}")
    
    def on_participant_joined(event):
        """Handle participant join events."""
        cluster_id = event['data']['cluster_id']
        socketio.emit('participant_joined', event['data'], room=f"cluster_{cluster_id}")
    
    # Register event handlers
    collab_engine.register_event_handler('cluster_created', on_cluster_created)
    collab_engine.register_event_handler('contribution_added', on_contribution_added)
    collab_engine.register_event_handler('live_session_started', on_live_session_started)
    collab_engine.register_event_handler('participant_joined', on_participant_joined)
    
    logger.info("WebSocket handlers initialized for collaborative intelligence")


# Health check endpoint
@collab_api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for collaborative intelligence."""
    try:
        status = collab_engine.get_system_status()
        
        return jsonify({
            'status': 'healthy',
            'service': 'collaborative_intelligence',
            'timestamp': datetime.now().isoformat(),
            'engine_status': status['engine_status'],
            'active_clusters': len(status['active_clusters']),
            'version': '1.0.0'
        }), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'service': 'collaborative_intelligence',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


if __name__ == "__main__":
    # Demo usage
    print("🤝 Collaborative Intelligence API Demo")
    print("=" * 50)
    print("API endpoints available:")
    print("• POST /api/collaborative/clusters - Create cluster")
    print("• POST /api/collaborative/clusters/<id>/join - Join cluster")
    print("• POST /api/collaborative/clusters/<id>/contribute - Add contribution")
    print("• POST /api/collaborative/clusters/<id>/session - Start live session")
    print("• GET  /api/collaborative/clusters/<id>/insights - Get insights")
    print("• GET  /api/collaborative/clusters - List clusters")
    print("• GET  /api/collaborative/clusters/<id> - Get cluster details")
    print("• GET  /api/collaborative/metrics - Get metrics")
    print("• GET  /api/collaborative/status - Get system status")
    print("• GET  /api/collaborative/health - Health check")
    print("\nWebSocket events:")
    print("• join_collaboration - Join collaborative space")
    print("• leave_collaboration - Leave collaborative space")
    print("• live_contribution - Real-time contribution")
    print("• request_insights - Request collaborative insights")
    print("• typing_indicator - Show typing status")
    print("\n🚀 Ready for revolutionary collaborative intelligence!")