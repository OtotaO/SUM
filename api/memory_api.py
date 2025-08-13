"""
memory_api.py - API Endpoints for Semantic Memory and Knowledge Graph

This module provides REST API endpoints for:
- Storing and retrieving semantic memories
- Knowledge graph operations
- Cross-document synthesis
- Memory search and exploration

Author: SUM Development Team
License: Apache License 2.0
"""

import time
import logging
from flask import Blueprint, request, jsonify, Response
from typing import Dict, Any, List
import json

from web.middleware import rate_limit, validate_json_input
from memory.semantic_memory import get_semantic_memory_engine
from memory.knowledge_graph import get_knowledge_graph_engine

logger = logging.getLogger(__name__)
memory_bp = Blueprint('memory', __name__)


@memory_bp.route('/memory/store', methods=['POST'])
@rate_limit(30, 60)  # 30 calls per minute
@validate_json_input()
def store_memory():
    """
    Store a new memory entry with semantic embedding.
    
    Expected JSON:
    {
        "text": "Full text content",
        "summary": "Summarized version",
        "metadata": {"source": "...", "tags": [...]},
        "relationships": ["related_memory_id1", ...]
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        if 'text' not in data or 'summary' not in data:
            return jsonify({'error': 'Missing required fields: text, summary'}), 400
        
        # Get semantic memory engine
        memory_engine = get_semantic_memory_engine()
        
        # Store memory
        memory_id = memory_engine.store_memory(
            text=data['text'],
            summary=data['summary'],
            metadata=data.get('metadata', {}),
            relationships=data.get('relationships', [])
        )
        
        # Extract entities and relationships for knowledge graph
        kg_engine = get_knowledge_graph_engine()
        kg_results = kg_engine.extract_entities_and_relationships(
            data['text'],
            source=memory_id
        )
        
        return jsonify({
            'success': True,
            'memory_id': memory_id,
            'entities_extracted': len(kg_results['entities']),
            'relationships_extracted': len(kg_results['relationships'])
        })
        
    except Exception as e:
        logger.error(f"Memory storage error: {e}")
        return jsonify({'error': str(e)}), 500


@memory_bp.route('/memory/search', methods=['POST'])
@rate_limit(60, 60)  # 60 calls per minute
@validate_json_input()
def search_memories():
    """
    Search for relevant memories using semantic similarity.
    
    Expected JSON:
    {
        "query": "Search query text",
        "top_k": 5,
        "threshold": 0.7
    }
    """
    try:
        data = request.get_json()
        
        if 'query' not in data:
            return jsonify({'error': 'Missing required field: query'}), 400
        
        # Get semantic memory engine
        memory_engine = get_semantic_memory_engine()
        
        # Search memories
        results = memory_engine.search_memories(
            query=data['query'],
            top_k=data.get('top_k', 5),
            threshold=data.get('threshold', 0.7)
        )
        
        # Format results
        formatted_results = []
        for memory, score in results:
            formatted_results.append({
                'memory_id': memory.id,
                'summary': memory.summary,
                'score': float(score),
                'metadata': memory.metadata,
                'timestamp': memory.timestamp,
                'access_count': memory.access_count
            })
        
        return jsonify({
            'success': True,
            'results': formatted_results,
            'query': data['query']
        })
        
    except Exception as e:
        logger.error(f"Memory search error: {e}")
        return jsonify({'error': str(e)}), 500


@memory_bp.route('/memory/synthesize', methods=['POST'])
@rate_limit(20, 60)  # 20 calls per minute
@validate_json_input()
def synthesize_memories():
    """
    Synthesize insights from multiple memories.
    
    Expected JSON:
    {
        "memory_ids": ["id1", "id2", ...],
        "synthesis_type": "merge" | "compare" | "evolve"
    }
    """
    try:
        data = request.get_json()
        
        if 'memory_ids' not in data:
            return jsonify({'error': 'Missing required field: memory_ids'}), 400
        
        memory_ids = data['memory_ids']
        if not isinstance(memory_ids, list) or len(memory_ids) < 2:
            return jsonify({'error': 'At least 2 memory IDs required'}), 400
        
        # Get semantic memory engine
        memory_engine = get_semantic_memory_engine()
        
        # Synthesize memories
        synthesis = memory_engine.synthesize_memories(memory_ids)
        
        return jsonify({
            'success': True,
            'synthesis': synthesis
        })
        
    except Exception as e:
        logger.error(f"Memory synthesis error: {e}")
        return jsonify({'error': str(e)}), 500


@memory_bp.route('/memory/related/<memory_id>', methods=['GET'])
@rate_limit(60, 60)
def get_related_memories(memory_id: str):
    """Get memories related to a specific memory."""
    try:
        max_depth = request.args.get('max_depth', 2, type=int)
        
        # Get semantic memory engine
        memory_engine = get_semantic_memory_engine()
        
        # Get related memories
        related = memory_engine.get_related_memories(memory_id, max_depth)
        
        return jsonify({
            'success': True,
            'memory_id': memory_id,
            'related_memories': related
        })
        
    except Exception as e:
        logger.error(f"Related memories error: {e}")
        return jsonify({'error': str(e)}), 500


@memory_bp.route('/knowledge/entities', methods=['POST'])
@rate_limit(30, 60)
@validate_json_input()
def extract_entities():
    """
    Extract entities and relationships from text.
    
    Expected JSON:
    {
        "text": "Text to analyze",
        "source": "Source identifier"
    }
    """
    try:
        data = request.get_json()
        
        if 'text' not in data:
            return jsonify({'error': 'Missing required field: text'}), 400
        
        # Get knowledge graph engine
        kg_engine = get_knowledge_graph_engine()
        
        # Extract entities and relationships
        results = kg_engine.extract_entities_and_relationships(
            text=data['text'],
            source=data.get('source')
        )
        
        return jsonify({
            'success': True,
            **results
        })
        
    except Exception as e:
        logger.error(f"Entity extraction error: {e}")
        return jsonify({'error': str(e)}), 500


@memory_bp.route('/knowledge/path', methods=['POST'])
@rate_limit(40, 60)
@validate_json_input()
def find_knowledge_path():
    """
    Find paths between two entities in the knowledge graph.
    
    Expected JSON:
    {
        "start_entity_id": "entity1",
        "end_entity_id": "entity2",
        "max_length": 5
    }
    """
    try:
        data = request.get_json()
        
        if 'start_entity_id' not in data or 'end_entity_id' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Get knowledge graph engine
        kg_engine = get_knowledge_graph_engine()
        
        # Find paths
        paths = kg_engine.find_path(
            start_entity_id=data['start_entity_id'],
            end_entity_id=data['end_entity_id'],
            max_length=data.get('max_length', 5)
        )
        
        return jsonify({
            'success': True,
            'paths': paths,
            'path_count': len(paths)
        })
        
    except Exception as e:
        logger.error(f"Path finding error: {e}")
        return jsonify({'error': str(e)}), 500


@memory_bp.route('/knowledge/context/<entity_id>', methods=['GET'])
@rate_limit(60, 60)
def get_entity_context(entity_id: str):
    """Get the context around an entity in the knowledge graph."""
    try:
        depth = request.args.get('depth', 2, type=int)
        
        # Get knowledge graph engine
        kg_engine = get_knowledge_graph_engine()
        
        # Get entity context
        context = kg_engine.get_entity_context(entity_id, depth)
        
        return jsonify({
            'success': True,
            'context': context
        })
        
    except Exception as e:
        logger.error(f"Entity context error: {e}")
        return jsonify({'error': str(e)}), 500


@memory_bp.route('/knowledge/communities', methods=['GET'])
@rate_limit(10, 60)  # Expensive operation
def find_communities():
    """Find communities in the knowledge graph."""
    try:
        # Get knowledge graph engine
        kg_engine = get_knowledge_graph_engine()
        
        # Find communities
        communities = kg_engine.find_communities()
        
        return jsonify({
            'success': True,
            'communities': communities,
            'community_count': len(communities)
        })
        
    except Exception as e:
        logger.error(f"Community detection error: {e}")
        return jsonify({'error': str(e)}), 500


@memory_bp.route('/knowledge/visualize', methods=['POST'])
@rate_limit(20, 60)
@validate_json_input()
def visualize_knowledge():
    """
    Generate a visualization of the knowledge graph.
    
    Expected JSON:
    {
        "entity_ids": ["id1", "id2", ...],
        "show_labels": true
    }
    """
    try:
        data = request.get_json()
        
        if 'entity_ids' not in data:
            return jsonify({'error': 'Missing required field: entity_ids'}), 400
        
        # Get knowledge graph engine
        kg_engine = get_knowledge_graph_engine()
        
        # Generate visualization
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name
        
        result_path = kg_engine.visualize_subgraph(
            entity_ids=data['entity_ids'],
            output_path=output_path,
            show_labels=data.get('show_labels', True)
        )
        
        if result_path and os.path.exists(result_path):
            # Read the image and return as base64
            import base64
            with open(result_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode()
            
            # Clean up
            os.unlink(result_path)
            
            return jsonify({
                'success': True,
                'visualization': f'data:image/png;base64,{image_data}'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Visualization generation failed'
            }), 500
            
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        return jsonify({'error': str(e)}), 500


@memory_bp.route('/memory/stats', methods=['GET'])
def get_memory_stats():
    """Get statistics about the memory and knowledge systems."""
    try:
        # Get both engines
        memory_engine = get_semantic_memory_engine()
        kg_engine = get_knowledge_graph_engine()
        
        # Get stats
        memory_stats = memory_engine.get_memory_stats()
        kg_stats = kg_engine.get_stats()
        
        return jsonify({
            'success': True,
            'memory_stats': memory_stats,
            'knowledge_stats': kg_stats
        })
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'error': str(e)}), 500


@memory_bp.route('/memory/stream', methods=['POST'])
def stream_memory_search():
    """
    Stream memory search results as they are found.
    Uses Server-Sent Events for real-time updates.
    """
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({'error': 'Missing required field: query'}), 400
    
    def generate():
        try:
            memory_engine = get_semantic_memory_engine()
            
            # Start searching
            yield f"data: {json.dumps({'type': 'start', 'query': data['query']})}\n\n"
            
            # Get results
            results = memory_engine.search_memories(
                query=data['query'],
                top_k=data.get('top_k', 10),
                threshold=data.get('threshold', 0.5)
            )
            
            # Stream each result
            for i, (memory, score) in enumerate(results):
                result_data = {
                    'type': 'result',
                    'index': i,
                    'memory_id': memory.id,
                    'summary': memory.summary,
                    'score': float(score),
                    'metadata': memory.metadata
                }
                yield f"data: {json.dumps(result_data)}\n\n"
                time.sleep(0.1)  # Small delay for streaming effect
            
            # Send completion
            yield f"data: {json.dumps({'type': 'complete', 'total': len(results)})}\n\n"
            
        except Exception as e:
            error_data = {'type': 'error', 'message': str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


# Export blueprint
__all__ = ['memory_bp']