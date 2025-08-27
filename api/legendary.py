"""
Legendary API - Integrating GraphRAG, Multi-Agent, and RAPTOR
The ultimate knowledge crystallization endpoints
"""

from flask import Blueprint, request, jsonify, Response, stream_with_context
import asyncio
import json
import time
from typing import Dict, Any, List, Optional
import logging

# Import our legendary components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_rag_crystallizer import GraphRAGCrystallizer
from multi_agent_orchestrator import MultiAgentOrchestrator
from raptor_hierarchical import RAPTORBuilder, RAPTORQueryEngine
from knowledge_crystallizer import KnowledgeCrystallizer, CrystallizationConfig, DensityLevel

logger = logging.getLogger(__name__)

legendary_bp = Blueprint('legendary', __name__)

# Initialize legendary components
graph_rag = GraphRAGCrystallizer()
orchestrator = MultiAgentOrchestrator()
raptor_builder = RAPTORBuilder()
raptor_query = RAPTORQueryEngine()
crystallizer = KnowledgeCrystallizer()

# Cache for processed documents
document_cache = {}
tree_cache = {}


@legendary_bp.route('/api/legendary/graphrag', methods=['POST'])
def graphrag_crystallize():
    """
    GraphRAG corpus-level crystallization
    Handles questions like "What are the main themes?" across entire document collections
    """
    try:
        data = request.get_json()
        documents = data.get('documents', [])
        query = data.get('query')
        
        if not documents:
            return jsonify({'error': 'No documents provided'}), 400
        
        # Process with GraphRAG
        start_time = time.time()
        result = graph_rag.crystallize_corpus(documents, query)
        processing_time = time.time() - start_time
        
        # Handle specific query types
        if query:
            answer = graph_rag.answer_global_question(result, query)
        else:
            answer = result.global_summary
        
        response = {
            'answer': answer,
            'global_summary': result.global_summary,
            'hierarchical_summaries': result.hierarchical_summaries,
            'num_communities': len(result.communities),
            'communities': [
                {
                    'id': c.id,
                    'summary': c.summary,
                    'size': len(c.entities),
                    'level': c.level
                }
                for c in result.communities[:10]  # Top 10 communities
            ],
            'metadata': result.metadata,
            'processing_time': processing_time
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"GraphRAG error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@legendary_bp.route('/api/legendary/multiagent', methods=['POST'])
def multiagent_orchestrate():
    """
    Multi-agent orchestration for comprehensive summarization
    Leverages 10+ specialized AI agents working in parallel
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        parameters = data.get('parameters', {})
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Run async orchestration in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        start_time = time.time()
        result = loop.run_until_complete(
            orchestrator.orchestrate_crystallization(text, parameters)
        )
        processing_time = time.time() - start_time
        
        response = {
            'summary': result.summary,
            'essence': result.essence,
            'style_variations': result.style_variations,
            'facts': result.facts[:10],  # Top 10 facts
            'entities': result.entities[:20],  # Top 20 entities
            'quotes': result.quotes[:5],  # Top 5 quotes
            'sentiment': result.sentiment,
            'quality_score': result.quality_score,
            'consensus_score': result.consensus_score,
            'agent_contributions': result.agent_contributions,
            'processing_time': processing_time
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Multi-agent error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@legendary_bp.route('/api/legendary/raptor', methods=['POST'])
def raptor_hierarchical():
    """
    RAPTOR hierarchical tree summarization
    Multi-level abstraction from details to essence
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        query = data.get('query')
        level = data.get('level')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Generate cache key
        cache_key = hash(text)
        
        # Build or retrieve tree
        if cache_key in tree_cache:
            tree = tree_cache[cache_key]
        else:
            start_time = time.time()
            tree = raptor_builder.build_tree(text)
            tree_cache[cache_key] = tree
            build_time = time.time() - start_time
        
        response = {
            'root_summary': tree.root.summary,
            'levels': {},
            'metadata': tree.metadata
        }
        
        # Add summaries from each level
        for level_idx, nodes in tree.levels.items():
            level_summaries = [node.summary for node in nodes[:5]]  # Top 5 per level
            response['levels'][f'level_{level_idx}'] = level_summaries
        
        # Handle query if provided
        if query:
            results = raptor_query.query(tree, query, top_k=5, level=level)
            response['query_results'] = [
                {
                    'summary': node.summary,
                    'level': node.chunk.level,
                    'score': float(score)
                }
                for node, score in results
            ]
            
            # Get optimal context window
            response['context'] = raptor_query.get_context_window(tree, query)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"RAPTOR error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@legendary_bp.route('/api/legendary/unified', methods=['POST'])
def unified_crystallization():
    """
    Unified crystallization using all legendary technologies
    The ultimate summarization endpoint
    """
    try:
        data = request.get_json()
        text = data.get('text')
        documents = data.get('documents', [text] if text else [])
        density = data.get('density', 'standard')
        style = data.get('style', 'neutral')
        
        if not documents and not text:
            return jsonify({'error': 'No content provided'}), 400
        
        start_time = time.time()
        
        # 1. GraphRAG for corpus understanding
        graphrag_result = None
        if len(documents) > 1:
            graphrag_result = graph_rag.crystallize_corpus(documents)
            corpus_summary = graphrag_result.global_summary
        else:
            corpus_summary = None
        
        # 2. RAPTOR for hierarchical structure
        combined_text = ' '.join(documents) if documents else text
        raptor_tree = raptor_builder.build_tree(combined_text)
        
        # 3. Multi-agent orchestration for quality
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        agent_result = loop.run_until_complete(
            orchestrator.orchestrate_crystallization(
                raptor_tree.root.summary,
                {'style': style}
            )
        )
        
        # 4. Standard crystallization for density control
        config = CrystallizationConfig()
        config.density = DensityLevel[density.upper()]
        crystal_result = crystallizer.crystallize(agent_result.summary, config)
        
        processing_time = time.time() - start_time
        
        response = {
            'unified_summary': crystal_result.levels.get(density, agent_result.summary),
            'essence': agent_result.essence,
            'hierarchical_levels': {
                f'level_{i}': [n.summary for n in nodes[:3]]
                for i, nodes in raptor_tree.levels.items()
            },
            'key_themes': [c.summary for c in graphrag_result.communities[:5]] if graphrag_result else [],
            'facts': agent_result.facts[:10],
            'entities': agent_result.entities[:15],
            'sentiment': agent_result.sentiment,
            'quality_metrics': {
                'agent_consensus': agent_result.consensus_score,
                'crystallization_quality': crystal_result.quality_score,
                'multi_agent_quality': agent_result.quality_score
            },
            'processing_time': processing_time,
            'technologies_used': [
                'GraphRAG' if graphrag_result else None,
                'RAPTOR',
                'Multi-Agent',
                'Crystallization'
            ]
        }
        
        # Filter out None values
        response['technologies_used'] = [t for t in response['technologies_used'] if t]
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Unified crystallization error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@legendary_bp.route('/api/legendary/stream', methods=['POST'])
def stream_legendary():
    """
    Stream progressive crystallization using all technologies
    Real-time updates as processing happens
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        def generate():
            """Generate streaming results"""
            
            # Stage 1: Quick essence
            yield f"data: {json.dumps({'stage': 'essence', 'status': 'processing'})}\n\n"
            config = CrystallizationConfig()
            config.density = DensityLevel.ESSENCE
            essence_result = crystallizer.crystallize(text, config)
            yield f"data: {json.dumps({'stage': 'essence', 'result': essence_result.essence})}\n\n"
            
            # Stage 2: RAPTOR tree building
            yield f"data: {json.dumps({'stage': 'raptor', 'status': 'building_tree'})}\n\n"
            tree = raptor_builder.build_tree(text)
            yield f"data: {json.dumps({'stage': 'raptor', 'result': {'levels': tree.metadata['total_levels'], 'nodes': tree.metadata['total_nodes']}})}\n\n"
            
            # Stage 3: Multi-agent processing
            yield f"data: {json.dumps({'stage': 'agents', 'status': 'orchestrating'})}\n\n"
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            agent_result = loop.run_until_complete(
                orchestrator.orchestrate_crystallization(text[:1000])  # Sample for speed
            )
            yield f"data: {json.dumps({'stage': 'agents', 'result': {'consensus': agent_result.consensus_score, 'quality': agent_result.quality_score}})}\n\n"
            
            # Stage 4: Final summary
            yield f"data: {json.dumps({'stage': 'final', 'status': 'crystallizing'})}\n\n"
            final_summary = agent_result.summary if agent_result.summary else tree.root.summary
            yield f"data: {json.dumps({'stage': 'final', 'result': final_summary})}\n\n"
            
            # Complete
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
        logger.error(f"Streaming error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@legendary_bp.route('/api/legendary/capabilities', methods=['GET'])
def get_capabilities():
    """
    Get information about legendary capabilities
    """
    return jsonify({
        'technologies': {
            'graphrag': {
                'name': 'GraphRAG',
                'description': 'Microsoft Research approach for corpus-level summarization',
                'capabilities': [
                    'Handle millions of documents',
                    'Answer global questions',
                    'Detect themes and patterns',
                    'Build knowledge graphs'
                ]
            },
            'multiagent': {
                'name': 'Multi-Agent Orchestration',
                'description': '10+ specialized AI agents working in parallel',
                'agents': [
                    'Essence Extractor',
                    'Style Specialist',
                    'Fact Checker',
                    'Coherence Validator',
                    'Sentiment Analyzer',
                    'Entity Recognizer',
                    'Quote Extractor',
                    'Structure Analyzer',
                    'Quality Assessor'
                ]
            },
            'raptor': {
                'name': 'RAPTOR',
                'description': 'Recursive Abstractive Processing for Tree-Organized Retrieval',
                'capabilities': [
                    'Multi-level abstraction',
                    'Hierarchical summarization',
                    'Flexible querying',
                    'Semantic clustering'
                ]
            },
            'unified': {
                'name': 'Unified Crystallization',
                'description': 'Combines all technologies for ultimate summarization',
                'features': [
                    'Best of all approaches',
                    'Adaptive processing',
                    'Quality consensus',
                    'Multi-dimensional analysis'
                ]
            }
        },
        'endpoints': {
            '/api/legendary/graphrag': 'GraphRAG corpus analysis',
            '/api/legendary/multiagent': 'Multi-agent orchestration',
            '/api/legendary/raptor': 'RAPTOR hierarchical trees',
            '/api/legendary/unified': 'Combined legendary processing',
            '/api/legendary/stream': 'Real-time streaming results'
        },
        'status': 'operational',
        'version': '2.0.0-legendary'
    })


# Error handlers
@legendary_bp.errorhandler(Exception)
def handle_error(error):
    logger.error(f"Legendary API error: {str(error)}")
    return jsonify({'error': 'Internal server error', 'message': str(error)}), 500