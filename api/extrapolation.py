"""
Extrapolation API - Bi-directional Knowledge Transformation
Turns simple tags/concepts into expanded text (reverse summarization).
"""

from flask import Blueprint, request, jsonify, Response, stream_with_context
import logging
import time
import asyncio
from typing import Dict, Any, List, Optional
import json

# Import the core extrapolation engine
from extrapolation_engine import ExtrapolationEngine, ExtrapolationConfig
from llm_backend import llm_backend

logger = logging.getLogger(__name__)

extrapolation_bp = Blueprint('extrapolation', __name__)

# Initialize engine
engine = ExtrapolationEngine()

@extrapolation_bp.route('/api/extrapolate', methods=['POST'])
def extrapolate_text():
    """
    Expand a seed (tags, concept, or short text) into a larger form (paragraph, article, book chapter).
    
    Expected JSON:
    {
        "seed": "The future of AI is symbiotic, not competitive.",
        "target_format": "paragraph|article|blog_post|story|essay",
        "style": "academic|creative|journalistic|business",
        "tone": "optimistic|neutral|critical",
        "length_words": 500
    }
    """
    try:
        data = request.get_json()
        seed = data.get('seed')
        
        if not seed:
            return jsonify({'error': 'No seed text provided'}), 400
            
        config = ExtrapolationConfig(
            target_format=data.get('target_format', 'paragraph'),
            style=data.get('style', 'neutral'),
            tone=data.get('tone', 'neutral'),
            length_words=data.get('length_words', 200),
            creativity=data.get('creativity', 0.7)
        )
        
        start_time = time.time()
        
        # Async execution in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(engine.extrapolate(seed, config))
        
        processing_time = time.time() - start_time
        
        return jsonify({
            'seed': seed,
            'result': result,
            'metadata': {
                'processing_time': processing_time,
                'input_length': len(seed.split()),
                'output_length': len(result.split()),
                'expansion_ratio': len(result.split()) / max(1, len(seed.split()))
            }
        })
        
    except Exception as e:
        logger.error(f"Extrapolation error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@extrapolation_bp.route('/api/extrapolate/stream', methods=['POST'])
def stream_extrapolation():
    """
    Stream the extrapolation process token by token.
    """
    try:
        data = request.get_json()
        seed = data.get('seed')
        
        if not seed:
            return jsonify({'error': 'No seed text provided'}), 400
            
        config = ExtrapolationConfig(
            target_format=data.get('target_format', 'paragraph'),
            style=data.get('style', 'neutral'),
            tone=data.get('tone', 'neutral'),
            length_words=data.get('length_words', 200),
            creativity=data.get('creativity', 0.7)
        )
        
        async def generate():
            yield f"data: {json.dumps({'status': 'initializing'})}\n\n"
            
            async for chunk in engine.stream_extrapolate(seed, config):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                
            yield f"data: {json.dumps({'status': 'complete'})}\n\n"
        
        # Helper to run async generator in sync Flask
        def sync_generator():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # This is a bit tricky in sync Flask, usually we'd use Quart or proper async
            # For now, we'll use a blocking iterator over the async generator if possible,
            # but since that's hard, we will simplify: 
            # We will run the generator to completion if it were a list, but for streaming
            # we really need an async server. 
            # FALLBACK: We will assume the engine supports a sync iterator or we bridge it.
            
            # Real implementation of bridging async gen to sync gen:
            try:
                agen = generate()
                while True:
                    try:
                        yield loop.run_until_complete(agen.__anext__())
                    except StopAsyncIteration:
                        break
            finally:
                loop.close()

        return Response(
            sync_generator(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )
        
    except Exception as e:
        logger.error(f"Stream extrapolation error: {str(e)}")
        return jsonify({'error': str(e)}), 500
