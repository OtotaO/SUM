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

@extrapolation_bp.route('/api/extrapolate/stream', methods=['POST'])
def stream_extrapolation():
    """
    Stream the extrapolation process token by token.
    Uses Server-Sent Events (SSE) with structured data.
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
            
            # The engine now yields dicts {type, content}
            async for event in engine.stream_extrapolate(seed, config):
                # We wrap it in a JSON object for SSE
                yield f"data: {json.dumps(event)}\n\n"
                
            yield f"data: {json.dumps({'status': 'complete'})}\n\n"
        
        # Helper to run async generator in sync Flask
        def sync_generator():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
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

from unlimited_extrapolation_processor import get_unlimited_extrapolator

@extrapolation_bp.route('/api/extrapolate/stream_unlimited', methods=['POST'])
def stream_extrapolation_unlimited():
    """
    Stream unlimited extrapolation (e.g. multi-volume books).
    Uses Server-Sent Events (SSE) to emit text and progress updates.
    """
    try:
        data = request.get_json()
        seed = data.get('seed')

        if not seed:
            return jsonify({'error': 'No seed text provided'}), 400

        config = ExtrapolationConfig(
            target_format=data.get('target_format', 'book'),
            style=data.get('style', 'neutral'),
            tone=data.get('tone', 'neutral'),
            length_words=data.get('length_words', 10000),
            creativity=data.get('creativity', 0.7)
        )

        processor = get_unlimited_extrapolator()

        async def generate():
            yield f"data: {json.dumps({'status': 'initializing'})}\n\n"

            async for event in processor.process_extrapolation_stream(seed, config):
                yield f"data: {json.dumps(event)}\n\n"

            yield f"data: {json.dumps({'status': 'complete'})}\n\n"

        # Helper to run async generator in sync Flask
        def sync_generator():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
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
        logger.error(f"Unlimited stream extrapolation error: {str(e)}")
        return jsonify({'error': str(e)}), 500
