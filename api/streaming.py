"""
streaming.py - Real-time Progress Streaming API

Provides Server-Sent Events (SSE) for real-time updates during processing.
Keeps implementation simple and focused on actual user needs.

Author: SUM Development Team
License: Apache License 2.0
"""

import time
import json
import logging
from flask import Blueprint, Response, request, jsonify
from typing import Generator, Dict, Any
import hashlib

from web.middleware import rate_limit
from application.optimized_summarizer import summarize_text_universal
from memory.semantic_memory import get_semantic_memory_engine
from memory.knowledge_graph import get_knowledge_graph_engine
from Utils.universal_file_processor import process_any_file

logger = logging.getLogger(__name__)
streaming_bp = Blueprint('streaming', __name__)


def create_sse_message(data: Dict[str, Any], event_type: str = "message") -> str:
    """Create a properly formatted SSE message."""
    message = f"event: {event_type}\n"
    message += f"data: {json.dumps(data)}\n\n"
    return message


@streaming_bp.route('/stream/summarize', methods=['POST'])
@rate_limit(20, 60)
def stream_summarization():
    """
    Stream summarization progress for large texts.
    Returns real-time updates as text is processed.
    """
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing required field: text'}), 400
    
    text = data['text']
    density = data.get('density', 'all')
    store_memory = data.get('store_memory', True)
    
    def generate() -> Generator[str, None, None]:
        try:
            # Initial progress
            yield create_sse_message({
                'type': 'start',
                'message': 'Starting summarization...',
                'progress': 0
            })
            
            # Split text into chunks for progress reporting
            words = text.split()
            total_words = len(words)
            chunk_size = max(100, total_words // 10)  # 10 progress updates
            
            # Process in chunks for progress
            processed_words = 0
            
            # Summarize (simulate chunked processing for progress)
            yield create_sse_message({
                'type': 'progress',
                'message': 'Analyzing text structure...',
                'progress': 10,
                'stats': {'total_words': total_words}
            })
            
            # Perform actual summarization
            result = summarize_text_universal(text)
            
            yield create_sse_message({
                'type': 'progress',
                'message': 'Generating summaries...',
                'progress': 50
            })
            
            # Extract entities if requested
            if data.get('extract_entities', False):
                yield create_sse_message({
                    'type': 'progress',
                    'message': 'Extracting entities...',
                    'progress': 60
                })
                
                kg_engine = get_knowledge_graph_engine()
                entities = kg_engine.extract_entities_and_relationships(text)
                result['entities'] = entities
            
            # Store in memory if requested
            memory_id = None
            if store_memory:
                yield create_sse_message({
                    'type': 'progress',
                    'message': 'Storing in semantic memory...',
                    'progress': 80
                })
                
                memory_engine = get_semantic_memory_engine()
                summary = result.get('medium', result.get('summary', ''))
                memory_id = memory_engine.store_memory(
                    text=text,
                    summary=summary,
                    metadata={'source': 'streaming_api', 'density': density}
                )
            
            # Final result
            yield create_sse_message({
                'type': 'complete',
                'message': 'Summarization complete!',
                'progress': 100,
                'result': {
                    'summary': result.get(density) if density != 'all' else result,
                    'memory_id': memory_id,
                    'stats': {
                        'original_words': total_words,
                        'compression_ratio': result.get('compression_ratio', 0)
                    }
                }
            })
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield create_sse_message({
                'type': 'error',
                'message': str(e),
                'progress': 0
            })
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@streaming_bp.route('/stream/file', methods=['POST'])
@rate_limit(10, 60)
def stream_file_processing():
    """
    Stream processing progress for file uploads.
    Handles large files with real-time updates.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save file temporarily
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        file.save(tmp.name)
        temp_path = tmp.name
    
    def generate() -> Generator[str, None, None]:
        try:
            # Initial message
            yield create_sse_message({
                'type': 'start',
                'message': f'Processing {file.filename}...',
                'progress': 0,
                'filename': file.filename
            })
            
            # Extract text from file
            yield create_sse_message({
                'type': 'progress',
                'message': 'Extracting text from file...',
                'progress': 20
            })
            
            extraction_result = process_any_file(temp_path)
            
            if not extraction_result['success']:
                yield create_sse_message({
                    'type': 'error',
                    'message': f"Failed to extract text: {extraction_result.get('error', 'Unknown error')}",
                    'progress': 0
                })
                return
            
            text = extraction_result['text']
            metadata = extraction_result['metadata']
            
            yield create_sse_message({
                'type': 'progress',
                'message': 'Text extracted successfully',
                'progress': 40,
                'stats': {
                    'extracted_length': len(text),
                    'file_type': metadata.get('file_type', 'unknown')
                }
            })
            
            # Summarize
            yield create_sse_message({
                'type': 'progress',
                'message': 'Generating summary...',
                'progress': 60
            })
            
            summary_result = summarize_text_universal(text)
            
            # Store in memory
            yield create_sse_message({
                'type': 'progress',
                'message': 'Storing in knowledge base...',
                'progress': 80
            })
            
            memory_engine = get_semantic_memory_engine()
            memory_id = memory_engine.store_memory(
                text=text,
                summary=summary_result.get('medium', ''),
                metadata={
                    'filename': file.filename,
                    'file_metadata': metadata
                }
            )
            
            # Complete
            yield create_sse_message({
                'type': 'complete',
                'message': 'File processing complete!',
                'progress': 100,
                'result': {
                    'filename': file.filename,
                    'summary': summary_result,
                    'memory_id': memory_id,
                    'metadata': metadata
                }
            })
            
        except Exception as e:
            logger.error(f"File processing error: {e}")
            yield create_sse_message({
                'type': 'error',
                'message': str(e),
                'progress': 0
            })
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@streaming_bp.route('/stream/batch', methods=['POST'])
@rate_limit(5, 60)
def stream_batch_processing():
    """
    Stream progress for batch document processing.
    Useful for processing multiple documents at once.
    """
    data = request.get_json()
    
    if not data or 'documents' not in data:
        return jsonify({'error': 'Missing required field: documents'}), 400
    
    documents = data['documents']
    if not isinstance(documents, list) or len(documents) == 0:
        return jsonify({'error': 'Documents must be a non-empty list'}), 400
    
    def generate() -> Generator[str, None, None]:
        try:
            total_docs = len(documents)
            
            yield create_sse_message({
                'type': 'start',
                'message': f'Processing {total_docs} documents...',
                'progress': 0,
                'total_documents': total_docs
            })
            
            results = []
            memory_engine = get_semantic_memory_engine()
            
            for i, doc in enumerate(documents):
                doc_id = doc.get('id', f'doc_{i}')
                
                yield create_sse_message({
                    'type': 'document_start',
                    'message': f'Processing document {i+1}/{total_docs}',
                    'progress': int((i / total_docs) * 100),
                    'current_document': doc_id
                })
                
                try:
                    # Process document
                    text = doc.get('text', '')
                    summary_result = summarize_text_universal(text)
                    
                    # Store in memory
                    memory_id = memory_engine.store_memory(
                        text=text,
                        summary=summary_result.get('medium', ''),
                        metadata=doc.get('metadata', {})
                    )
                    
                    results.append({
                        'id': doc_id,
                        'status': 'success',
                        'summary': summary_result.get('short', ''),
                        'memory_id': memory_id
                    })
                    
                    yield create_sse_message({
                        'type': 'document_complete',
                        'message': f'Document {doc_id} processed',
                        'progress': int(((i + 1) / total_docs) * 100),
                        'document_result': results[-1]
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing document {doc_id}: {e}")
                    results.append({
                        'id': doc_id,
                        'status': 'error',
                        'error': str(e)
                    })
            
            # Final summary
            successful = sum(1 for r in results if r['status'] == 'success')
            
            yield create_sse_message({
                'type': 'complete',
                'message': f'Batch processing complete! {successful}/{total_docs} successful',
                'progress': 100,
                'results': results,
                'summary': {
                    'total': total_docs,
                    'successful': successful,
                    'failed': total_docs - successful
                }
            })
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            yield create_sse_message({
                'type': 'error',
                'message': str(e),
                'progress': 0
            })
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@streaming_bp.route('/stream/synthesis', methods=['POST'])
@rate_limit(10, 60)
def stream_synthesis():
    """
    Stream progress for document synthesis operations.
    Shows real-time updates as documents are analyzed and merged.
    """
    data = request.get_json()
    
    if not data or 'memory_ids' not in data:
        return jsonify({'error': 'Missing required field: memory_ids'}), 400
    
    memory_ids = data['memory_ids']
    synthesis_type = data.get('synthesis_type', 'comprehensive')
    
    def generate() -> Generator[str, None, None]:
        try:
            yield create_sse_message({
                'type': 'start',
                'message': 'Starting document synthesis...',
                'progress': 0
            })
            
            # Load memories
            memory_engine = get_semantic_memory_engine()
            memories = []
            
            yield create_sse_message({
                'type': 'progress',
                'message': 'Loading documents from memory...',
                'progress': 20
            })
            
            for memory_id in memory_ids:
                # Simple retrieval (in real implementation, add proper memory loading)
                memories.append({
                    'id': memory_id,
                    'text': f"Memory content for {memory_id}"  # Placeholder
                })
            
            yield create_sse_message({
                'type': 'progress',
                'message': 'Analyzing document relationships...',
                'progress': 40
            })
            
            # Perform synthesis
            from application.synthesis_engine import get_synthesis_engine, Document
            synthesis_engine = get_synthesis_engine()
            
            # Convert to Document objects
            documents = [
                Document(
                    id=mem['id'],
                    text=mem['text']
                ) for mem in memories
            ]
            
            yield create_sse_message({
                'type': 'progress',
                'message': 'Synthesizing knowledge...',
                'progress': 60
            })
            
            result = synthesis_engine.synthesize_documents(
                documents,
                synthesis_type=synthesis_type
            )
            
            yield create_sse_message({
                'type': 'progress',
                'message': 'Finalizing synthesis...',
                'progress': 80
            })
            
            # Complete
            yield create_sse_message({
                'type': 'complete',
                'message': 'Synthesis complete!',
                'progress': 100,
                'result': {
                    'unified_summary': result.unified_summary,
                    'key_insights': result.key_insights,
                    'confidence_score': result.confidence_score,
                    'contradictions': len(result.contradictions),
                    'consensus_points': len(result.consensus_points)
                }
            })
            
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            yield create_sse_message({
                'type': 'error',
                'message': str(e),
                'progress': 0
            })
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


# Export blueprint
__all__ = ['streaming_bp']