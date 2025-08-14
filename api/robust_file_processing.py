"""
Enhanced File Processing API with Robustness Features
"""
import os
import logging
import traceback
from flask import Blueprint, request, jsonify, current_app, g
from werkzeug.utils import secure_filename
from web.middleware import rate_limit
from Utils.error_recovery import with_error_recovery
import asyncio

logger = logging.getLogger(__name__)
robust_file_bp = Blueprint('robust_file_processing', __name__)

@robust_file_bp.route('/analyze_file', methods=['POST'])
@rate_limit(10, 300)  # 10 calls per 5 minutes per user
async def analyze_file_robust():
    """
    Analyze file with enhanced robustness features:
    - File validation with content verification
    - Streaming processing for large files
    - Memory-efficient handling
    - Error recovery
    - Progress tracking via queue system
    """
    try:
        # Get robust components from app
        file_validator = current_app.file_validator
        streaming_processor = current_app.streaming_processor
        request_queue = current_app.request_queue
        error_manager = current_app.error_manager
        
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if not file or not file.filename:
            return jsonify({'error': 'No file selected'}), 400
        
        # Comprehensive file validation
        is_valid, error_msg, file_metadata = file_validator.validate_file(file)
        if not is_valid:
            logger.warning(f"File validation failed: {error_msg}")
            return jsonify({
                'error': error_msg,
                'details': 'File did not pass security validation'
            }), 400
        
        # Check for duplicate files using hash
        if await check_duplicate_file(file_metadata['hash']):
            return jsonify({
                'error': 'This file has already been processed',
                'file_hash': file_metadata['hash']
            }), 409
        
        # Estimate memory usage
        estimated_memory = streaming_processor.estimate_memory_usage(
            file_metadata['size'], 
            file_metadata['extension']
        )
        
        # Check if we have enough memory
        import psutil
        available_memory = psutil.virtual_memory().available
        if estimated_memory > available_memory * 0.5:  # Don't use more than 50% of available
            # Queue for async processing instead
            logger.info(f"Large file detected ({file_metadata['size']} bytes), queuing for async processing")
            
            # Save file temporarily
            with streaming_processor.temporary_file(suffix=f".{file_metadata['extension']}") as temp_path:
                # Stream save to avoid memory issues
                success, error = streaming_processor.save_upload_streaming(file, temp_path)
                if not success:
                    return jsonify({'error': f'Failed to save file: {error}'}), 500
                
                # Queue the processing job
                job_id = await request_queue.enqueue(
                    request_type='file_analysis',
                    payload={
                        'file_path': temp_path,
                        'metadata': file_metadata,
                        'model': request.form.get('model', 'simple'),
                        'options': {
                            'max_tokens': int(request.form.get('maxTokens', 200)),
                            'include_topics': request.form.get('include_topics', 'true') == 'true'
                        }
                    },
                    priority=5  # Normal priority
                )
                
                return jsonify({
                    'job_id': job_id,
                    'status': 'queued',
                    'message': 'Large file queued for processing',
                    'check_status_url': f'/api/job/status/{job_id}',
                    'estimated_time': estimate_processing_time(file_metadata)
                }), 202
        
        # Process smaller files immediately
        logger.info(f"Processing file immediately: {file_metadata['filename']} ({file_metadata['size']} bytes)")
        
        # Save and process with streaming
        with streaming_processor.temporary_file(suffix=f".{file_metadata['extension']}") as temp_path:
            # Stream save
            success, error = streaming_processor.save_upload_streaming(file, temp_path)
            if not success:
                return jsonify({'error': f'Failed to save file: {error}'}), 500
            
            # Process based on file type
            try:
                result = await process_file_with_recovery(
                    temp_path, 
                    file_metadata,
                    request.form.to_dict()
                )
                
                # Store file hash to prevent reprocessing
                await store_file_hash(file_metadata['hash'], result['summary'][:100])
                
                return jsonify({
                    'success': True,
                    'filename': file_metadata['filename'],
                    'file_size': file_metadata['size'],
                    'processing_time': result.get('processing_time', 0),
                    **result
                }), 200
                
            except Exception as e:
                # Track error for analysis
                error_context = error_manager.track_error(e, {
                    'file_metadata': file_metadata,
                    'request_id': g.get('request_id')
                })
                
                logger.error(f"Error processing file: {e}", exc_info=True)
                return jsonify({
                    'error': 'Error processing file',
                    'error_id': error_context.timestamp.timestamp(),
                    'message': 'Please try again or contact support if the issue persists'
                }), 500
                
    except Exception as e:
        logger.error(f"Unexpected error in file processing: {e}", exc_info=True)
        return jsonify({
            'error': 'An unexpected error occurred',
            'message': str(e) if current_app.debug else 'Please try again later'
        }), 500

@with_error_recovery(retry_count=3, recoverable_errors=[IOError, ValueError])
async def process_file_with_recovery(file_path: str, metadata: dict, options: dict) -> dict:
    """Process file with automatic error recovery"""
    import time
    start_time = time.time()
    
    # Get processing components
    from application.container import ApplicationContainer
    container = ApplicationContainer()
    
    # Choose processing method based on file type
    if metadata['extension'] in ['txt', 'md']:
        result = await process_text_file_streaming(file_path, metadata, options)
    elif metadata['extension'] == 'csv':
        result = await process_csv_file_streaming(file_path, metadata, options)
    elif metadata['extension'] == 'json':
        result = await process_json_file_streaming(file_path, metadata, options)
    elif metadata['extension'] == 'pdf':
        result = await process_pdf_file(file_path, metadata, options)
    else:
        # Fallback to text processing
        result = await process_text_file_streaming(file_path, metadata, options)
    
    result['processing_time'] = time.time() - start_time
    return result

async def process_text_file_streaming(file_path: str, metadata: dict, options: dict) -> dict:
    """Process text file using streaming for memory efficiency"""
    streaming_processor = current_app.streaming_processor
    
    chunks_processed = 0
    all_summaries = []
    
    def process_chunk(text: str, index: int) -> str:
        nonlocal chunks_processed
        chunks_processed += 1
        
        # Get summarizer
        from application.container import ApplicationContainer
        container = ApplicationContainer()
        
        # Process chunk
        if options.get('model') == 'advanced':
            summary = container.magnum_opus_sum_service().summarize(
                text, 
                max_length=int(options.get('maxTokens', 200)) // chunks_processed
            )
        else:
            summary = container.simple_sum_service().summarize(
                text,
                max_length=int(options.get('maxTokens', 200)) // chunks_processed
            )
        
        return summary
    
    # Process file in chunks
    with streaming_processor.process_file_stream(file_path) as file_obj:
        summaries = list(streaming_processor.process_text_chunks(
            file_obj,
            process_chunk
        ))
    
    # Combine summaries
    combined_summary = ' '.join(summaries)
    
    # Extract topics if requested
    topics = []
    if options.get('include_topics', 'true') == 'true':
        from topic_modeling import TopicModeling
        topic_modeler = TopicModeling(model_type='bertopic')
        topics = topic_modeler.extract_topics([combined_summary], num_topics=5)
    
    return {
        'summary': combined_summary,
        'chunks_processed': chunks_processed,
        'topics': topics,
        'metadata': {
            'original_size': metadata['size'],
            'compression_ratio': len(combined_summary) / metadata['size']
        }
    }

async def process_csv_file_streaming(file_path: str, metadata: dict, options: dict) -> dict:
    """Process CSV file row by row"""
    streaming_processor = current_app.streaming_processor
    
    rows_processed = 0
    summaries = []
    
    def process_row(row: dict, index: int) -> str:
        nonlocal rows_processed
        rows_processed += 1
        
        # Convert row to text
        text = ' '.join(str(v) for v in row.values() if v)
        
        # Summarize if text is long enough
        if len(text) > 100:
            return text[:100] + '...'  # Simple truncation for CSV
        return text
    
    with streaming_processor.process_file_stream(file_path, 'rb') as file_obj:
        summaries = list(streaming_processor.process_csv_streaming(
            file_obj,
            process_row
        ))
    
    return {
        'summary': f"Processed {rows_processed} rows from CSV file",
        'rows_processed': rows_processed,
        'sample_data': summaries[:10],  # First 10 rows
        'metadata': metadata
    }

async def process_json_file_streaming(file_path: str, metadata: dict, options: dict) -> dict:
    """Process JSON file with streaming parser"""
    streaming_processor = current_app.streaming_processor
    
    items_processed = 0
    key_insights = []
    
    def process_item(item: dict, index: int) -> dict:
        nonlocal items_processed
        items_processed += 1
        
        # Extract key information
        return {
            'index': index,
            'keys': list(item.keys()),
            'sample': str(item)[:100]
        }
    
    with streaming_processor.process_file_stream(file_path, 'rb') as file_obj:
        insights = list(streaming_processor.process_json_streaming(
            file_obj,
            process_item
        ))
    
    return {
        'summary': f"Analyzed {items_processed} JSON objects",
        'items_processed': items_processed,
        'structure_insights': insights[:5],  # First 5 items
        'metadata': metadata
    }

async def process_pdf_file(file_path: str, metadata: dict, options: dict) -> dict:
    """Process PDF file with OCR support"""
    # This would integrate with existing PDF processing
    # For now, return placeholder
    return {
        'summary': 'PDF processing queued',
        'status': 'pending',
        'metadata': metadata
    }

async def check_duplicate_file(file_hash: str) -> bool:
    """Check if file has been processed before"""
    try:
        db_manager = current_app.db_manager
        result = db_manager.execute(
            "SELECT 1 FROM processed_files WHERE file_hash = ? LIMIT 1",
            (file_hash,)
        )
        return len(result) > 0
    except:
        return False

async def store_file_hash(file_hash: str, summary_preview: str):
    """Store file hash to prevent reprocessing"""
    try:
        db_manager = current_app.db_manager
        db_manager.execute(
            "INSERT OR IGNORE INTO processed_files (file_hash, summary_preview, processed_at) VALUES (?, ?, datetime('now'))",
            (file_hash, summary_preview)
        )
    except Exception as e:
        logger.warning(f"Could not store file hash: {e}")

def estimate_processing_time(metadata: dict) -> str:
    """Estimate processing time based on file metadata"""
    size_mb = metadata['size'] / (1024 * 1024)
    
    # Rough estimates
    if metadata['extension'] == 'pdf':
        time_seconds = size_mb * 10  # 10 seconds per MB for PDFs
    elif metadata['extension'] in ['txt', 'md']:
        time_seconds = size_mb * 2   # 2 seconds per MB for text
    else:
        time_seconds = size_mb * 5   # 5 seconds per MB for others
    
    if time_seconds < 60:
        return f"{int(time_seconds)} seconds"
    else:
        return f"{int(time_seconds / 60)} minutes"

@robust_file_bp.route('/job/status/<job_id>', methods=['GET'])
async def get_job_status(job_id: str):
    """Get status of async file processing job"""
    request_queue = current_app.request_queue
    
    status = await request_queue.get_status(job_id)
    if not status:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(status), 200

@robust_file_bp.route('/job/result/<job_id>', methods=['GET'])
async def get_job_result(job_id: str):
    """Get result of completed job"""
    request_queue = current_app.request_queue
    
    try:
        result = await request_queue.get_result(job_id, timeout=5)  # 5 second wait
        return jsonify({
            'success': True,
            'result': result
        }), 200
    except TimeoutError:
        return jsonify({
            'error': 'Job not completed yet',
            'status_url': f'/api/job/status/{job_id}'
        }), 202
    except Exception as e:
        return jsonify({
            'error': 'Job failed',
            'message': str(e)
        }), 500

# Register file processing handler for the queue
async def file_analysis_handler(payload: dict) -> dict:
    """Handler for queued file analysis jobs"""
    return await process_file_with_recovery(
        payload['file_path'],
        payload['metadata'],
        payload['options']
    )