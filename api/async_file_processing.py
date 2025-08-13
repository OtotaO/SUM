"""
async_file_processing.py - Asynchronous File Processing with Background Tasks

Implements async file processing to prevent timeouts on large files:
- Background task processing using ThreadPoolExecutor
- Job status tracking and retrieval
- Progress streaming via SSE
- Automatic cleanup of completed jobs

Author: SUM Development Team
License: Apache License 2.0
"""

import os
import uuid
import time
import tempfile
import logging
import asyncio
import aiofiles
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, Response, current_app
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import json

from web.middleware import rate_limit, allowed_file
from core.engine import SumEngine
from FileReader.file_reader_factory import FileReaderFactory
from Utils.optimized_summarizer import summarize_text_universal
from Utils.error_handler import handle_errors, ProcessingError

logger = logging.getLogger(__name__)
async_file_bp = Blueprint('async_file_processing', __name__)

# Job tracking
jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = Lock()

# Background executor
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix='FileProcessor')

# Configuration
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
JOB_TIMEOUT = 3600  # 1 hour
CLEANUP_INTERVAL = 300  # 5 minutes


class AsyncFileProcessor:
    """Handles asynchronous file processing with progress tracking."""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.progress = 0
        self.status = 'initializing'
        self.result = None
        self.error = None
        self.start_time = time.time()
        
    def update_progress(self, progress: int, status: str):
        """Update job progress and status."""
        with jobs_lock:
            if self.job_id in jobs:
                jobs[self.job_id].update({
                    'progress': progress,
                    'status': status,
                    'updated_at': datetime.now().isoformat()
                })
    
    async def process_file_async(self, file_path: str, options: Dict[str, Any]):
        """Process file asynchronously with progress updates."""
        try:
            self.update_progress(10, 'reading_file')
            
            # Read file content
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            file_size = len(content)
            if file_size > MAX_FILE_SIZE:
                raise ProcessingError(f"File too large: {file_size} bytes (max: {MAX_FILE_SIZE})")
            
            self.update_progress(20, 'extracting_text')
            
            # Extract text based on file type
            _, ext = os.path.splitext(file_path)
            if ext.lower() in ['.pdf', '.docx', '.pptx']:
                # Use FileReaderFactory for complex formats
                reader_factory = FileReaderFactory()
                reader = reader_factory.create_reader(ext[1:])
                content = reader.read_content(file_path)
            
            self.update_progress(40, 'processing_chunks')
            
            # Process in chunks for large files
            chunk_size = 10000  # Characters per chunk
            chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
            
            summaries = []
            total_chunks = len(chunks)
            
            for i, chunk in enumerate(chunks):
                # Update progress for each chunk
                chunk_progress = 40 + int((i / total_chunks) * 40)
                self.update_progress(chunk_progress, f'processing_chunk_{i+1}_of_{total_chunks}')
                
                # Process chunk
                summary = await self._process_chunk(chunk, options)
                summaries.append(summary)
                
                # Yield control to prevent blocking
                await asyncio.sleep(0.1)
            
            self.update_progress(80, 'combining_results')
            
            # Combine chunk summaries
            combined_text = ' '.join(summaries)
            
            # Final summarization
            final_summary = await self._process_chunk(combined_text, {
                **options,
                'max_length': options.get('max_length', 200)
            })
            
            self.update_progress(90, 'generating_metadata')
            
            # Generate metadata
            metadata = {
                'file_size': file_size,
                'chunks_processed': total_chunks,
                'processing_time': time.time() - self.start_time,
                'algorithm': options.get('algorithm', 'auto')
            }
            
            self.update_progress(100, 'completed')
            
            return {
                'summary': final_summary,
                'metadata': metadata,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error processing file {self.job_id}: {e}")
            self.update_progress(0, 'failed')
            raise
    
    async def _process_chunk(self, text: str, options: Dict[str, Any]) -> str:
        """Process a single text chunk."""
        # Run CPU-intensive summarization in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            summarize_text_universal,
            text,
            options.get('max_length', 100),
            options.get('algorithm', 'auto')
        )


def process_file_background(job_id: str, file_path: str, options: Dict[str, Any]):
    """Background task to process file."""
    processor = AsyncFileProcessor(job_id)
    
    try:
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run async processing
        result = loop.run_until_complete(
            processor.process_file_async(file_path, options)
        )
        
        # Update job status
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id].update({
                    'status': 'completed',
                    'result': result,
                    'completed_at': datetime.now().isoformat(),
                    'processing_time': time.time() - processor.start_time
                })
                
    except Exception as e:
        # Update job with error
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id].update({
                    'status': 'failed',
                    'error': str(e),
                    'completed_at': datetime.now().isoformat()
                })
    finally:
        # Clean up temp file
        try:
            os.unlink(file_path)
        except Exception as e:
            logger.warning(f"Failed to delete temp file {file_path}: {e}")
        
        # Close event loop
        loop.close()


def cleanup_old_jobs():
    """Remove old completed jobs to prevent memory bloat."""
    cutoff_time = datetime.now() - timedelta(seconds=JOB_TIMEOUT)
    
    with jobs_lock:
        jobs_to_remove = []
        for job_id, job_data in jobs.items():
            # Remove completed or failed jobs older than timeout
            if job_data['status'] in ['completed', 'failed']:
                completed_at = job_data.get('completed_at')
                if completed_at:
                    completed_time = datetime.fromisoformat(completed_at)
                    if completed_time < cutoff_time:
                        jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del jobs[job_id]
        
        if jobs_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")


@async_file_bp.route('/async/upload', methods=['POST'])
@rate_limit(10, 60)
@handle_errors
def upload_file_async():
    """
    Upload file for asynchronous processing.
    Returns immediately with a job ID.
    
    Expected form data:
    - file: File to process
    - max_length: Maximum summary length (optional)
    - algorithm: Summarization algorithm (optional)
    """
    # Validate file
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save file temporarily
    filename = secure_filename(file.filename)
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"{job_id}_{filename}")
    file.save(temp_path)
    
    # Get processing options
    options = {
        'max_length': int(request.form.get('max_length', 100)),
        'algorithm': request.form.get('algorithm', 'auto')
    }
    
    # Create job entry
    with jobs_lock:
        jobs[job_id] = {
            'id': job_id,
            'filename': filename,
            'status': 'queued',
            'progress': 0,
            'created_at': datetime.now().isoformat(),
            'options': options
        }
    
    # Submit to background processing
    executor.submit(process_file_background, job_id, temp_path, options)
    
    # Cleanup old jobs periodically
    if len(jobs) % 10 == 0:
        executor.submit(cleanup_old_jobs)
    
    return jsonify({
        'job_id': job_id,
        'status': 'queued',
        'message': 'File uploaded successfully. Processing started.',
        'status_url': f'/api/async/status/{job_id}',
        'stream_url': f'/api/async/stream/{job_id}'
    }), 202


@async_file_bp.route('/async/status/<job_id>', methods=['GET'])
def get_job_status(job_id: str):
    """Get the status of an async processing job."""
    with jobs_lock:
        job = jobs.get(job_id)
    
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    # Return job info without the full result
    response = {
        'id': job['id'],
        'filename': job['filename'],
        'status': job['status'],
        'progress': job['progress'],
        'created_at': job['created_at']
    }
    
    if job['status'] == 'completed':
        response['completed_at'] = job.get('completed_at')
        response['processing_time'] = job.get('processing_time')
        response['result_url'] = f'/api/async/result/{job_id}'
    elif job['status'] == 'failed':
        response['error'] = job.get('error')
        response['completed_at'] = job.get('completed_at')
    
    return jsonify(response)


@async_file_bp.route('/async/result/<job_id>', methods=['GET'])
def get_job_result(job_id: str):
    """Get the result of a completed async processing job."""
    with jobs_lock:
        job = jobs.get(job_id)
    
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    if job['status'] != 'completed':
        return jsonify({
            'error': 'Job not completed',
            'status': job['status']
        }), 400
    
    return jsonify({
        'job_id': job_id,
        'filename': job['filename'],
        'result': job.get('result'),
        'processing_time': job.get('processing_time'),
        'completed_at': job.get('completed_at')
    })


@async_file_bp.route('/async/stream/<job_id>', methods=['GET'])
def stream_job_progress(job_id: str):
    """Stream job progress updates via Server-Sent Events."""
    def generate():
        last_progress = -1
        
        while True:
            with jobs_lock:
                job = jobs.get(job_id)
            
            if not job:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                break
            
            # Send update if progress changed
            if job['progress'] != last_progress:
                last_progress = job['progress']
                yield f"data: {json.dumps({
                    'progress': job['progress'],
                    'status': job['status'],
                    'updated_at': job.get('updated_at')
                })}\n\n"
            
            # Check if job is done
            if job['status'] in ['completed', 'failed']:
                yield f"data: {json.dumps({
                    'status': job['status'],
                    'completed': True
                })}\n\n"
                break
            
            time.sleep(0.5)  # Poll every 500ms
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@async_file_bp.route('/async/cancel/<job_id>', methods=['POST'])
@rate_limit(20, 60)
def cancel_job(job_id: str):
    """Cancel a running job."""
    with jobs_lock:
        job = jobs.get(job_id)
        
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        if job['status'] in ['completed', 'failed']:
            return jsonify({'error': 'Job already finished'}), 400
        
        # Mark as cancelled
        job['status'] = 'cancelled'
        job['completed_at'] = datetime.now().isoformat()
    
    return jsonify({
        'job_id': job_id,
        'status': 'cancelled',
        'message': 'Job cancelled successfully'
    })