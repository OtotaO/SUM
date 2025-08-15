"""
Mass Document Processing API

Handles unlimited document uploads and processing.
The endpoint that makes SUM unbeatable.
"""

import os
import logging
import uuid
from pathlib import Path
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
import threading
import queue
import json
from typing import Dict, Any, List

from mass_document_engine import MassDocumentEngine, StreamingMassProcessor
from config import active_config

logger = logging.getLogger(__name__)
mass_bp = Blueprint('mass_processing', __name__)

# Global processing queue and status tracker
processing_queue = queue.Queue()
processing_status = {}
processing_lock = threading.Lock()

# Background processor thread
processor_thread = None
mass_engine = MassDocumentEngine()


def background_processor():
    """Background thread that processes document batches."""
    while True:
        try:
            job = processing_queue.get(timeout=1)
            if job is None:  # Shutdown signal
                break
                
            job_id = job['id']
            file_paths = job['files']
            
            # Update status
            with processing_lock:
                processing_status[job_id]['status'] = 'processing'
                processing_status[job_id]['message'] = f'Processing {len(file_paths)} documents...'
            
            # Process documents
            def progress_callback(progress):
                with processing_lock:
                    processing_status[job_id]['progress'] = progress
                    processing_status[job_id]['message'] = f"Processing {progress['current_file']}..."
            
            try:
                result = mass_engine.process_document_collection(
                    file_paths,
                    output_dir=f"jobs/{job_id}",
                    progress_callback=progress_callback
                )
                
                # Update status with completion
                with processing_lock:
                    processing_status[job_id]['status'] = 'completed'
                    processing_status[job_id]['result'] = {
                        'total_documents': result.total_documents,
                        'total_words': result.total_words,
                        'processing_time': result.processing_time,
                        'key_themes': result.key_themes[:10],
                        'executive_summary': result.executive_summary
                    }
                    processing_status[job_id]['message'] = 'Processing complete!'
                    
            except Exception as e:
                logger.error(f"Error processing job {job_id}: {e}")
                with processing_lock:
                    processing_status[job_id]['status'] = 'failed'
                    processing_status[job_id]['error'] = str(e)
                    processing_status[job_id]['message'] = f'Processing failed: {str(e)}'
                    
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Background processor error: {e}")


@mass_bp.route('/api/mass/upload', methods=['POST'])
def upload_documents():
    """
    Upload multiple documents for mass processing.
    
    Accepts up to 10,000 files in a single request.
    """
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    if not files or len(files) == 0:
        return jsonify({'error': 'No files selected'}), 400
    
    # Create job ID
    job_id = str(uuid.uuid4())
    job_dir = Path(f"uploads/mass/{job_id}")
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded files
    saved_files = []
    for file in files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = job_dir / filename
            file.save(str(filepath))
            saved_files.append(str(filepath))
    
    if not saved_files:
        return jsonify({'error': 'No valid files uploaded'}), 400
    
    # Initialize job status
    with processing_lock:
        processing_status[job_id] = {
            'id': job_id,
            'status': 'queued',
            'files': len(saved_files),
            'progress': {'completed': 0, 'total': len(saved_files)},
            'message': 'Waiting in queue...'
        }
    
    # Add to processing queue
    processing_queue.put({
        'id': job_id,
        'files': saved_files
    })
    
    return jsonify({
        'job_id': job_id,
        'files_uploaded': len(saved_files),
        'status': 'queued',
        'message': f'{len(saved_files)} files queued for processing'
    })


@mass_bp.route('/api/mass/process/folder', methods=['POST'])
def process_folder():
    """
    Process all documents in a server-side folder.
    
    For when you have documents already on the server.
    """
    data = request.get_json()
    folder_path = data.get('folder_path')
    
    if not folder_path:
        return jsonify({'error': 'No folder path provided'}), 400
    
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return jsonify({'error': 'Invalid folder path'}), 400
    
    # Find all documents
    extensions = ['.pdf', '.docx', '.txt', '.md', '.html', '.rtf']
    files = []
    for ext in extensions:
        files.extend(folder.glob(f'**/*{ext}'))
    
    if not files:
        return jsonify({'error': 'No documents found in folder'}), 404
    
    # Create job
    job_id = str(uuid.uuid4())
    file_paths = [str(f) for f in files]
    
    # Initialize job status
    with processing_lock:
        processing_status[job_id] = {
            'id': job_id,
            'status': 'queued',
            'files': len(file_paths),
            'progress': {'completed': 0, 'total': len(file_paths)},
            'message': 'Waiting in queue...'
        }
    
    # Add to processing queue
    processing_queue.put({
        'id': job_id,
        'files': file_paths
    })
    
    return jsonify({
        'job_id': job_id,
        'files_found': len(file_paths),
        'status': 'queued',
        'message': f'{len(file_paths)} files queued for processing'
    })


@mass_bp.route('/api/mass/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get the status of a mass processing job."""
    with processing_lock:
        if job_id not in processing_status:
            return jsonify({'error': 'Job not found'}), 404
        
        status = processing_status[job_id].copy()
    
    return jsonify(status)


@mass_bp.route('/api/mass/results/<job_id>', methods=['GET'])
def get_job_results(job_id):
    """
    Get the complete results of a processing job.
    
    Returns detailed summaries and analysis.
    """
    with processing_lock:
        if job_id not in processing_status:
            return jsonify({'error': 'Job not found'}), 404
        
        status = processing_status[job_id]
        if status['status'] != 'completed':
            return jsonify({'error': 'Job not completed', 'status': status['status']}), 400
    
    # Load complete results
    results_dir = Path(f"jobs/{job_id}")
    
    # Load collection summary
    summary_file = results_dir / "collection_summary.json"
    if not summary_file.exists():
        return jsonify({'error': 'Results not found'}), 404
    
    with open(summary_file, 'r') as f:
        collection_summary = json.load(f)
    
    # Load document index
    docs_index_file = results_dir / "documents" / "index.json"
    with open(docs_index_file, 'r') as f:
        docs_index = json.load(f)
    
    return jsonify({
        'job_id': job_id,
        'summary': collection_summary,
        'documents': docs_index,
        'download_url': f'/api/mass/download/{job_id}'
    })


@mass_bp.route('/api/mass/download/<job_id>', methods=['GET'])
def download_results(job_id):
    """Download complete results as a ZIP file."""
    import zipfile
    import io
    
    results_dir = Path(f"jobs/{job_id}")
    if not results_dir.exists():
        return jsonify({'error': 'Results not found'}), 404
    
    # Create ZIP file in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add all result files
        for file_path in results_dir.rglob('*'):
            if file_path.is_file():
                arc_name = str(file_path.relative_to(results_dir))
                zip_file.write(file_path, arc_name)
    
    zip_buffer.seek(0)
    
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'sum_results_{job_id}.zip'
    )


@mass_bp.route('/api/mass/stream', methods=['POST'])
def stream_process():
    """
    Stream processing endpoint for unlimited documents.
    
    Processes documents in batches and streams results.
    """
    data = request.get_json()
    folder_path = data.get('folder_path')
    batch_size = data.get('batch_size', 100)
    
    if not folder_path:
        return jsonify({'error': 'No folder path provided'}), 400
    
    folder = Path(folder_path)
    if not folder.exists():
        return jsonify({'error': 'Folder not found'}), 404
    
    # Find all documents
    extensions = ['.pdf', '.docx', '.txt', '.md', '.html', '.rtf']
    files = []
    for ext in extensions:
        files.extend(folder.glob(f'**/*{ext}'))
    
    def generate():
        """Generate streaming results."""
        streaming_processor = StreamingMassProcessor(mass_engine)
        
        # Process in batches
        for i, batch_summary in enumerate(streaming_processor.process_document_stream(
            (str(f) for f in files), batch_size=batch_size
        )):
            yield json.dumps({
                'batch': i + 1,
                'documents_in_batch': batch_summary.total_documents,
                'words_in_batch': batch_summary.total_words,
                'key_themes': batch_summary.key_themes[:5],
                'executive_summary': batch_summary.executive_summary[:500] + '...'
            }) + '\n'
    
    return generate(), {'Content-Type': 'application/x-ndjson'}


@mass_bp.route('/api/mass/stats', methods=['GET'])
def get_processing_stats():
    """Get overall processing statistics."""
    with processing_lock:
        active_jobs = sum(1 for s in processing_status.values() 
                         if s['status'] == 'processing')
        queued_jobs = sum(1 for s in processing_status.values() 
                         if s['status'] == 'queued')
        completed_jobs = sum(1 for s in processing_status.values() 
                           if s['status'] == 'completed')
    
    return jsonify({
        'engine_stats': mass_engine.stats,
        'queue_size': processing_queue.qsize(),
        'active_jobs': active_jobs,
        'queued_jobs': queued_jobs,
        'completed_jobs': completed_jobs,
        'max_workers': mass_engine.max_workers
    })


# Start background processor on import
def start_background_processor():
    global processor_thread
    if processor_thread is None or not processor_thread.is_alive():
        processor_thread = threading.Thread(target=background_processor, daemon=True)
        processor_thread.start()
        logger.info("Started mass document background processor")

start_background_processor()