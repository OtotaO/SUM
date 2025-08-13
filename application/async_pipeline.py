"""
async_pipeline.py - Asynchronous Processing Pipeline

This module implements high-performance async processing:
- Concurrent document processing
- Stream-based large file handling
- Non-blocking API operations
- Memory-efficient chunking

Author: SUM Development Team
License: Apache License 2.0
"""

import asyncio
import aiofiles
import time
import logging
from typing import List, Dict, Any, Optional, AsyncIterator, Callable
from dataclasses import dataclass
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json

# Try to import async libraries
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    UVLOOP_AVAILABLE = True
except ImportError:
    UVLOOP_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ProcessingTask:
    """Represents a processing task in the pipeline."""
    id: str
    content: str
    task_type: str
    metadata: Dict[str, Any]
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class AsyncProcessingPipeline:
    """
    High-performance asynchronous processing pipeline.
    Handles concurrent processing with memory efficiency.
    """
    
    def __init__(self, 
                 max_concurrent_tasks: int = 10,
                 chunk_size: int = 1024 * 1024,  # 1MB chunks
                 use_thread_pool: bool = True):
        """
        Initialize the async pipeline.
        
        Args:
            max_concurrent_tasks: Maximum concurrent processing tasks
            chunk_size: Size of chunks for streaming
            use_thread_pool: Whether to use thread pool for CPU-bound tasks
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.chunk_size = chunk_size
        self.use_thread_pool = use_thread_pool
        
        # Task management
        self.active_tasks: Dict[str, ProcessingTask] = {}
        self.task_queue: asyncio.Queue = None
        self.results_queue: asyncio.Queue = None
        
        # Thread pool for CPU-bound operations
        if use_thread_pool:
            self.thread_pool = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        else:
            self.thread_pool = None
        
        # Statistics
        self.stats = {
            'tasks_processed': 0,
            'tasks_failed': 0,
            'total_processing_time': 0.0,
            'average_task_time': 0.0
        }
        
        # Processing functions registry
        self.processors: Dict[str, Callable] = {}
        self._register_default_processors()
    
    def _register_default_processors(self):
        """Register default processing functions."""
        from application.optimized_summarizer import summarize_text_universal
        from memory.semantic_memory import get_semantic_memory_engine
        from memory.knowledge_graph import get_knowledge_graph_engine
        
        self.processors['summarize'] = self._process_summarization
        self.processors['extract_entities'] = self._process_entity_extraction
        self.processors['generate_embedding'] = self._process_embedding_generation
        self.processors['store_memory'] = self._process_memory_storage
    
    async def process_documents_stream(self, 
                                     document_stream: AsyncIterator[Dict[str, Any]],
                                     task_type: str = "summarize") -> AsyncIterator[ProcessingTask]:
        """
        Process a stream of documents asynchronously.
        
        Args:
            document_stream: Async iterator of documents
            task_type: Type of processing task
            
        Yields:
            Completed ProcessingTask objects
        """
        # Initialize queues
        self.task_queue = asyncio.Queue(maxsize=self.max_concurrent_tasks * 2)
        self.results_queue = asyncio.Queue()
        
        # Start worker tasks
        workers = []
        for i in range(self.max_concurrent_tasks):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            workers.append(worker)
        
        # Start result collector
        collector = asyncio.create_task(self._result_collector())
        
        try:
            # Feed documents to queue
            async for document in document_stream:
                task = ProcessingTask(
                    id=self._generate_task_id(document),
                    content=document.get('content', ''),
                    task_type=task_type,
                    metadata=document.get('metadata', {})
                )
                
                await self.task_queue.put(task)
                self.active_tasks[task.id] = task
            
            # Signal end of stream
            for _ in range(self.max_concurrent_tasks):
                await self.task_queue.put(None)
            
            # Wait for workers to complete
            await asyncio.gather(*workers)
            
            # Signal collector to stop
            await self.results_queue.put(None)
            await collector
            
            # Yield all results
            for task_id, task in self.active_tasks.items():
                if task.status == "completed":
                    yield task
                    
        finally:
            # Cleanup
            for worker in workers:
                if not worker.done():
                    worker.cancel()
            
            if not collector.done():
                collector.cancel()
    
    async def process_file_stream(self, 
                                file_path: str,
                                task_type: str = "summarize") -> AsyncIterator[Dict[str, Any]]:
        """
        Process a large file in streaming chunks.
        
        Args:
            file_path: Path to file
            task_type: Type of processing
            
        Yields:
            Processing results for each chunk
        """
        chunk_number = 0
        
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
            while True:
                chunk = await file.read(self.chunk_size)
                if not chunk:
                    break
                
                # Create task for chunk
                task = ProcessingTask(
                    id=f"{file_path}-chunk-{chunk_number}",
                    content=chunk,
                    task_type=task_type,
                    metadata={
                        'file_path': file_path,
                        'chunk_number': chunk_number,
                        'chunk_size': len(chunk)
                    }
                )
                
                # Process chunk
                result = await self._process_task(task)
                
                yield {
                    'chunk_number': chunk_number,
                    'result': result.result if result.status == "completed" else None,
                    'error': result.error,
                    'metadata': result.metadata
                }
                
                chunk_number += 1
    
    async def batch_process(self, 
                          items: List[Dict[str, Any]],
                          task_type: str = "summarize",
                          progress_callback: Optional[Callable] = None) -> List[ProcessingTask]:
        """
        Process a batch of items concurrently.
        
        Args:
            items: List of items to process
            task_type: Type of processing
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of completed ProcessingTask objects
        """
        tasks = []
        
        # Create processing tasks
        for i, item in enumerate(items):
            task = ProcessingTask(
                id=self._generate_task_id(item),
                content=item.get('content', ''),
                task_type=task_type,
                metadata=item.get('metadata', {})
            )
            tasks.append(task)
        
        # Process in batches
        results = []
        
        for i in range(0, len(tasks), self.max_concurrent_tasks):
            batch = tasks[i:i + self.max_concurrent_tasks]
            
            # Process batch concurrently
            batch_results = await asyncio.gather(
                *[self._process_task(task) for task in batch],
                return_exceptions=True
            )
            
            # Handle results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    batch[j].status = "failed"
                    batch[j].error = str(result)
                else:
                    results.append(result)
            
            # Progress callback
            if progress_callback:
                progress = (i + len(batch)) / len(tasks)
                await progress_callback(progress, i + len(batch), len(tasks))
        
        return results
    
    async def _worker(self, worker_id: str):
        """Worker coroutine for processing tasks."""
        while True:
            task = await self.task_queue.get()
            
            if task is None:
                break
            
            try:
                # Process task
                result = await self._process_task(task)
                
                # Put result in queue
                await self.results_queue.put(result)
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                task.status = "failed"
                task.error = str(e)
                await self.results_queue.put(task)
    
    async def _result_collector(self):
        """Collect and process results."""
        while True:
            result = await self.results_queue.get()
            
            if result is None:
                break
            
            # Update statistics
            if result.status == "completed":
                self.stats['tasks_processed'] += 1
                if result.start_time and result.end_time:
                    task_time = result.end_time - result.start_time
                    self.stats['total_processing_time'] += task_time
                    self.stats['average_task_time'] = (
                        self.stats['total_processing_time'] / 
                        self.stats['tasks_processed']
                    )
            else:
                self.stats['tasks_failed'] += 1
    
    async def _process_task(self, task: ProcessingTask) -> ProcessingTask:
        """Process a single task."""
        task.start_time = time.time()
        task.status = "processing"
        
        try:
            # Get processor function
            processor = self.processors.get(task.task_type)
            
            if not processor:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            # Process based on type
            if self.use_thread_pool and task.task_type in ['summarize', 'extract_entities']:
                # CPU-bound tasks in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.thread_pool,
                    processor,
                    task.content,
                    task.metadata
                )
            else:
                # I/O-bound tasks in event loop
                result = await processor(task.content, task.metadata)
            
            task.result = result
            task.status = "completed"
            
        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            task.status = "failed"
            task.error = str(e)
        
        finally:
            task.end_time = time.time()
        
        return task
    
    # Processing functions
    
    def _process_summarization(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process summarization task."""
        from application.optimized_summarizer import summarize_text_universal
        
        density = metadata.get('density', 'medium')
        result = summarize_text_universal(content)
        
        return {
            'summary': result.get(density, result.get('summary', '')),
            'all_densities': result,
            'metadata': metadata
        }
    
    def _process_entity_extraction(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process entity extraction task."""
        from memory.knowledge_graph import get_knowledge_graph_engine
        
        kg_engine = get_knowledge_graph_engine()
        result = kg_engine.extract_entities_and_relationships(
            content,
            source=metadata.get('source', 'async_pipeline')
        )
        
        return result
    
    async def _process_embedding_generation(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process embedding generation task."""
        from memory.semantic_memory import get_semantic_memory_engine
        
        memory_engine = get_semantic_memory_engine()
        
        # For async compatibility
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            memory_engine.generate_embedding,
            content
        )
        
        return {
            'embedding': embedding,
            'dimension': len(embedding),
            'metadata': metadata
        }
    
    async def _process_memory_storage(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process memory storage task."""
        from memory.semantic_memory import get_semantic_memory_engine
        
        memory_engine = get_semantic_memory_engine()
        
        # Extract summary if not provided
        summary = metadata.get('summary')
        if not summary:
            summary_result = self._process_summarization(content, metadata)
            summary = summary_result['summary']
        
        # Store in memory
        loop = asyncio.get_event_loop()
        memory_id = await loop.run_in_executor(
            None,
            memory_engine.store_memory,
            content,
            summary,
            metadata,
            metadata.get('relationships', [])
        )
        
        return {
            'memory_id': memory_id,
            'stored': True,
            'metadata': metadata
        }
    
    def _generate_task_id(self, item: Dict[str, Any]) -> str:
        """Generate unique task ID."""
        content = str(item.get('content', ''))[:100]
        timestamp = str(time.time())
        return hashlib.sha256(f"{content}{timestamp}".encode()).hexdigest()[:16]
    
    async def close(self):
        """Close the pipeline and cleanup resources."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        # Cancel any remaining tasks
        for task_id, task in self.active_tasks.items():
            if task.status == "processing":
                task.status = "cancelled"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return dict(self.stats)


# Convenience functions for async operations

async def process_files_async(file_paths: List[str], 
                            task_type: str = "summarize",
                            max_concurrent: int = 5) -> List[Dict[str, Any]]:
    """
    Process multiple files asynchronously.
    
    Args:
        file_paths: List of file paths
        task_type: Type of processing
        max_concurrent: Maximum concurrent operations
        
    Returns:
        List of processing results
    """
    pipeline = AsyncProcessingPipeline(max_concurrent_tasks=max_concurrent)
    
    try:
        # Create document stream from files
        async def file_stream():
            for file_path in file_paths:
                try:
                    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        yield {
                            'content': content,
                            'metadata': {'file_path': file_path}
                        }
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")
        
        # Process stream
        results = []
        async for task in pipeline.process_documents_stream(file_stream(), task_type):
            results.append({
                'file_path': task.metadata.get('file_path'),
                'result': task.result,
                'status': task.status,
                'error': task.error
            })
        
        return results
        
    finally:
        await pipeline.close()


async def stream_large_file_processing(file_path: str,
                                     chunk_callback: Callable,
                                     task_type: str = "summarize") -> Dict[str, Any]:
    """
    Stream process a large file with progress callbacks.
    
    Args:
        file_path: Path to large file
        chunk_callback: Callback for each chunk result
        task_type: Type of processing
        
    Returns:
        Final processing summary
    """
    pipeline = AsyncProcessingPipeline()
    
    try:
        chunks_processed = 0
        all_results = []
        
        async for chunk_result in pipeline.process_file_stream(file_path, task_type):
            chunks_processed += 1
            all_results.append(chunk_result)
            
            # Call callback
            await chunk_callback(chunk_result, chunks_processed)
        
        return {
            'file_path': file_path,
            'chunks_processed': chunks_processed,
            'stats': pipeline.get_stats()
        }
        
    finally:
        await pipeline.close()


# Example usage
if __name__ == "__main__":
    async def main():
        # Example 1: Batch processing
        items = [
            {'content': 'Text 1 to summarize...'},
            {'content': 'Text 2 to summarize...'},
            {'content': 'Text 3 to summarize...'}
        ]
        
        pipeline = AsyncProcessingPipeline(max_concurrent_tasks=3)
        
        async def progress(p, current, total):
            print(f"Progress: {p:.1%} ({current}/{total})")
        
        results = await pipeline.batch_process(items, progress_callback=progress)
        
        for result in results:
            print(f"Task {result.id}: {result.status}")
        
        await pipeline.close()
    
    # Run example
    asyncio.run(main())