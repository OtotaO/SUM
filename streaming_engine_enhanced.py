"""
streaming_engine_enhanced.py - Enhanced Streaming Engine with Proper Resource Management

Major improvements:
- Proper ThreadPoolExecutor lifecycle management with context managers
- Graceful shutdown handling
- Resource cleanup on errors
- Timeout handling for all operations
- Better memory management
- Thread-safe operations

Author: SUM Development Team (Enhanced)
License: Apache License 2.0
"""

import time
import json
import hashlib
import logging
import numpy as np
import threading
from typing import Dict, List, Any, Optional, Tuple, Iterator, Generator, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from contextlib import contextmanager
import psutil
import gc
import signal
import atexit
import weakref
from functools import wraps

# Import our existing components
from summarization_engine import HierarchicalDensificationEngine
from nltk.tokenize import sent_tokenize, word_tokenize

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def timeout_handler(seconds: int):
    """Decorator to add timeout to functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a future for timeout handling
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except concurrent.futures.TimeoutError:
                    logger.error(f"{func.__name__} timed out after {seconds} seconds")
                    raise TimeoutError(f"Operation timed out after {seconds} seconds")
        return wrapper
    return decorator


@dataclass
class ChunkMetadata:
    """Metadata for processed text chunks."""
    chunk_id: str
    start_position: int
    end_position: int
    semantic_boundaries: List[int]
    topic_coherence_score: float
    processing_time: float
    memory_usage: int
    cross_references: List[str] = field(default_factory=list)
    error_count: int = 0
    retry_count: int = 0


@dataclass
class StreamingConfig:
    """Configuration for streaming text processing."""
    chunk_size_words: int = 1000
    overlap_ratio: float = 0.15
    max_memory_mb: int = 512
    enable_progressive_refinement: bool = True
    semantic_coherence_threshold: float = 0.7
    max_concurrent_chunks: int = 4
    cache_processed_chunks: bool = True
    operation_timeout: int = 300  # 5 minutes
    retry_on_error: bool = True
    max_retries: int = 3


class MemoryMonitor:
    """Monitor and manage memory usage during processing."""
    
    def __init__(self, max_memory_mb: int = 512):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.process = psutil.Process()
        self._lock = threading.Lock()
        
    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        with self._lock:
            return self.process.memory_info().rss
        
    def is_memory_available(self, required_mb: int = 50) -> bool:
        """Check if enough memory is available."""
        current_usage = self.get_memory_usage()
        required_bytes = required_mb * 1024 * 1024
        return (current_usage + required_bytes) < self.max_memory_bytes
        
    def trigger_cleanup(self):
        """Trigger garbage collection if memory usage is high."""
        with self._lock:
            if self.get_memory_usage() > (0.8 * self.max_memory_bytes):
                gc.collect()
                logger.info("Triggered garbage collection due to high memory usage")


class ResourceManager:
    """Manage thread pool and other resources with proper cleanup."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._executor: Optional[ThreadPoolExecutor] = None
        self._lock = threading.Lock()
        self._shutdown = False
        self._active_futures = weakref.WeakSet()
        
        # Register cleanup handlers
        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals gracefully."""
        logger.info(f"Received signal {signum}, initiating cleanup...")
        self.cleanup()
    
    @contextmanager
    def get_executor(self) -> ThreadPoolExecutor:
        """Get thread pool executor with automatic cleanup."""
        with self._lock:
            if self._shutdown:
                raise RuntimeError("ResourceManager has been shut down")
            
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.max_workers,
                    thread_name_prefix="SUM-Worker"
                )
                logger.info(f"Created ThreadPoolExecutor with {self.max_workers} workers")
        
        try:
            yield self._executor
        except Exception as e:
            logger.error(f"Error in executor context: {e}")
            raise
    
    def submit_task(self, fn, *args, **kwargs):
        """Submit task with tracking."""
        with self.get_executor() as executor:
            future = executor.submit(fn, *args, **kwargs)
            self._active_futures.add(future)
            return future
    
    def cleanup(self):
        """Clean up all resources properly."""
        with self._lock:
            if self._shutdown:
                return
            
            self._shutdown = True
            
            if self._executor:
                # Cancel pending futures
                for future in self._active_futures:
                    if not future.done():
                        future.cancel()
                
                # Shutdown executor
                self._executor.shutdown(wait=True, cancel_futures=True)
                self._executor = None
                logger.info("ThreadPoolExecutor shut down successfully")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class EnhancedStreamingEngine:
    """
    Enhanced streaming text processing engine with proper resource management.
    
    Key improvements:
    - Proper thread pool lifecycle management
    - Graceful error handling and recovery
    - Memory-aware processing
    - Timeout handling for all operations
    """
    
    def __init__(self, config: Optional[StreamingConfig] = None):
        self.config = config or StreamingConfig()
        self.memory_monitor = MemoryMonitor(self.config.max_memory_mb)
        self.resource_manager = ResourceManager(self.config.max_concurrent_chunks)
        
        # Initialize processing components
        self.hierarchical_engine = HierarchicalDensificationEngine()
        
        # Processing state with thread safety
        self._lock = threading.Lock()
        self.processing_cache: Dict[str, Any] = {}
        self.chunk_metadata: Dict[str, ChunkMetadata] = {}
        self.processing_errors: List[Dict[str, Any]] = []
        
        # Statistics
        self.stats = {
            'chunks_processed': 0,
            'total_processing_time': 0,
            'errors_encountered': 0,
            'retries_performed': 0,
            'cache_hits': 0
        }
    
    @timeout_handler(60)  # 1 minute timeout for chunk creation
    def create_semantic_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create semantically coherent chunks with proper error handling."""
        if not text or not text.strip():
            return []
        
        chunks = []
        sentences = sent_tokenize(text)
        
        if not sentences:
            return []
        
        current_chunk = []
        current_words = 0
        chunk_id = 0
        
        for i, sentence in enumerate(sentences):
            words = word_tokenize(sentence)
            word_count = len(words)
            
            # Check memory before processing
            if not self.memory_monitor.is_memory_available():
                self.memory_monitor.trigger_cleanup()
                if not self.memory_monitor.is_memory_available():
                    logger.warning("Memory limit reached, returning partial chunks")
                    break
            
            # Check if adding this sentence exceeds chunk size
            if current_words + word_count > self.config.chunk_size_words and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
                
                chunks.append({
                    'chunk_id': f"chunk_{chunk_id}_{chunk_hash}",
                    'content': chunk_text,
                    'sentences': current_chunk.copy(),
                    'word_count': current_words,
                    'start_sentence': chunk_id * self.config.chunk_size_words,
                    'semantic_boundary': True
                })
                
                # Handle overlap
                overlap_sentences = self._calculate_overlap(current_chunk)
                current_chunk = overlap_sentences
                current_words = sum(len(word_tokenize(s)) for s in current_chunk)
                chunk_id += 1
            
            current_chunk.append(sentence)
            current_words += word_count
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
            
            chunks.append({
                'chunk_id': f"chunk_{chunk_id}_{chunk_hash}",
                'content': chunk_text,
                'sentences': current_chunk,
                'word_count': current_words,
                'start_sentence': chunk_id * self.config.chunk_size_words,
                'semantic_boundary': True
            })
        
        logger.info(f"Created {len(chunks)} semantic chunks from text")
        return chunks
    
    def _calculate_overlap(self, sentences: List[str]) -> List[str]:
        """Calculate overlap sentences for context preservation."""
        if not sentences:
            return []
        
        overlap_count = max(1, int(self.config.overlap_ratio * len(sentences)))
        return sentences[-overlap_count:]
    
    @timeout_handler(300)  # 5 minute timeout for chunk processing
    def process_chunk_with_retry(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Process a chunk with retry logic and error handling."""
        chunk_id = chunk['chunk_id']
        
        for attempt in range(self.config.max_retries):
            try:
                result = self._process_chunk_internal(chunk)
                return result
            except Exception as e:
                with self._lock:
                    self.stats['errors_encountered'] += 1
                    if attempt < self.config.max_retries - 1:
                        self.stats['retries_performed'] += 1
                
                logger.error(f"Error processing chunk {chunk_id} (attempt {attempt + 1}): {e}")
                
                if attempt == self.config.max_retries - 1:
                    # Final attempt failed
                    error_info = {
                        'chunk_id': chunk_id,
                        'error': str(e),
                        'timestamp': time.time(),
                        'attempts': attempt + 1
                    }
                    
                    with self._lock:
                        self.processing_errors.append(error_info)
                    
                    # Return error result
                    return {
                        'chunk_id': chunk_id,
                        'error': True,
                        'error_message': str(e),
                        'summary': f"Error processing chunk: {str(e)[:100]}",
                        'insights': []
                    }
                
                # Wait before retry with exponential backoff
                time.sleep(2 ** attempt)
    
    def _process_chunk_internal(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Internal chunk processing with caching."""
        chunk_id = chunk['chunk_id']
        
        # Check cache first
        if self.config.cache_processed_chunks:
            with self._lock:
                if chunk_id in self.processing_cache:
                    self.stats['cache_hits'] += 1
                    logger.info(f"Using cached result for chunk {chunk_id}")
                    return self.processing_cache[chunk_id]
        
        start_time = time.time()
        
        # Process through hierarchical engine
        processing_config = {
            'max_words': min(200, chunk['word_count'] // 5),
            'enable_insights': True,
            'insight_depth': 'balanced'
        }
        
        result = self.hierarchical_engine.process_text(chunk['content'], processing_config)
        
        # Enrich result
        enriched_result = {
            'chunk_id': chunk_id,
            'summary': result.get('summary', ''),
            'insights': result.get('insights', []),
            'processing_time': time.time() - start_time,
            'word_count': chunk['word_count'],
            'compression_ratio': len(result.get('summary', '').split()) / max(1, chunk['word_count'])
        }
        
        # Update stats and cache
        with self._lock:
            self.stats['chunks_processed'] += 1
            self.stats['total_processing_time'] += enriched_result['processing_time']
            
            if self.config.cache_processed_chunks:
                self.processing_cache[chunk_id] = enriched_result
        
        logger.info(f"Processed chunk {chunk_id} in {enriched_result['processing_time']:.2f}s")
        return enriched_result
    
    def process_text_streaming(self, text: str) -> Generator[Dict[str, Any], None, None]:
        """
        Process text in streaming fashion with proper resource management.
        
        Yields results as they become available.
        """
        try:
            # Create chunks
            chunks = self.create_semantic_chunks(text)
            
            if not chunks:
                yield {
                    'type': 'error',
                    'message': 'No chunks created from text'
                }
                return
            
            # Process chunks concurrently
            with self.resource_manager.get_executor() as executor:
                # Submit all chunks for processing
                future_to_chunk = {
                    executor.submit(self.process_chunk_with_retry, chunk): chunk
                    for chunk in chunks
                }
                
                # Yield results as they complete
                for future in as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    
                    try:
                        result = future.result(timeout=self.config.operation_timeout)
                        yield {
                            'type': 'chunk_result',
                            'data': result
                        }
                    except TimeoutError:
                        logger.error(f"Timeout processing chunk {chunk['chunk_id']}")
                        yield {
                            'type': 'chunk_error',
                            'chunk_id': chunk['chunk_id'],
                            'error': 'Processing timeout'
                        }
                    except Exception as e:
                        logger.error(f"Error getting result for chunk {chunk['chunk_id']}: {e}")
                        yield {
                            'type': 'chunk_error',
                            'chunk_id': chunk['chunk_id'],
                            'error': str(e)
                        }
            
            # Yield final statistics
            yield {
                'type': 'processing_complete',
                'stats': self.get_stats()
            }
            
        except Exception as e:
            logger.error(f"Critical error in streaming processing: {e}")
            yield {
                'type': 'critical_error',
                'error': str(e)
            }
    
    def process_text_batch(self, text: str) -> Dict[str, Any]:
        """Process text in batch mode with aggregated results."""
        results = []
        errors = []
        
        for item in self.process_text_streaming(text):
            if item['type'] == 'chunk_result':
                results.append(item['data'])
            elif item['type'] in ['chunk_error', 'critical_error']:
                errors.append(item)
        
        # Aggregate results
        if not results:
            return {
                'error': True,
                'message': 'No results produced',
                'errors': errors
            }
        
        # Combine summaries
        combined_summary = ' '.join([r['summary'] for r in results if 'summary' in r])
        all_insights = []
        
        for r in results:
            if 'insights' in r:
                all_insights.extend(r['insights'])
        
        return {
            'summary': combined_summary,
            'insights': all_insights,
            'chunks_processed': len(results),
            'errors': errors,
            'stats': self.get_stats()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        with self._lock:
            stats = self.stats.copy()
            stats['cache_size'] = len(self.processing_cache)
            stats['memory_usage_mb'] = self.memory_monitor.get_memory_usage() / (1024 * 1024)
            stats['error_count'] = len(self.processing_errors)
            
            if stats['chunks_processed'] > 0:
                stats['avg_processing_time'] = stats['total_processing_time'] / stats['chunks_processed']
                stats['cache_hit_rate'] = stats['cache_hits'] / stats['chunks_processed']
            else:
                stats['avg_processing_time'] = 0
                stats['cache_hit_rate'] = 0
        
        return stats
    
    def clear_cache(self):
        """Clear processing cache and trigger garbage collection."""
        with self._lock:
            self.processing_cache.clear()
            self.chunk_metadata.clear()
        
        self.memory_monitor.trigger_cleanup()
        logger.info("Cleared processing cache and triggered garbage collection")
    
    def cleanup(self):
        """Clean up all resources."""
        self.resource_manager.cleanup()
        self.clear_cache()
        logger.info("Enhanced streaming engine cleaned up successfully")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# Example usage with proper resource management
if __name__ == "__main__":
    print("Testing Enhanced Streaming Engine")
    print("=" * 50)
    
    # Sample text
    sample_text = """
    Artificial Intelligence has transformed how we process information. 
    Machine learning algorithms can now understand context and meaning in ways 
    that were impossible just a few years ago. This has led to breakthroughs 
    in natural language processing, computer vision, and decision-making systems.
    
    The implications are profound. We're seeing AI systems that can diagnose 
    diseases, translate languages in real-time, and even create art. However, 
    with these capabilities come responsibilities. We must ensure that AI 
    systems are fair, transparent, and aligned with human values.
    
    The future of AI is bright but requires careful consideration of ethical 
    implications and societal impact. As we continue to develop more powerful 
    systems, we must also develop frameworks for their responsible use.
    """ * 10  # Make it longer for testing
    
    # Test with context manager for automatic cleanup
    config = StreamingConfig(
        chunk_size_words=100,
        max_concurrent_chunks=2,
        operation_timeout=60
    )
    
    print("\nTesting streaming processing:")
    with EnhancedStreamingEngine(config) as engine:
        for result in engine.process_text_streaming(sample_text):
            print(f"Result type: {result['type']}")
            if result['type'] == 'chunk_result':
                print(f"  Chunk ID: {result['data']['chunk_id']}")
                print(f"  Processing time: {result['data'].get('processing_time', 0):.2f}s")
            elif result['type'] == 'processing_complete':
                print(f"  Stats: {result['stats']}")
    
    print("\nTesting batch processing:")
    with EnhancedStreamingEngine(config) as engine:
        batch_result = engine.process_text_batch(sample_text)
        print(f"Chunks processed: {batch_result.get('chunks_processed', 0)}")
        print(f"Errors encountered: {len(batch_result.get('errors', []))}")
        print(f"Summary length: {len(batch_result.get('summary', '').split())} words")
    
    print("\nEnhanced streaming engine test complete!")