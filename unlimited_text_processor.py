"""
Unlimited Text Processor - Handle texts of ANY length efficiently

This module provides intelligent chunking and streaming capabilities
to process texts from 1 byte to 1 terabyte without memory overflow.

Key Features:
- Dynamic chunk sizing based on content
- Overlapping chunks for context preservation
- Streaming processing for massive texts
- Hierarchical summarization for long documents
- Memory-efficient processing

Author: SUM Team
License: Apache License 2.0
"""

import logging
import hashlib
from typing import Iterator, Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import io
import mmap
import os
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a chunk of text for processing."""
    chunk_id: int
    content: str
    start_pos: int
    end_pos: int
    overlap_start: int = 0
    overlap_end: int = 0
    

@dataclass
class ChunkSummary:
    """Summary of a text chunk."""
    chunk_id: int
    summary: str
    key_concepts: List[str]
    word_count: int
    compression_ratio: float


class UnlimitedTextProcessor:
    """
    Process texts of unlimited length through intelligent chunking.
    
    This processor can handle:
    - Small texts (< 10KB): Process directly
    - Medium texts (10KB - 10MB): Smart chunking
    - Large texts (10MB - 1GB): Streaming with memory mapping
    - Massive texts (> 1GB): Multi-level hierarchical processing
    """
    
    # Size thresholds
    SMALL_TEXT = 10 * 1024  # 10KB
    MEDIUM_TEXT = 10 * 1024 * 1024  # 10MB
    LARGE_TEXT = 1024 * 1024 * 1024  # 1GB
    
    # Chunk sizes for different text sizes
    SMALL_CHUNK = 5000  # ~1000 words
    MEDIUM_CHUNK = 25000  # ~5000 words
    LARGE_CHUNK = 100000  # ~20000 words
    
    def __init__(self, 
                 overlap_ratio: float = 0.1,
                 max_memory_usage: int = 512 * 1024 * 1024):  # 512MB
        """
        Initialize the unlimited text processor.
        
        Args:
            overlap_ratio: Percentage of chunk to overlap (0.1 = 10%)
            max_memory_usage: Maximum memory to use for processing
        """
        self.overlap_ratio = overlap_ratio
        self.max_memory_usage = max_memory_usage
        
        # Lazy import to avoid circular dependency
        from summarization_engine import HierarchicalDensificationEngine
        self.summarizer = HierarchicalDensificationEngine()
        
    def process_text(self, 
                     text_or_path: Any,
                     config: Optional[Dict[str, Any]] = None,
                     progress_callback=None) -> Dict[str, Any]:
        """
        Process text of any length intelligently.
        
        Args:
            text_or_path: String text, file path, or file-like object
            config: Processing configuration
            
        Returns:
            Dictionary with summary results
        """
        # Determine input type and size
        text_size, text_source = self._analyze_input(text_or_path)
        
        logger.info(f"Processing text of size: {text_size:,} bytes")
        
        # Route to appropriate processor based on size
        if text_size < self.SMALL_TEXT:
            return self._process_small_text(text_source, config, progress_callback)
        elif text_size < self.MEDIUM_TEXT:
            return self._process_medium_text(text_source, text_size, config, progress_callback)
        elif text_size < self.LARGE_TEXT:
            return self._process_large_text(text_source, text_size, config, progress_callback)
        else:
            return self._process_massive_text(text_source, text_size, config, progress_callback)
            

    def process_text_stream(self, text_or_path: Any, config: Optional[Dict[str, Any]] = None):
        import queue
        import threading

        q = queue.Queue()

        def callback(event):
            q.put(event)

        def worker():
            try:
                result = self.process_text(text_or_path, config, progress_callback=callback)
                q.put({'type': 'result', 'data': result})
            except Exception as e:
                q.put({'type': 'error', 'content': str(e)})
            finally:
                q.put(None)

        t = threading.Thread(target=worker)
        t.start()

        while True:
            item = q.get()
            if item is None:
                break
            yield item

    def _analyze_input(self, text_or_path: Any) -> Tuple[int, Any]:
        """Analyze input to determine size and type."""
        if isinstance(text_or_path, str):
            if os.path.exists(text_or_path):
                # File path
                size = os.path.getsize(text_or_path)
                return size, text_or_path
            else:
                # Direct text
                size = len(text_or_path.encode('utf-8'))
                return size, text_or_path
        elif hasattr(text_or_path, 'read'):
            # File-like object
            current_pos = text_or_path.tell()
            text_or_path.seek(0, 2)  # Seek to end
            size = text_or_path.tell()
            text_or_path.seek(current_pos)  # Reset position
            return size, text_or_path
        else:
            raise ValueError("Input must be string, file path, or file-like object")
            
    def _process_small_text(self, 
                           text_source: Any, 
                           config: Optional[Dict[str, Any]],
                           progress_callback=None) -> Dict[str, Any]:
        """Process small texts directly without chunking."""
        # Get text content
        if isinstance(text_source, str) and not os.path.exists(text_source):
            text = text_source
        else:
            text = self._read_text(text_source, limit=self.SMALL_TEXT)
            
        # Process with standard summarizer
        if progress_callback: progress_callback({'type': 'log', 'content': 'Processing small text directly...'})
        result = self.summarizer.process_text(text, config)
        result['processing_method'] = 'direct'
        result['chunks_processed'] = 1
        
        return result
        
    def _process_medium_text(self,
                            text_source: Any,
                            text_size: int,
                            config: Optional[Dict[str, Any]],
                            progress_callback=None) -> Dict[str, Any]:
        """Process medium texts with smart chunking."""
        # Determine optimal chunk size
        chunk_size = self._calculate_chunk_size(text_size)
        overlap_size = int(chunk_size * self.overlap_ratio)
        
        # Process chunks
        chunks = list(self._create_chunks(text_source, chunk_size, overlap_size))
        logger.info(f"Processing {len(chunks)} chunks of ~{chunk_size:,} chars each")
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            if progress_callback: progress_callback({'type': 'log', 'content': f'Processing chunk {i+1}/{len(chunks)}...'})
            summary = self._process_chunk(chunk, config)
            chunk_summaries.append(summary)
            
        # Combine chunk summaries
        combined_result = self._combine_chunk_summaries(chunk_summaries, config)
        combined_result['processing_method'] = 'chunked'
        combined_result['chunks_processed'] = len(chunks)
        combined_result['chunk_size'] = chunk_size
        
        return combined_result
        
    def _process_large_text(self,
                           text_source: Any,
                           text_size: int,
                           config: Optional[Dict[str, Any]],
                           progress_callback=None) -> Dict[str, Any]:
        """Process large texts with memory-mapped streaming."""
        # Use larger chunks for efficiency
        chunk_size = self.LARGE_CHUNK
        overlap_size = int(chunk_size * self.overlap_ratio)
        
        # Create temporary file if needed
        if isinstance(text_source, str) and not os.path.exists(text_source):
            # Write string to temp file for memory mapping
            with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
                tmp.write(text_source)
                temp_path = tmp.name
            text_source = temp_path
            cleanup_temp = True
        else:
            cleanup_temp = False
            
        try:
            # Process using memory mapping
            chunk_summaries = []
            
            with open(text_source, 'r+b') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped:
                    # Process chunks from memory map
                    for i, chunk in enumerate(self._create_mmap_chunks(mmapped, chunk_size, overlap_size)):
                        if progress_callback: progress_callback({'type': 'log', 'content': f'Processing large memory-mapped chunk {i+1}...'})
                        summary = self._process_chunk(chunk, config)
                        chunk_summaries.append(summary)
                        
                        # Free memory periodically
                        if len(chunk_summaries) % 10 == 0:
                            import gc
                            gc.collect()
                            
            # Hierarchical combination for large texts
            result = self._hierarchical_combine(chunk_summaries, config)
            result['processing_method'] = 'streaming'
            result['chunks_processed'] = len(chunk_summaries)
            result['chunk_size'] = chunk_size
            
            return result
            
        finally:
            if cleanup_temp and os.path.exists(temp_path):
                os.unlink(temp_path)
                
    def _process_massive_text(self,
                             text_source: Any,
                             text_size: int,
                             config: Optional[Dict[str, Any]],
                             progress_callback=None) -> Dict[str, Any]:
        """Process massive texts with multi-level hierarchical summarization."""
        # Use very large chunks
        chunk_size = self.LARGE_CHUNK * 2
        overlap_size = int(chunk_size * self.overlap_ratio)
        
        logger.info(f"Processing massive text ({text_size:,} bytes) with hierarchical approach")
        
        # First level: Process in batches
        batch_size = 50  # Process 50 chunks at a time
        batch_summaries = []
        
        chunk_generator = self._create_streaming_chunks(text_source, chunk_size, overlap_size)
        
        batch = []
        batch_idx = 1
        for chunk in chunk_generator:
            batch.append(chunk)
            
            if len(batch) >= batch_size:
                # Process batch
                if progress_callback: progress_callback({'type': 'log', 'content': f'Processing massive text batch {batch_idx}...'})
                batch_summary = self._process_batch(batch, config, progress_callback)
                batch_idx += 1
                batch_summaries.append(batch_summary)
                batch = []
                
                # Free memory
                import gc
                gc.collect()
                
        # Process remaining batch
        if batch:
            if progress_callback: progress_callback({'type': 'log', 'content': 'Processing final massive text batch...'})
            batch_summary = self._process_batch(batch, config, progress_callback)
            batch_summaries.append(batch_summary)
            
        # Second level: Combine batch summaries
        final_result = self._combine_batch_summaries(batch_summaries, config)
        final_result['processing_method'] = 'hierarchical_streaming'
        final_result['total_chunks'] = sum(bs['chunks'] for bs in batch_summaries)
        final_result['batches_processed'] = len(batch_summaries)
        
        return final_result
        
    def _calculate_chunk_size(self, text_size: int) -> int:
        """Calculate optimal chunk size based on text size."""
        if text_size < 100 * 1024:  # < 100KB
            return self.SMALL_CHUNK
        elif text_size < 1024 * 1024:  # < 1MB
            return self.MEDIUM_CHUNK
        else:
            return self.LARGE_CHUNK
            
    def _create_chunks(self, 
                      text_source: Any,
                      chunk_size: int,
                      overlap_size: int) -> Iterator[TextChunk]:
        """Create overlapping chunks from text source."""
        # Read text
        if isinstance(text_source, str) and not os.path.exists(text_source):
            text = text_source
        else:
            text = self._read_text(text_source)
            
        # Create chunks
        chunk_id = 0
        pos = 0
        text_len = len(text)
        
        while pos < text_len:
            # Calculate chunk boundaries
            chunk_start = max(0, pos - overlap_size if pos > 0 else 0)
            chunk_end = min(text_len, pos + chunk_size)
            
            # Extract chunk
            chunk_content = text[chunk_start:chunk_end]
            
            # Find sentence boundaries for clean cuts
            if chunk_end < text_len:
                # Try to end at sentence boundary
                last_period = chunk_content.rfind('. ')
                if last_period > chunk_size * 0.8:  # If we're at least 80% through
                    chunk_end = chunk_start + last_period + 1
                    chunk_content = text[chunk_start:chunk_end]
                    
            yield TextChunk(
                chunk_id=chunk_id,
                content=chunk_content,
                start_pos=pos,
                end_pos=chunk_end,
                overlap_start=chunk_start if pos > 0 else 0,
                overlap_end=min(overlap_size, chunk_end - pos)
            )
            
            chunk_id += 1
            pos = chunk_end - overlap_size
            
    def _create_mmap_chunks(self,
                           mmapped: mmap.mmap,
                           chunk_size: int,
                           overlap_size: int) -> Iterator[TextChunk]:
        """Create chunks from memory-mapped file."""
        chunk_id = 0
        pos = 0
        file_size = len(mmapped)
        
        while pos < file_size:
            # Calculate chunk boundaries
            chunk_start = max(0, pos - overlap_size if pos > 0 else 0)
            chunk_end = min(file_size, pos + chunk_size)
            
            # Read chunk
            mmapped.seek(chunk_start)
            chunk_bytes = mmapped.read(chunk_end - chunk_start)
            
            # Decode with error handling
            try:
                chunk_content = chunk_bytes.decode('utf-8', errors='replace')
            except:
                chunk_content = chunk_bytes.decode('latin-1', errors='replace')
                
            yield TextChunk(
                chunk_id=chunk_id,
                content=chunk_content,
                start_pos=pos,
                end_pos=chunk_end,
                overlap_start=chunk_start if pos > 0 else 0,
                overlap_end=min(overlap_size, chunk_end - pos)
            )
            
            chunk_id += 1
            pos = chunk_end - overlap_size
            
    def _create_streaming_chunks(self,
                                text_source: Any,
                                chunk_size: int,
                                overlap_size: int) -> Iterator[TextChunk]:
        """Create chunks using streaming for massive files."""
        if isinstance(text_source, str) and os.path.exists(text_source):
            file_path = text_source
        else:
            # Need a file path for streaming
            raise ValueError("Streaming requires a file path")
            
        chunk_id = 0
        overlap_buffer = ""
        
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            while True:
                # Read chunk
                chunk_content = overlap_buffer + f.read(chunk_size - len(overlap_buffer))
                
                if not chunk_content:
                    break
                    
                # Save overlap for next chunk
                if len(chunk_content) == chunk_size:
                    overlap_buffer = chunk_content[-overlap_size:]
                else:
                    overlap_buffer = ""
                    
                yield TextChunk(
                    chunk_id=chunk_id,
                    content=chunk_content,
                    start_pos=chunk_id * (chunk_size - overlap_size),
                    end_pos=(chunk_id + 1) * (chunk_size - overlap_size),
                    overlap_start=0 if chunk_id == 0 else overlap_size,
                    overlap_end=overlap_size if overlap_buffer else 0
                )
                
                chunk_id += 1
                
    def _process_chunk(self, 
                      chunk: TextChunk,
                      config: Optional[Dict[str, Any]]) -> ChunkSummary:
        """Process a single chunk."""
        # Adjust config for chunk processing
        chunk_config = config.copy() if config else {}
        chunk_config['max_summary_tokens'] = 200  # Consistent chunk summaries
        
        # Process chunk
        result = self.summarizer.process_text(chunk.content, chunk_config)
        
        # Extract summary
        if 'hierarchical_summary' in result:
            summary = result['hierarchical_summary'].get('level_3_expanded', 
                     result.get('summary', ''))
        else:
            summary = result.get('summary', '')
            
        return ChunkSummary(
            chunk_id=chunk.chunk_id,
            summary=summary,
            key_concepts=result.get('key_concepts', [])[:5],
            word_count=len(chunk.content.split()),
            compression_ratio=len(summary) / len(chunk.content) if chunk.content else 1.0
        )
        
    def _combine_chunk_summaries(self,
                                summaries: List[ChunkSummary],
                                config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multiple chunk summaries into final summary."""
        # Combine all chunk summaries
        combined_text = "\n\n".join(s.summary for s in summaries)
        
        # Extract all key concepts
        all_concepts = []
        for s in summaries:
            all_concepts.extend(s.key_concepts)
            
        # Count concept frequency
        from collections import Counter
        concept_freq = Counter(all_concepts)
        top_concepts = [c for c, _ in concept_freq.most_common(20)]
        
        # Create final summary from combined summaries
        final_config = config.copy() if config else {}
        final_config['max_summary_tokens'] = final_config.get('max_summary_tokens', 500)
        
        final_result = self.summarizer.process_text(combined_text, final_config)
        
        # Add chunk processing metadata
        final_result['key_concepts'] = top_concepts
        final_result['total_word_count'] = sum(s.word_count for s in summaries)
        final_result['chunk_summaries'] = [
            {
                'chunk_id': s.chunk_id,
                'summary': s.summary,
                'word_count': s.word_count
            }
            for s in summaries[:10]  # Include first 10 chunk summaries
        ]
        
        return final_result
        
    def _hierarchical_combine(self,
                             summaries: List[ChunkSummary],
                             config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Hierarchically combine many summaries."""
        # Group summaries into batches
        batch_size = 20
        batches = [summaries[i:i+batch_size] for i in range(0, len(summaries), batch_size)]
        
        # First level: Combine within batches
        batch_results = []
        for batch in batches:
            batch_result = self._combine_chunk_summaries(batch, config)
            batch_results.append(batch_result)
            
        # Second level: Combine batch results
        if len(batch_results) > 1:
            # Create meta-summaries from batch results
            meta_summaries = []
            for i, result in enumerate(batch_results):
                meta_summaries.append(ChunkSummary(
                    chunk_id=i,
                    summary=result.get('summary', ''),
                    key_concepts=result.get('key_concepts', [])[:5],
                    word_count=result.get('total_word_count', 0),
                    compression_ratio=1.0
                ))
                
            # Final combination
            final_result = self._combine_chunk_summaries(meta_summaries, config)
        else:
            final_result = batch_results[0]
            
        final_result['hierarchical_levels'] = 2
        return final_result
        
    def _process_batch(self,
                      chunks: List[TextChunk],
                      config: Optional[Dict[str, Any]],
                      progress_callback=None) -> Dict[str, Any]:
        """Process a batch of chunks."""
        summaries = []
        for i, chunk in enumerate(chunks):
            if progress_callback: progress_callback({'type': 'log', 'content': f'  -> Inner chunk {i+1}/{len(chunks)}...'})
            summary = self._process_chunk(chunk, config)
            summaries.append(summary)
            
        result = self._combine_chunk_summaries(summaries, config)
        result['chunks'] = len(chunks)
        return result
        
    def _combine_batch_summaries(self,
                                batch_summaries: List[Dict[str, Any]],
                                config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine summaries from multiple batches."""
        # Create chunk summaries from batch summaries
        meta_summaries = []
        for i, batch in enumerate(batch_summaries):
            meta_summaries.append(ChunkSummary(
                chunk_id=i,
                summary=batch.get('summary', ''),
                key_concepts=batch.get('key_concepts', [])[:5],
                word_count=batch.get('total_word_count', 0),
                compression_ratio=1.0
            ))
            
        # Final combination
        return self._combine_chunk_summaries(meta_summaries, config)
        
    def _read_text(self, source: Any, limit: Optional[int] = None) -> str:
        """Read text from various sources."""
        if isinstance(source, str) and os.path.exists(source):
            # File path
            with open(source, 'r', encoding='utf-8', errors='replace') as f:
                if limit:
                    return f.read(limit)
                return f.read()
        elif hasattr(source, 'read'):
            # File-like object
            if limit:
                return source.read(limit)
            return source.read()
        else:
            # Direct string
            return source


# Lazy initialization to avoid circular imports
_unlimited_processor = None


def get_unlimited_processor():
    """Get or create the unlimited processor instance."""
    global _unlimited_processor
    if _unlimited_processor is None:
        _unlimited_processor = UnlimitedTextProcessor()
    return _unlimited_processor


def process_unlimited_text(text_or_path: Any, 
                          config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process text of any length.
    
    Args:
        text_or_path: Text string, file path, or file-like object
        config: Processing configuration
        
    Returns:
        Summary results with metadata
    """
    processor = get_unlimited_processor()
    return processor.process_text(text_or_path, config)