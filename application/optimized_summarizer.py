"""
optimized_summarizer.py - Memory-Optimized Summarization

Provides efficient summarization for large texts with:
- Chunked processing to limit memory usage
- Intelligent text splitting
- Progressive summarization
- Memory-aware processing

Author: SUM Development Team
License: Apache License 2.0
"""

import logging
from typing import Dict, Any, List, Optional
import time
from dataclasses import dataclass

from Models.summarizer import SimpleSummarizer, MagnumOpusSummarizer
from application.feedback_system import apply_feedback_preferences

logger = logging.getLogger(__name__)


@dataclass
class ChunkSummary:
    """Summary of a text chunk"""
    chunk_id: int
    text: str
    summary: str
    word_count: int
    compression_ratio: float


class OptimizedSummarizer:
    """
    Memory-optimized summarizer that processes large texts efficiently.
    """
    
    def __init__(self, 
                 max_chunk_size: int = 50000,
                 max_memory_mb: int = 500):
        """
        Initialize optimized summarizer.
        
        Args:
            max_chunk_size: Maximum characters per chunk
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_chunk_size = max_chunk_size
        self.max_memory_mb = max_memory_mb
        self.simple_summarizer = SimpleSummarizer()
        self.opus_summarizer = MagnumOpusSummarizer()
    
    def summarize_text_universal(self, 
                               text: str,
                               density: str = "all",
                               stream_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Summarize text with memory optimization.
        
        Args:
            text: Input text
            density: Summarization density (all, minimal, short, medium, detailed)
            stream_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with summaries at different densities
        """
        # Apply feedback preferences
        params = apply_feedback_preferences({
            'density': density,
            'text_length': len(text)
        })
        
        # Check if chunking is needed
        if len(text) > self.max_chunk_size:
            return self._chunked_summarization(text, density, stream_callback)
        else:
            return self._simple_summarization(text, density, params)
    
    def _simple_summarization(self, text: str, density: str, params: Dict) -> Dict[str, Any]:
        """Summarize text that fits in memory."""
        start_time = time.time()
        
        # Get word count
        words = text.split()
        word_count = len(words)
        
        # Generate summaries based on density
        result = {}
        
        if density == "all" or density == "tags":
            # Extract key terms (simple version)
            result['tags'] = self._extract_tags(text)
        
        if density == "all" or density == "minimal":
            result['minimal'] = self.simple_summarizer.summarize(text, num_sentences=1)
        
        if density == "all" or density == "short":
            result['short'] = self.simple_summarizer.summarize(text, num_sentences=3)
        
        if density == "all" or density == "medium":
            result['medium'] = self.simple_summarizer.summarize(text, num_sentences=5)
        
        if density == "all" or density == "detailed":
            # For detailed, use opus if available
            try:
                result['detailed'] = self.opus_summarizer.summarize(text)
            except Exception as e:
                logger.warning(f"Opus summarizer failed, falling back to simple: {e}")
                result['detailed'] = self.simple_summarizer.summarize(text, num_sentences=10)
        
        # Add metadata
        result['original_words'] = word_count
        result['processing_time'] = time.time() - start_time
        
        # Calculate compression ratio for primary summary
        primary_key = density if density != "all" else "medium"
        if primary_key in result and result[primary_key]:
            summary_words = len(result[primary_key].split())
            result['compression_ratio'] = word_count / summary_words if summary_words > 0 else 0
        
        return result
    
    def _chunked_summarization(self, 
                             text: str,
                             density: str,
                             stream_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Summarize large text in chunks to optimize memory.
        """
        chunks = self._split_text_intelligently(text)
        chunk_summaries = []
        
        total_chunks = len(chunks)
        logger.info(f"Processing {total_chunks} chunks for large text")
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            if stream_callback:
                stream_callback({
                    'type': 'chunk_progress',
                    'current': i + 1,
                    'total': total_chunks,
                    'progress': ((i + 1) / total_chunks) * 100
                })
            
            # Summarize chunk
            chunk_summary = self.simple_summarizer.summarize(chunk, num_sentences=3)
            
            chunk_summaries.append(ChunkSummary(
                chunk_id=i,
                text=chunk[:100] + "...",  # Store only preview to save memory
                summary=chunk_summary,
                word_count=len(chunk.split()),
                compression_ratio=len(chunk.split()) / len(chunk_summary.split())
            ))
            
            # Clear chunk from memory
            del chunk
        
        # Combine chunk summaries
        combined_summary = " ".join([cs.summary for cs in chunk_summaries])
        
        # Generate final summaries from combined text
        result = self._simple_summarization(combined_summary, density, {})
        
        # Add chunking metadata
        result['chunked'] = True
        result['chunk_count'] = total_chunks
        result['original_words'] = sum(cs.word_count for cs in chunk_summaries)
        
        return result
    
    def _split_text_intelligently(self, text: str) -> List[str]:
        """
        Split text into chunks intelligently at sentence boundaries.
        """
        chunks = []
        current_chunk = []
        current_size = 0
        
        # Split by sentences (simple approach)
        sentences = text.replace('\n', ' ').split('. ')
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add last chunk
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks
    
    def _extract_tags(self, text: str, limit: int = 10) -> List[str]:
        """Extract key terms from text (simple implementation)."""
        # Simple frequency-based extraction
        words = text.lower().split()
        
        # Filter stop words (basic list)
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 
                     'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 'that',
                     'this', 'it', 'from', 'was', 'are', 'been', 'were', 'be'}
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            # Clean word
            word = word.strip('.,!?;:"')
            
            if len(word) > 3 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        tags = [word for word, freq in sorted_words[:limit]]
        
        return tags
    
    def estimate_memory_usage(self, text_length: int) -> Dict[str, Any]:
        """
        Estimate memory usage for processing text of given length.
        
        Returns:
            Dictionary with memory estimates
        """
        # Rough estimates
        text_memory_mb = text_length / (1024 * 1024)  # 1 char = 1 byte approx
        processing_overhead_mb = text_memory_mb * 2  # Processing overhead
        total_mb = text_memory_mb + processing_overhead_mb
        
        return {
            'text_memory_mb': round(text_memory_mb, 2),
            'processing_overhead_mb': round(processing_overhead_mb, 2),
            'total_memory_mb': round(total_mb, 2),
            'requires_chunking': total_mb > self.max_memory_mb,
            'estimated_chunks': max(1, int(total_mb / self.max_memory_mb))
        }


# Global instance
_optimized_summarizer = None


def get_optimized_summarizer() -> OptimizedSummarizer:
    """Get or create global optimized summarizer."""
    global _optimized_summarizer
    if _optimized_summarizer is None:
        _optimized_summarizer = OptimizedSummarizer()
    return _optimized_summarizer


# Convenience function
def summarize_text_universal(text: str, 
                           density: str = "all",
                           stream_callback: Optional[callable] = None) -> Dict[str, Any]:
    """
    Universal text summarization with memory optimization.
    
    Args:
        text: Input text
        density: Summarization density
        stream_callback: Optional progress callback
        
    Returns:
        Dictionary with summaries
    """
    summarizer = get_optimized_summarizer()
    return summarizer.summarize_text_universal(text, density, stream_callback)


if __name__ == "__main__":
    # Example usage
    sample_text = """
    Artificial intelligence is transforming how we live and work. 
    Machine learning algorithms can now perform tasks that were once 
    thought to be exclusively human. From medical diagnosis to creative 
    writing, AI systems are becoming increasingly sophisticated.
    """ * 100  # Repeat to create larger text
    
    # Estimate memory usage
    summarizer = get_optimized_summarizer()
    memory_est = summarizer.estimate_memory_usage(len(sample_text))
    print(f"Memory estimate: {memory_est}")
    
    # Summarize with progress
    def progress_callback(update):
        print(f"Progress: {update}")
    
    result = summarize_text_universal(
        sample_text,
        density="all",
        stream_callback=progress_callback
    )
    
    print(f"Summary generated: {list(result.keys())}")