#!/usr/bin/env python3
"""
StreamingEngine.py - Arbitrary Length Text Processing Engine

This module implements a streaming architecture that can process text of ANY size
by intelligently chunking, processing, and aggregating results while maintaining
semantic coherence and memory efficiency.

Design Philosophy:
- Stream processing for unlimited scalability
- Semantic-aware chunking to preserve meaning
- Progressive refinement for continuous improvement
- Memory-efficient processing with intelligent caching
- Hierarchical aggregation for coherent results

Author: ototao
License: Apache License 2.0
"""

import os
import sys
import time
import logging
import hashlib
import threading
from typing import Dict, List, Any, Optional, Tuple, Iterator, Generator
from dataclasses import dataclass, field
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc

# Import our existing components
from summarization_engine import HierarchicalDensificationEngine
from nltk.tokenize import sent_tokenize, word_tokenize

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


class MemoryMonitor:
    """Monitor and manage memory usage during processing."""
    
    def __init__(self, max_memory_mb: int = 512):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.process = psutil.Process()
        
    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        return self.process.memory_info().rss
    
    def is_memory_critical(self) -> bool:
        """Check if memory usage is approaching limits."""
        current_usage = self.get_memory_usage()
        usage_ratio = current_usage / self.max_memory_bytes
        return usage_ratio > 0.85
    
    def force_garbage_collection(self):
        """Force garbage collection to free memory."""
        gc.collect()
        logger.info(f"Memory after GC: {self.get_memory_usage() / 1024 / 1024:.1f} MB")


class SemanticChunker:
    """Create semantically coherent chunks that preserve meaning and context."""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.sentence_cache = {}
        
    def create_semantic_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        Break text into semantically coherent chunks.
        
        Args:
            text: Input text of arbitrary length
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        logger.info(f"Creating semantic chunks for text of {len(text)} characters")
        
        # Step 1: Analyze document structure
        document_structure = self._analyze_document_structure(text)
        
        # Step 2: Create initial sentence-level segmentation
        sentences = sent_tokenize(text)
        sentence_boundaries = self._calculate_sentence_positions(text, sentences)
        
        # Step 3: Group sentences into coherent chunks
        chunks = self._group_sentences_into_chunks(
            sentences, sentence_boundaries, document_structure
        )
        
        # Step 4: Add overlap for context preservation
        overlapped_chunks = self._add_semantic_overlap(chunks, sentences)
        
        # Step 5: Generate metadata for each chunk
        enriched_chunks = self._enrich_chunks_with_metadata(overlapped_chunks, text)
        
        logger.info(f"Created {len(enriched_chunks)} semantic chunks")
        return enriched_chunks
    
    def _analyze_document_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure to inform chunking strategy."""
        structure = {
            'total_length': len(text),
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
            'sentence_count': len(sent_tokenize(text)),
            'avg_sentence_length': 0,
            'section_markers': [],
            'topic_transitions': []
        }
        
        sentences = sent_tokenize(text)
        if sentences:
            total_words = sum(len(word_tokenize(s)) for s in sentences)
            structure['avg_sentence_length'] = total_words / len(sentences)
        
        # Detect section markers (headers, numbered sections, etc.)
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if self._is_section_marker(line):
                structure['section_markers'].append({
                    'line_number': i,
                    'text': line,
                    'marker_type': self._classify_section_marker(line)
                })
        
        return structure
    
    def _is_section_marker(self, line: str) -> bool:
        """Detect if a line is likely a section header or marker."""
        line = line.strip()
        if not line:
            return False
            
        # Check for common section markers
        section_patterns = [
            lambda l: l.startswith('#'),  # Markdown headers
            lambda l: l.isupper() and len(l.split()) <= 8,  # ALL CAPS headers
            lambda l: l.startswith(('Chapter', 'Section', 'Part')),  # Explicit markers
            lambda l: l[0].isdigit() and '.' in l[:10],  # Numbered sections
            lambda l: len(l) < 50 and l.endswith(':'),  # Short lines ending with colon
        ]
        
        return any(pattern(line) for pattern in section_patterns)
    
    def _classify_section_marker(self, line: str) -> str:
        """Classify the type of section marker."""
        line = line.strip()
        if line.startswith('#'):
            return 'markdown_header'
        elif line.isupper():
            return 'caps_header' 
        elif line.startswith(('Chapter', 'Section', 'Part')):
            return 'explicit_section'
        elif line[0].isdigit():
            return 'numbered_section'
        elif line.endswith(':'):
            return 'colon_header'
        else:
            return 'unknown'
    
    def _calculate_sentence_positions(self, text: str, sentences: List[str]) -> List[Tuple[int, int]]:
        """Calculate start and end positions of each sentence in the original text."""
        positions = []
        current_pos = 0
        
        for sentence in sentences:
            # Find the sentence in the text starting from current position
            start_pos = text.find(sentence, current_pos)
            if start_pos == -1:
                # Fallback: approximate position
                start_pos = current_pos
            
            end_pos = start_pos + len(sentence)
            positions.append((start_pos, end_pos))
            current_pos = end_pos
        
        return positions
    
    def _group_sentences_into_chunks(self, sentences: List[str], positions: List[Tuple[int, int]], 
                                   structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Group sentences into coherent chunks based on semantic boundaries."""
        chunks = []
        current_chunk_sentences = []
        current_chunk_word_count = 0
        chunk_start_pos = 0
        
        target_chunk_size = self.config.chunk_size_words
        
        for i, (sentence, (start_pos, end_pos)) in enumerate(zip(sentences, positions)):
            sentence_word_count = len(word_tokenize(sentence))
            
            # Check if adding this sentence would exceed chunk size
            if (current_chunk_word_count + sentence_word_count > target_chunk_size and 
                current_chunk_sentences):  # Don't create empty chunks
                
                # Create chunk from accumulated sentences
                chunk = self._create_chunk_from_sentences(
                    current_chunk_sentences, chunk_start_pos, positions[i-len(current_chunk_sentences):i]
                )
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk_sentences = [sentence]
                current_chunk_word_count = sentence_word_count
                chunk_start_pos = start_pos
            else:
                # Add sentence to current chunk
                current_chunk_sentences.append(sentence)
                current_chunk_word_count += sentence_word_count
                
                if not current_chunk_sentences or len(current_chunk_sentences) == 1:
                    chunk_start_pos = start_pos
        
        # Don't forget the last chunk
        if current_chunk_sentences:
            chunk = self._create_chunk_from_sentences(
                current_chunk_sentences, chunk_start_pos, 
                positions[len(sentences)-len(current_chunk_sentences):]
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk_from_sentences(self, sentences: List[str], start_pos: int, 
                                   sentence_positions: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Create a chunk dictionary from a list of sentences."""
        content = ' '.join(sentences)
        end_pos = sentence_positions[-1][1] if sentence_positions else start_pos + len(content)
        
        return {
            'content': content,
            'start_position': start_pos,
            'end_position': end_pos,
            'sentence_count': len(sentences),
            'word_count': len(word_tokenize(content)),
            'sentences': sentences
        }
    
    def _add_semantic_overlap(self, chunks: List[Dict[str, Any]], all_sentences: List[str]) -> List[Dict[str, Any]]:
        """Add overlapping content between chunks to preserve context."""
        if len(chunks) <= 1:
            return chunks
        
        overlap_sentence_count = max(1, int(self.config.overlap_ratio * 10))  # Rough estimate
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            overlapped_chunk = chunk.copy()
            
            # Add sentences from previous chunk (prefix overlap)
            if i > 0:
                prev_sentences = chunks[i-1]['sentences']
                overlap_sentences = prev_sentences[-overlap_sentence_count:]
                
                overlapped_chunk['content'] = ' '.join(overlap_sentences) + ' ' + chunk['content']
                overlapped_chunk['has_prefix_overlap'] = True
                overlapped_chunk['prefix_sentences'] = overlap_sentences
            else:
                overlapped_chunk['has_prefix_overlap'] = False
            
            # Add sentences from next chunk (suffix overlap)  
            if i < len(chunks) - 1:
                next_sentences = chunks[i+1]['sentences']
                overlap_sentences = next_sentences[:overlap_sentence_count]
                
                overlapped_chunk['content'] = overlapped_chunk['content'] + ' ' + ' '.join(overlap_sentences)
                overlapped_chunk['has_suffix_overlap'] = True
                overlapped_chunk['suffix_sentences'] = overlap_sentences
            else:
                overlapped_chunk['has_suffix_overlap'] = False
            
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
    
    def _enrich_chunks_with_metadata(self, chunks: List[Dict[str, Any]], original_text: str) -> List[Dict[str, Any]]:
        """Add rich metadata to each chunk for better processing."""
        enriched_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Generate unique chunk ID
            chunk_id = hashlib.md5(
                f"{chunk['start_position']}_{chunk['end_position']}_{chunk['content'][:100]}".encode()
            ).hexdigest()[:12]
            
            # Calculate topic coherence (simplified)
            coherence_score = self._calculate_topic_coherence(chunk['content'])
            
            # Create enriched chunk
            enriched_chunk = {
                **chunk,
                'chunk_id': chunk_id,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'topic_coherence_score': coherence_score,
                'processing_priority': self._calculate_processing_priority(chunk, i, len(chunks)),
            }
            
            enriched_chunks.append(enriched_chunk)
        
        return enriched_chunks
    
    def _calculate_topic_coherence(self, text: str) -> float:
        """Calculate a simple topic coherence score for the chunk."""
        # This is a simplified implementation
        # In production, you'd use more sophisticated NLP techniques
        
        words = word_tokenize(text.lower())
        unique_words = set(words)
        
        if not words:
            return 0.0
            
        # Simple coherence based on word repetition and length
        repetition_score = 1.0 - (len(unique_words) / len(words))
        length_score = min(1.0, len(words) / 500)  # Optimal around 500 words
        
        return (repetition_score + length_score) / 2
    
    def _calculate_processing_priority(self, chunk: Dict[str, Any], index: int, total: int) -> float:
        """Calculate processing priority for this chunk."""
        # Higher priority for:
        # 1. Chunks with better topic coherence
        # 2. Chunks at the beginning (often contain introductions)
        # 3. Chunks with good word count (not too short, not too long)
        
        coherence_score = chunk.get('topic_coherence_score', 0.5)
        position_score = 1.0 - (index / total)  # Earlier chunks get higher score
        
        word_count = chunk.get('word_count', 0)
        optimal_word_count = self.config.chunk_size_words
        length_score = 1.0 - abs(word_count - optimal_word_count) / optimal_word_count
        
        return (coherence_score + position_score + length_score) / 3


class ChunkProcessor:
    """Process individual chunks using the existing hierarchical engine."""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.hierarchical_engine = HierarchicalDensificationEngine()
        self.processing_cache = {}
        
    def process_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single chunk through hierarchical densification.
        
        Args:
            chunk: Chunk dictionary with content and metadata
            
        Returns:
            Processed result with hierarchical summary and insights
        """
        chunk_id = chunk['chunk_id']
        
        # Check cache first
        if self.config.cache_processed_chunks and chunk_id in self.processing_cache:
            logger.info(f"Using cached result for chunk {chunk_id}")
            return self.processing_cache[chunk_id]
        
        start_time = time.time()
        
        # Configure processing based on chunk characteristics
        processing_config = self._adapt_config_for_chunk(chunk)
        
        # Process through hierarchical engine
        try:
            result = self.hierarchical_engine.process_text(chunk['content'], processing_config)
            
            # Enrich result with chunk metadata
            enriched_result = self._enrich_result_with_chunk_info(result, chunk, start_time)
            
            # Cache result if enabled
            if self.config.cache_processed_chunks:
                self.processing_cache[chunk_id] = enriched_result
            
            logger.info(f"Processed chunk {chunk_id} in {time.time() - start_time:.2f}s")
            return enriched_result
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_id}: {e}")
            return self._create_error_result(chunk, str(e))
    
    def _adapt_config_for_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt processing configuration based on chunk characteristics."""
        base_config = {
            'max_concepts': 5,
            'max_summary_tokens': 50,
            'complexity_threshold': 0.7,
            'max_insights': 3,
            'min_insight_score': 0.6
        }
        
        # Adjust based on chunk size
        word_count = chunk.get('word_count', 500)
        if word_count > 1500:
            base_config['max_concepts'] = 8
            base_config['max_summary_tokens'] = 80
            base_config['max_insights'] = 5
        elif word_count < 300:
            base_config['max_concepts'] = 3
            base_config['max_summary_tokens'] = 30
            base_config['max_insights'] = 2
        
        # Adjust based on topic coherence
        coherence = chunk.get('topic_coherence_score', 0.5)
        if coherence > 0.8:
            base_config['complexity_threshold'] = 0.6  # Allow more expansion for coherent chunks
        
        return base_config
    
    def _enrich_result_with_chunk_info(self, result: Dict[str, Any], 
                                     chunk: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Add chunk-specific metadata to the processing result."""
        processing_time = time.time() - start_time
        
        # Create enriched result
        enriched_result = {
            **result,
            'chunk_metadata': ChunkMetadata(
                chunk_id=chunk['chunk_id'],
                start_position=chunk['start_position'],
                end_position=chunk['end_position'],
                semantic_boundaries=[],  # Could be enhanced
                topic_coherence_score=chunk.get('topic_coherence_score', 0.0),
                processing_time=processing_time,
                memory_usage=0,  # Could add memory tracking
            ),
            'chunk_info': {
                'chunk_index': chunk.get('chunk_index', 0),
                'total_chunks': chunk.get('total_chunks', 1),
                'word_count': chunk.get('word_count', 0),
                'has_overlap': chunk.get('has_prefix_overlap', False) or chunk.get('has_suffix_overlap', False)
            }
        }
        
        return enriched_result
    
    def _create_error_result(self, chunk: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """Create an error result for a failed chunk."""
        return {
            'error': f"Chunk processing failed: {error_message}",
            'chunk_metadata': ChunkMetadata(
                chunk_id=chunk['chunk_id'],
                start_position=chunk['start_position'],
                end_position=chunk['end_position'],
                semantic_boundaries=[],
                topic_coherence_score=0.0,
                processing_time=0.0,
                memory_usage=0,
            ),
            'chunk_info': {
                'chunk_index': chunk.get('chunk_index', 0),
                'total_chunks': chunk.get('total_chunks', 1),
                'word_count': chunk.get('word_count', 0),
                'processing_failed': True
            }
        }


class StreamingHierarchicalEngine:
    """
    Streaming engine for processing text of arbitrary length.
    
    This engine can handle documents of any size by:
    1. Breaking text into semantically coherent chunks
    2. Processing chunks in parallel while managing memory
    3. Aggregating results while preserving global coherence
    4. Providing progressive refinement as more text is processed
    """
    
    def __init__(self, config: Optional[StreamingConfig] = None):
        self.config = config or StreamingConfig()
        self.semantic_chunker = SemanticChunker(self.config)
        self.chunk_processor = ChunkProcessor(self.config)
        self.memory_monitor = MemoryMonitor(self.config.max_memory_mb)
        
        # Threading setup for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_chunks)
        self.processing_lock = threading.Lock()
        
        logger.info("StreamingHierarchicalEngine initialized")
    
    def process_streaming_text(self, text: str, target_compression: Optional[float] = None) -> Dict[str, Any]:
        """
        Process text of arbitrary length through streaming pipeline.
        
        Args:
            text: Input text of any size
            target_compression: Optional target compression ratio
            
        Returns:
            Comprehensive hierarchical processing result
        """
        logger.info(f"Starting streaming processing of {len(text)} character text")
        start_time = time.time()
        
        try:
            # Phase 1: Create semantic chunks
            chunks = self.semantic_chunker.create_semantic_chunks(text)
            logger.info(f"Created {len(chunks)} semantic chunks")
            
            # Phase 2: Process chunks with memory management
            chunk_results = self._process_chunks_with_memory_management(chunks)
            
            # Phase 3: Aggregate results into coherent output
            final_result = self._aggregate_chunk_results(chunk_results, text)
            
            # Phase 4: Add streaming metadata
            final_result['streaming_metadata'] = {
                'total_processing_time': time.time() - start_time,
                'chunks_processed': len(chunks),
                'original_text_length': len(text),
                'memory_efficiency': self._calculate_memory_efficiency(),
                'processing_method': 'streaming_hierarchical'
            }
            
            logger.info(f"Streaming processing completed in {time.time() - start_time:.2f}s")
            return final_result
            
        except Exception as e:
            logger.error(f"Streaming processing failed: {e}")
            return {
                'error': f"Streaming processing failed: {str(e)}",
                'streaming_metadata': {
                    'processing_failed': True,
                    'error_time': time.time() - start_time
                }
            }
    
    def _process_chunks_with_memory_management(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process chunks while managing memory usage."""
        results = []
        chunk_batches = self._create_memory_efficient_batches(chunks)
        
        for batch_index, batch in enumerate(chunk_batches):
            logger.info(f"Processing batch {batch_index + 1}/{len(chunk_batches)} with {len(batch)} chunks")
            
            # Process batch in parallel
            batch_results = self._process_chunk_batch(batch)
            results.extend(batch_results)
            
            # Memory management
            if self.memory_monitor.is_memory_critical():
                logger.warning("Memory usage critical, forcing garbage collection")
                self.memory_monitor.force_garbage_collection()
        
        return results
    
    def _create_memory_efficient_batches(self, chunks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Create batches of chunks that fit within memory constraints."""
        batches = []
        current_batch = []
        current_batch_size = 0
        
        # Estimate memory per chunk (rough heuristic)
        avg_chunk_memory = 1024 * 50  # 50KB per chunk estimate
        max_batch_memory = self.config.max_memory_mb * 1024 * 1024 * 0.3  # Use 30% of available memory
        
        for chunk in chunks:
            chunk_memory = len(chunk['content']) * 2  # Rough estimate
            
            if current_batch_size + chunk_memory > max_batch_memory and current_batch:
                batches.append(current_batch)
                current_batch = [chunk]
                current_batch_size = chunk_memory
            else:
                current_batch.append(chunk)
                current_batch_size += chunk_memory
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _process_chunk_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of chunks in parallel."""
        futures = []
        
        # Submit chunks for processing
        for chunk in batch:
            future = self.executor.submit(self.chunk_processor.process_chunk, chunk)
            futures.append(future)
        
        # Collect results as they complete
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Chunk processing failed: {e}")
                # Create error result
                error_result = {'error': str(e), 'chunk_failed': True}
                results.append(error_result)
        
        return results
    
    def _aggregate_chunk_results(self, chunk_results: List[Dict[str, Any]], original_text: str) -> Dict[str, Any]:
        """Aggregate results from all chunks into a coherent final result."""
        logger.info(f"Aggregating results from {len(chunk_results)} chunks")
        
        # Filter out failed chunks
        successful_results = [r for r in chunk_results if 'error' not in r]
        failed_count = len(chunk_results) - len(successful_results)
        
        if failed_count > 0:
            logger.warning(f"{failed_count} chunks failed processing")
        
        if not successful_results:
            return {
                'error': 'All chunks failed processing',
                'failed_chunks': failed_count
            }
        
        # Aggregate Level 1: Concepts
        all_concepts = []
        for result in successful_results:
            if 'hierarchical_summary' in result:
                concepts = result['hierarchical_summary'].get('level_1_concepts', [])
                all_concepts.extend(concepts)
        
        # Deduplicate and rank concepts
        unified_concepts = self._merge_and_rank_concepts(all_concepts)
        
        # Aggregate Level 2: Core summaries
        all_summaries = []
        for result in successful_results:
            if 'hierarchical_summary' in result:
                summary = result['hierarchical_summary'].get('level_2_core', '')
                if summary:
                    all_summaries.append(summary)
        
        unified_summary = self._synthesize_summaries(all_summaries)
        
        # Aggregate insights
        all_insights = []
        for result in successful_results:
            insights = result.get('key_insights', [])
            all_insights.extend(insights)
        
        unified_insights = self._consolidate_insights(all_insights)
        
        # Calculate aggregate metadata
        aggregate_metadata = self._calculate_aggregate_metadata(successful_results, original_text)
        
        return {
            'hierarchical_summary': {
                'level_1_concepts': unified_concepts,
                'level_2_core': unified_summary,
                'level_3_expanded': self._generate_expanded_context(unified_summary, unified_concepts)
            },
            'key_insights': unified_insights,
            'metadata': aggregate_metadata,
            'processing_stats': {
                'total_chunks': len(chunk_results),
                'successful_chunks': len(successful_results),
                'failed_chunks': failed_count,
                'success_rate': len(successful_results) / len(chunk_results) if chunk_results else 0
            }
        }
    
    def _merge_and_rank_concepts(self, all_concepts: List[str]) -> List[str]:
        """Merge concepts from all chunks and rank by importance."""
        # Count concept frequency
        concept_counts = defaultdict(int)
        for concept in all_concepts:
            concept_counts[concept.lower()] += 1
        
        # Sort by frequency and return top concepts
        sorted_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top 10 concepts, preserving original case
        top_concepts = []
        seen = set()
        for concept in all_concepts:
            if concept.lower() not in seen and len(top_concepts) < 10:
                top_concepts.append(concept)
                seen.add(concept.lower())
        
        return top_concepts
    
    def _synthesize_summaries(self, summaries: List[str]) -> str:
        """Synthesize multiple chunk summaries into a coherent overall summary."""
        if not summaries:
            return ""
        
        if len(summaries) == 1:
            return summaries[0]
        
        # Simple synthesis: combine key sentences from each summary
        # In production, this would use more sophisticated NLP
        combined_sentences = []
        
        for summary in summaries:
            sentences = sent_tokenize(summary)
            # Take the first sentence from each summary (usually the most important)
            if sentences:
                combined_sentences.append(sentences[0])
        
        # Join and clean up
        synthesized = ' '.join(combined_sentences)
        
        # Truncate if too long
        words = word_tokenize(synthesized)
        if len(words) > 100:
            synthesized = ' '.join(words[:100]) + '...'
        
        return synthesized
    
    def _consolidate_insights(self, all_insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Consolidate insights from all chunks, removing duplicates and ranking."""
        if not all_insights:
            return []
        
        # Remove duplicates based on text similarity
        unique_insights = []
        seen_texts = set()
        
        # Sort insights by score first
        sorted_insights = sorted(all_insights, key=lambda x: x.get('score', 0), reverse=True)
        
        for insight in sorted_insights:
            insight_text = insight.get('text', '').lower().strip()
            
            # Simple duplicate detection (could be enhanced with semantic similarity)
            if insight_text not in seen_texts and len(unique_insights) < 10:
                unique_insights.append(insight)
                seen_texts.add(insight_text)
        
        return unique_insights
    
    def _generate_expanded_context(self, summary: str, concepts: List[str]) -> Optional[str]:
        """Generate expanded context if the summary is too brief."""
        # Simple heuristic: expand if summary is very short
        word_count = len(word_tokenize(summary))
        
        if word_count < 20 and concepts:
            # Create expanded context from concepts
            expanded = f"{summary} This analysis focuses on key concepts including {', '.join(concepts[:5])}."
            return expanded
        
        return None  # No expansion needed
    
    def _calculate_aggregate_metadata(self, results: List[Dict[str, Any]], original_text: str) -> Dict[str, Any]:
        """Calculate aggregate metadata from all chunk results."""
        total_processing_time = sum(
            result.get('chunk_metadata', ChunkMetadata('', 0, 0, [], 0.0, 0.0, 0)).processing_time 
            for result in results
        )
        
        original_word_count = len(word_tokenize(original_text))
        
        # Calculate compression ratio (rough estimate)
        total_summary_words = 0
        for result in results:
            if 'hierarchical_summary' in result:
                summary = result['hierarchical_summary'].get('level_2_core', '')
                total_summary_words += len(word_tokenize(summary))
        
        compression_ratio = total_summary_words / original_word_count if original_word_count > 0 else 1.0
        
        return {
            'processing_time': total_processing_time,
            'original_word_count': original_word_count,
            'compression_ratio': compression_ratio,
            'processing_method': 'streaming_hierarchical',
            'chunks_processed': len(results)
        }
    
    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency score."""
        current_memory = self.memory_monitor.get_memory_usage()
        max_memory = self.memory_monitor.max_memory_bytes
        
        efficiency = 1.0 - (current_memory / max_memory)
        return max(0.0, efficiency)
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# Example usage and testing
if __name__ == "__main__":
    # Test with a sample long text
    sample_long_text = """
    Machine learning has revolutionized the way we approach complex problems across various domains.
    From healthcare to finance, from transportation to entertainment, machine learning algorithms
    are driving innovation and efficiency at an unprecedented scale.
    
    The foundation of machine learning lies in its ability to learn patterns from data without
    being explicitly programmed for every possible scenario. This capability makes it particularly
    valuable for handling tasks that are too complex for traditional rule-based programming approaches.
    
    Deep learning, a subset of machine learning, has been particularly transformative. Neural networks
    with multiple layers can learn hierarchical representations of data, enabling breakthroughs in
    image recognition, natural language processing, and speech recognition.
    
    However, with great power comes great responsibility. The deployment of machine learning systems
    raises important questions about bias, fairness, and interpretability. As these systems become
    more prevalent in decision-making processes, ensuring they operate ethically and transparently
    becomes crucial.
    
    The future of machine learning promises even more exciting developments. Quantum machine learning,
    federated learning, and neuromorphic computing represent just a few of the emerging paradigms
    that could reshape the field in the coming years.
    """ * 10  # Simulate longer text
    
    print("🚀 Testing StreamingHierarchicalEngine")
    print(f"📝 Text length: {len(sample_long_text)} characters")
    print(f"📊 Word count: {len(word_tokenize(sample_long_text))}")
    
    # Create and configure engine
    config = StreamingConfig(
        chunk_size_words=200,
        overlap_ratio=0.1,
        max_memory_mb=256,
        max_concurrent_chunks=2
    )
    
    engine = StreamingHierarchicalEngine(config)
    
    # Process the text
    start_time = time.time()
    result = engine.process_streaming_text(sample_long_text)
    processing_time = time.time() - start_time
    
    # Display results
    print(f"\n⚡ Processing completed in {processing_time:.2f} seconds")
    print(f"📈 Success rate: {result.get('processing_stats', {}).get('success_rate', 0):.1%}")
    
    if 'hierarchical_summary' in result:
        print(f"\n🎯 Level 1 Concepts: {result['hierarchical_summary']['level_1_concepts']}")
        print(f"\n💎 Level 2 Core Summary:\n{result['hierarchical_summary']['level_2_core']}")
        
        if result['hierarchical_summary']['level_3_expanded']:
            print(f"\n📖 Level 3 Expanded:\n{result['hierarchical_summary']['level_3_expanded']}")
        
        print(f"\n🌟 Key Insights ({len(result.get('key_insights', []))}):")
        for i, insight in enumerate(result.get('key_insights', [])[:3], 1):
            print(f"   {i}. [{insight.get('type', 'INSIGHT')}] {insight.get('text', '')}")
            print(f"      Score: {insight.get('score', 0):.2f}")
    
    if 'streaming_metadata' in result:
        metadata = result['streaming_metadata']
        print(f"\n📊 Streaming Metadata:")
        print(f"   Chunks processed: {metadata.get('chunks_processed', 0)}")
        print(f"   Original length: {metadata.get('original_text_length', 0)} chars")
        print(f"   Memory efficiency: {metadata.get('memory_efficiency', 0):.1%}")
    
    print("\n🎉 StreamingHierarchicalEngine test completed!")