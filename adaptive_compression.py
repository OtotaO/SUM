"""
Adaptive Compression Engine

Implements content-aware text compression that selects appropriate strategies
based on text characteristics and information density analysis.

Features:
- Automatic content type detection (philosophical, technical, narrative, etc.)
- Information density measurement for compression ratio adjustment
- Multiple compression strategies optimized for different content types
- Quality benchmarking using reference texts with known compression limits
- Temporal compression hierarchy for time-based data

Author: ototao & Claude
License: Apache 2.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import math
import re
from abc import ABC, abstractmethod
import logging

# Configure logging
logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Types of content requiring different compression strategies."""
    PHILOSOPHICAL = "philosophical"  # Dense ideas, every word matters
    TECHNICAL = "technical"          # Precise terminology, formulas
    NARRATIVE = "narrative"          # Stories, can compress more
    CONVERSATIONAL = "conversational" # Dialogue, high redundancy
    POETIC = "poetic"               # Rhythm and form matter
    ACTIVITY_LOG = "activity_log"    # User activities, timestamps
    LIFE_SUMMARY = "life_summary"    # Compressed life periods


@dataclass
class CompressionMetrics:
    """Metrics for evaluating compression quality."""
    compression_ratio: float
    information_retention: float
    semantic_coherence: float
    readability_score: float
    processing_time: float


class CompressionStrategy(ABC):
    """Abstract base class for compression strategies."""
    
    @abstractmethod
    def compress(self, text: str, target_ratio: float) -> Dict[str, Any]:
        """Compress text to target ratio."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy identifier."""
        pass


class PhilosophicalCompression(CompressionStrategy):
    """
    Compression strategy for philosophical and dense texts.
    Preserves logical connectors, arguments, and definitional statements.
    """
    
    def compress(self, text: str, target_ratio: float = 0.3) -> Dict[str, Any]:
        sentences = sent_tokenize(text)
        
        # Extract logical connectors and key concepts
        logical_patterns = [
            r'\btherefore\b', r'\bhowever\b', r'\bthus\b',
            r'\bconsequently\b', r'\bnevertheless\b'
        ]
        
        # Score sentences by philosophical importance
        scored_sentences = []
        for sent in sentences:
            score = 0
            
            # Logical connectors
            for pattern in logical_patterns:
                if re.search(pattern, sent.lower()):
                    score += 2
            
            # Question sentences (philosophical inquiry)
            if '?' in sent:
                score += 1.5
            
            # Definitional patterns
            if re.search(r'\bis\s+(defined|considered|understood)\s+as\b', sent.lower()):
                score += 2
            
            # Abstract concepts (capitalized non-proper nouns)
            abstract_concepts = re.findall(r'\b[A-Z][a-z]+\b', sent)
            score += len(abstract_concepts) * 0.5
            
            scored_sentences.append((score, sent))
        
        # Sort by importance
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        # Select top sentences to meet ratio
        target_words = int(len(word_tokenize(text)) * target_ratio)
        selected = []
        word_count = 0
        
        for score, sent in scored_sentences:
            words = word_tokenize(sent)
            if word_count + len(words) <= target_words:
                selected.append(sent)
                word_count += len(words)
        
        # Maintain original order
        selected = [s for s in sentences if s in selected]
        
        return {
            'compressed': ' '.join(selected),
            'strategy': self.get_strategy_name(),
            'preserved_elements': {
                'logical_structure': True,
                'key_arguments': True,
                'definitions': True
            }
        }
    
    def get_strategy_name(self) -> str:
        return "philosophical_preservation"


class TechnicalCompression(CompressionStrategy):
    """
    Compression strategy for technical content.
    Preserves code blocks, formulas, technical terminology, and measurements.
    """
    
    def compress(self, text: str, target_ratio: float = 0.4) -> Dict[str, Any]:
        sentences = sent_tokenize(text)
        
        # Technical patterns to preserve
        code_pattern = r'`[^`]+`|\b\w+\(\)|{\s*[\s\S]*?\s*}'
        formula_pattern = r'[A-Za-z]+\s*=\s*[^.]+|O\([^)]+\)'
        
        scored_sentences = []
        for sent in sentences:
            score = 0
            
            # Code snippets
            code_matches = re.findall(code_pattern, sent)
            score += len(code_matches) * 3
            
            # Formulas
            formula_matches = re.findall(formula_pattern, sent)
            score += len(formula_matches) * 2.5
            
            # Technical terms (camelCase, snake_case)
            tech_terms = re.findall(r'\b[a-z]+_[a-z]+\b|\b[a-z]+[A-Z][a-zA-Z]*\b', sent)
            score += len(tech_terms) * 1.5
            
            # Numbers and measurements
            numbers = re.findall(r'\b\d+\.?\d*\s*[A-Za-z]*\b', sent)
            score += len(numbers) * 1
            
            scored_sentences.append((score, sent))
        
        # Include high-score technical sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        target_words = int(len(word_tokenize(text)) * target_ratio)
        selected = []
        word_count = 0
        
        for score, sent in scored_sentences:
            words = word_tokenize(sent)
            if word_count + len(words) <= target_words:
                selected.append(sent)
                word_count += len(words)
        
        selected = [s for s in sentences if s in selected]
        
        return {
            'compressed': ' '.join(selected),
            'strategy': self.get_strategy_name(),
            'preserved_elements': {
                'code_snippets': True,
                'formulas': True,
                'technical_terms': True,
                'measurements': True
            }
        }
    
    def get_strategy_name(self) -> str:
        return "technical_precision"


class ActivityLogCompression(CompressionStrategy):
    """
    Compression for activity logs and life events.
    Groups similar activities and preserves temporal patterns.
    """
    
    def compress(self, text: str, target_ratio: float = 0.1) -> Dict[str, Any]:
        lines = text.strip().split('\n')
        
        # Parse activities (expecting timestamp + activity format)
        activities = []
        for line in lines:
            # Simple parsing - can be enhanced
            parts = line.split(' ', 2)
            if len(parts) >= 3:
                timestamp = parts[0] + ' ' + parts[1]
                activity = parts[2]
                activities.append((timestamp, activity))
        
        # Group similar activities
        activity_groups = {}
        for timestamp, activity in activities:
            # Simple grouping by first significant word
            words = activity.lower().split()
            key_word = next((w for w in words if w not in stopwords.words('english')), 'misc')
            
            if key_word not in activity_groups:
                activity_groups[key_word] = []
            activity_groups[key_word].append((timestamp, activity))
        
        # Compress each group
        compressed_activities = []
        for key, group in activity_groups.items():
            if len(group) > 3:
                # Summarize repeated activities
                first_time = group[0][0]
                last_time = group[-1][0]
                compressed_activities.append(
                    f"{first_time} - {last_time}: {key} activities ({len(group)} times)"
                )
            else:
                # Keep individual activities
                for timestamp, activity in group:
                    compressed_activities.append(f"{timestamp}: {activity}")
        
        # Sort by timestamp
        compressed_activities.sort()
        
        # Apply target ratio
        target_items = max(1, int(len(compressed_activities) * target_ratio))
        
        # Select most important (for now, evenly distributed)
        step = max(1, len(compressed_activities) // target_items)
        selected = compressed_activities[::step]
        
        return {
            'compressed': '\n'.join(selected),
            'strategy': self.get_strategy_name(),
            'metadata': {
                'original_activities': len(activities),
                'compressed_activities': len(selected),
                'time_range': f"{activities[0][0]} to {activities[-1][0]}" if activities else "N/A"
            }
        }
    
    def get_strategy_name(self) -> str:
        return "activity_temporal_grouping"


class AdaptiveCompressionEngine:
    """
    Content-aware compression engine that analyzes text characteristics
    and applies appropriate compression strategies based on content type
    and information density measurements.
    """
    
    def __init__(self):
        self.strategies = {
            ContentType.PHILOSOPHICAL: PhilosophicalCompression(),
            ContentType.TECHNICAL: TechnicalCompression(),
            ContentType.ACTIVITY_LOG: ActivityLogCompression(),
            # More strategies to be added
        }
        
        # Initialize golden texts collection for benchmarking
        from golden_texts import GoldenTextsCollection
        self.golden_texts_collection = GoldenTextsCollection()
        
        # Load NLTK data
        self._ensure_nltk_data()
    
    def _ensure_nltk_data(self):
        """Ensure required NLTK data is downloaded."""
        required_data = ['punkt', 'stopwords']
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                nltk.download(data)
    
    def analyze_content_type(self, text: str) -> ContentType:
        """
        Analyze text to determine its content type.
        Uses heuristics and pattern matching.
        """
        # Check for activity log patterns
        if re.search(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}', text, re.MULTILINE):
            return ContentType.ACTIVITY_LOG
        
        # Analyze vocabulary and structure
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        # Technical indicators
        tech_score = 0
        tech_patterns = [
            r'\b(function|def|class|import|return)\b',
            r'\b\w+\(\)',
            r'[A-Z_]+',  # Constants
            r'\d+\.\d+',  # Decimals
        ]
        for pattern in tech_patterns:
            tech_score += len(re.findall(pattern, text))
        
        # Philosophical indicators
        phil_score = 0
        phil_words = ['therefore', 'thus', 'consequently', 'meaning', 
                      'existence', 'truth', 'reality', 'consciousness']
        for word in phil_words:
            phil_score += words.count(word) * 2
        
        # Question ratio (philosophical texts often pose questions)
        question_ratio = sum(1 for s in sentences if '?' in s) / max(1, len(sentences))
        phil_score += question_ratio * 10
        
        # Determine type based on scores
        if tech_score > phil_score * 1.5:
            return ContentType.TECHNICAL
        elif phil_score > tech_score:
            return ContentType.PHILOSOPHICAL
        else:
            return ContentType.NARRATIVE  # Default
    
    def measure_information_density(self, text: str) -> float:
        """
        Measure information density of text.
        Higher density = more unique information per word.
        """
        words = word_tokenize(text.lower())
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        content_words = [w for w in words if w not in stop_words and w.isalnum()]
        
        if not content_words:
            return 0.0
        
        # Calculate entropy
        word_freq = Counter(content_words)
        total_words = len(content_words)
        
        entropy = 0
        for count in word_freq.values():
            probability = count / total_words
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        # Normalize by max possible entropy
        max_entropy = math.log2(len(word_freq))
        density = entropy / max_entropy if max_entropy > 0 else 0
        
        # Adjust for unique word ratio
        unique_ratio = len(word_freq) / total_words
        density = (density + unique_ratio) / 2
        
        return min(1.0, density)
    
    def compress(self, text: str, target_ratio: float = 0.2,
                 force_type: Optional[ContentType] = None) -> Dict[str, Any]:
        """
        Compress text using adaptive strategy selection.
        
        Args:
            text: Input text to compress
            target_ratio: Target compression ratio (0.1 = 10% of original)
            force_type: Force specific content type (optional)
        
        Returns:
            Dictionary with compressed text and metadata
        """
        import time
        start_time = time.time()
        
        # Determine content type
        content_type = force_type or self.analyze_content_type(text)
        
        # Measure information density
        density = self.measure_information_density(text)
        
        # Adjust target ratio based on density
        # Higher density = less aggressive compression
        adjusted_ratio = target_ratio * (1 + density * 0.5)
        adjusted_ratio = min(0.8, adjusted_ratio)  # Cap at 80%
        
        # Select and apply strategy
        strategy = self.strategies.get(content_type)
        if not strategy:
            # Fallback to narrative compression
            strategy = self.strategies.get(ContentType.NARRATIVE, PhilosophicalCompression())
        
        result = strategy.compress(text, adjusted_ratio)
        
        # Calculate metrics
        original_words = len(word_tokenize(text))
        compressed_words = len(word_tokenize(result['compressed']))
        actual_ratio = compressed_words / max(1, original_words)
        
        # Enhanced result with metrics
        result.update({
            'content_type': content_type.value if hasattr(content_type, 'value') else str(content_type),
            'information_density': density,
            'target_ratio': target_ratio,
            'adjusted_ratio': adjusted_ratio,
            'actual_ratio': actual_ratio,
            'processing_time': time.time() - start_time,
            'original_length': original_words,
            'compressed_length': compressed_words
        })
        
        content_type_str = content_type.value if hasattr(content_type, 'value') else str(content_type)
        logger.info(f"Compressed {content_type_str} text: "
                   f"{original_words} → {compressed_words} words "
                   f"(ratio: {actual_ratio:.2%})")
        
        return result
    
    def benchmark_compression(self) -> Dict[str, CompressionMetrics]:
        """
        Benchmark compression quality using golden texts.
        Returns metrics for each content category.
        """
        results = {}
        benchmark_suite = self.golden_texts_collection.get_benchmark_suite()
        
        for category, texts in benchmark_suite.items():
            metrics = []
            
            for golden_text in texts:
                # Compress the text
                result = self.compress(
                    golden_text.content, 
                    target_ratio=0.3, 
                    force_type=golden_text.content_type
                )
                
                # Analyze compression quality using golden text analysis
                quality_analysis = self.golden_texts_collection.analyze_compression_resistance(
                    golden_text, result
                )
                
                # Create comprehensive metrics
                metric = CompressionMetrics(
                    compression_ratio=result['actual_ratio'],
                    information_retention=quality_analysis['phrase_preservation'],
                    semantic_coherence=quality_analysis['semantic_coherence'],
                    readability_score=quality_analysis['quality_score'],
                    processing_time=result['processing_time']
                )
                metrics.append(metric)
            
            # Average metrics for category
            if metrics:
                avg_metric = CompressionMetrics(
                    compression_ratio=np.mean([m.compression_ratio for m in metrics]),
                    information_retention=np.mean([m.information_retention for m in metrics]),
                    semantic_coherence=np.mean([m.semantic_coherence for m in metrics]),
                    readability_score=np.mean([m.readability_score for m in metrics]),
                    processing_time=np.mean([m.processing_time for m in metrics])
                )
                results[category] = avg_metric
        
        return results


class TemporalCompressionHierarchy:
    """
    Handles compression across time scales: day→week→month→year→decade→lifetime.
    Essential for the life-logging vision.
    """
    
    def __init__(self, engine: AdaptiveCompressionEngine):
        self.engine = engine
        self.time_scales = ['day', 'week', 'month', 'year', 'decade', 'lifetime']
        self.compression_ratios = {
            'day': 1.0,      # Keep full detail
            'week': 0.5,     # 50% compression
            'month': 0.3,    # 70% compression
            'year': 0.15,    # 85% compression
            'decade': 0.08,  # 92% compression
            'lifetime': 0.03 # 97% compression
        }
    
    def compress_temporal_unit(self, activities: List[str], 
                              time_scale: str) -> Dict[str, Any]:
        """Compress activities for a specific time scale."""
        ratio = self.compression_ratios.get(time_scale, 0.3)
        combined_text = '\n'.join(activities)
        
        result = self.engine.compress(
            combined_text,
            target_ratio=ratio,
            force_type=ContentType.ACTIVITY_LOG
        )
        
        result['time_scale'] = time_scale
        result['activity_count'] = len(activities)
        
        return result
    
    def create_life_summary(self, all_activities: Dict[str, List[str]]) -> str:
        """
        Create a coherent life summary from temporal compressions.
        This is where we approach the philosophical - compressing a life.
        """
        summaries = []
        
        for time_scale in self.time_scales:
            if time_scale in all_activities:
                activities = all_activities[time_scale]
                compressed = self.compress_temporal_unit(activities, time_scale)
                summaries.append(f"[{time_scale.upper()}]\n{compressed['compressed']}")
        
        # Final philosophical compression
        life_text = '\n\n'.join(summaries)
        final_compression = self.engine.compress(
            life_text,
            target_ratio=0.05,  # Ultra compression for lifetime
            force_type=ContentType.PHILOSOPHICAL
        )
        
        return final_compression['compressed']


# Example usage and testing
if __name__ == "__main__":
    # Initialize engine
    engine = AdaptiveCompressionEngine()
    
    # Test philosophical text
    philosophical_text = """
    The essence of human existence lies not in the mere accumulation of experiences,
    but in the profound understanding of their interconnected nature. What defines us
    is not the quantity of our days, but the quality of our consciousness. Therefore,
    we must ask ourselves: what is the true meaning of a life well-lived? Is it found
    in achievement, in relationships, or in the simple act of being present? The answer,
    perhaps, is that meaning itself is not discovered but created through our choices.
    """
    
    print("=== Philosophical Compression Test ===")
    result = engine.compress(philosophical_text, target_ratio=0.3)
    print(f"Original: {result['original_length']} words")
    print(f"Compressed: {result['compressed_length']} words")
    print(f"Ratio: {result['actual_ratio']:.2%}")
    print(f"Compressed text:\n{result['compressed']}\n")
    
    # Test technical text
    technical_text = """
    The quicksort algorithm operates with an average time complexity of O(n log n).
    It works by selecting a pivot element and partitioning the array into two sub-arrays.
    Elements smaller than the pivot go to the left, larger elements go to the right.
    The function recursively sorts both sub-arrays. In the worst case, when the pivot
    is always the smallest or largest element, the complexity degrades to O(n^2).
    def quicksort(arr): return sorted(arr) if len(arr) <= 1 else quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x >= arr[0]])
    """
    
    print("=== Technical Compression Test ===")
    result = engine.compress(technical_text, target_ratio=0.4)
    print(f"Original: {result['original_length']} words")
    print(f"Compressed: {result['compressed_length']} words")
    print(f"Ratio: {result['actual_ratio']:.2%}")
    print(f"Compressed text:\n{result['compressed']}\n")
    
    # Test activity log
    activity_log = """
    2024-01-15 09:00 Started work on SUM project
    2024-01-15 09:30 Reviewed pull requests
    2024-01-15 10:00 Team standup meeting
    2024-01-15 10:30 Coding adaptive compression engine
    2024-01-15 11:00 Coding adaptive compression engine
    2024-01-15 11:30 Coding adaptive compression engine
    2024-01-15 12:00 Lunch break
    2024-01-15 13:00 Reviewed technical documentation
    2024-01-15 14:00 Implemented philosophical compression
    2024-01-15 15:00 Testing compression strategies
    2024-01-15 16:00 Writing unit tests
    2024-01-15 17:00 End of work day
    """
    
    print("=== Activity Log Compression Test ===")
    result = engine.compress(activity_log, target_ratio=0.3)
    print(f"Compressed activities:\n{result['compressed']}\n")
    
    # Benchmark all strategies
    print("=== Compression Benchmarks ===")
    benchmarks = engine.benchmark_compression()
    for content_type, metrics in benchmarks.items():
        print(f"{content_type.value}: ratio={metrics.compression_ratio:.2%}, "
              f"time={metrics.processing_time:.3f}s")