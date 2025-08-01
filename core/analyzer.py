"""
ContentAnalyzer - Advanced Content Analysis

Carmack principles:
- Fast: Efficient algorithms with caching
- Simple: Clear analysis methods
- Clear: Obvious data flow
- Bulletproof: Graceful degradation

Author: ototao (optimized with Claude Code)
License: Apache License 2.0
"""

import logging
import re
from typing import List, Dict, Set, Optional, Tuple
from functools import lru_cache
from collections import Counter, defaultdict
import threading

logger = logging.getLogger(__name__)


class ContentAnalyzer:
    """
    High-performance content analysis engine.
    
    Features:
    - Concept extraction
    - Importance scoring
    - Keyword analysis
    - Topic identification
    - Sentiment analysis (basic)
    """
    
    def __init__(self):
        """Initialize analyzer with intelligent defaults."""
        self._importance_patterns = None
        self._concept_weights = None
        self._lock = threading.Lock()
    
    @property
    def importance_patterns(self) -> Dict[str, float]:
        """Lazy-loaded importance pattern weights."""
        if self._importance_patterns is None:
            with self._lock:
                if self._importance_patterns is None:
                    self._importance_patterns = {
                        # High importance markers
                        r'\b(critical|crucial|essential|vital|key|important)\b': 0.8,
                        r'\b(fundamental|primary|main|core|central)\b': 0.7,
                        r'\b(significant|major|notable|remarkable)\b': 0.6,
                        r'\b(therefore|thus|consequently|hence)\b': 0.5,
                        r'\b(conclusion|summary|result|finding)\b': 0.6,
                        
                        # Medium importance markers  
                        r'\b(additionally|furthermore|moreover)\b': 0.3,
                        r'\b(however|nevertheless|nonetheless)\b': 0.4,
                        r'\b(example|instance|case|illustration)\b': 0.3,
                        
                        # Low importance markers
                        r'\b(possibly|maybe|perhaps|might|could)\b': -0.2,
                        r'\b(seems|appears|looks|sounds)\b': -0.1,
                    }
        return self._importance_patterns
    
    @property
    def concept_weights(self) -> Dict[str, float]:
        """Lazy-loaded concept importance weights."""
        if self._concept_weights is None:
            with self._lock:
                if self._concept_weights is None:
                    self._concept_weights = {
                        # High-value concepts
                        'algorithm': 0.9, 'system': 0.8, 'method': 0.8, 'process': 0.7,
                        'framework': 0.8, 'approach': 0.7, 'strategy': 0.7, 'technique': 0.6,
                        'principle': 0.9, 'concept': 0.8, 'theory': 0.8, 'model': 0.7,
                        'analysis': 0.7, 'evaluation': 0.6, 'assessment': 0.6, 'research': 0.6,
                        'solution': 0.8, 'problem': 0.7, 'issue': 0.6, 'challenge': 0.6,
                        'result': 0.7, 'outcome': 0.6, 'conclusion': 0.8, 'finding': 0.7,
                        'data': 0.7, 'information': 0.6, 'knowledge': 0.8, 'insight': 0.8,
                        'performance': 0.7, 'efficiency': 0.7, 'optimization': 0.8, 'improvement': 0.6,
                        
                        # Domain-specific high-value terms
                        'artificial': 0.8, 'intelligence': 0.8, 'machine': 0.7, 'learning': 0.8,
                        'neural': 0.7, 'network': 0.7, 'deep': 0.6, 'classification': 0.6,
                        'optimization': 0.8, 'algorithm': 0.9, 'computation': 0.7, 'processing': 0.6,
                        
                        # Lower value but still relevant
                        'implementation': 0.5, 'application': 0.5, 'usage': 0.4, 'utilization': 0.4,
                        'development': 0.5, 'design': 0.6, 'architecture': 0.7, 'structure': 0.5,
                    }
        return self._concept_weights
    
    @lru_cache(maxsize=256)
    def extract_keywords(self, text: str, count: int = 10) -> List[str]:
        """
        Extract top keywords using TF-IDF-like scoring.
        
        Args:
            text: Input text
            count: Number of keywords to return
            
        Returns:
            List of top keywords
        """
        if not text or not text.strip():
            return []
        
        # Tokenize and clean
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stopwords
        stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word for word in words if word not in stopwords]
        
        if not words:
            return []
        
        # Calculate word frequencies
        word_freq = Counter(words)
        
        # Apply concept weighting
        weighted_scores = {}
        for word, freq in word_freq.items():
            base_score = freq / len(words)  # Normalized frequency
            concept_weight = self.concept_weights.get(word, 0.1)  # Default low weight
            weighted_scores[word] = base_score * (1 + concept_weight)
        
        # Return top keywords
        top_keywords = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, score in top_keywords[:count]]
    
    def extract_concepts(self, text: str, max_concepts: int = 10) -> List[str]:
        """
        Extract key concepts from text.
        
        Args:
            text: Input text
            max_concepts: Maximum concepts to return
            
        Returns:
            List of key concepts
        """
        concepts = []
        text_lower = text.lower()
        
        # Find multi-word concepts (phrases)
        concept_patterns = [
            r'\b([a-z]+ (?:algorithm|system|method|process|framework|approach|strategy))\b',
            r'\b([a-z]+ (?:analysis|evaluation|assessment|research|study))\b',
            r'\b([a-z]+ (?:model|theory|principle|concept|technique))\b',
            r'\b([a-z]+ (?:solution|problem|issue|challenge))\b',
            r'\b(artificial intelligence|machine learning|deep learning|neural network)\b',
            r'\b(data (?:analysis|processing|mining|science))\b',
            r'\b(natural language (?:processing|understanding))\b',
        ]
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, text_lower)
            concepts.extend(matches)
        
        # Add single-word high-value concepts
        keywords = self.extract_keywords(text, count=max_concepts * 2)
        high_value_keywords = [
            word for word in keywords 
            if self.concept_weights.get(word, 0) > 0.6
        ]
        concepts.extend(high_value_keywords)
        
        # Remove duplicates and limit
        unique_concepts = list(dict.fromkeys(concepts))  # Preserve order
        return unique_concepts[:max_concepts]
    
    def calculate_sentence_importance(self, sentences: List[str], concepts: List[str]) -> List[float]:
        """
        Calculate importance scores for sentences.
        
        Args:
            sentences: List of sentences
            concepts: List of key concepts
            
        Returns:
            List of importance scores (0-1)
        """
        if not sentences:
            return []
        
        scores = []
        concept_set = set(word.lower() for word in concepts)
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            score = 0.0
            
            # Base score from concept presence
            sentence_words = set(re.findall(r'\b[a-z]+\b', sentence_lower))
            concept_overlap = len(sentence_words.intersection(concept_set))
            if sentence_words:
                score += (concept_overlap / len(sentence_words)) * 0.4
            
            # Pattern-based importance scoring
            for pattern, weight in self.importance_patterns.items():
                if re.search(pattern, sentence_lower):
                    score += weight * 0.1
            
            # Position-based scoring (earlier sentences slightly favored)
            position_score = 1.0 - (0.1 * i / len(sentences))
            score *= position_score
            
            # Length normalization (avoid very short/long sentences)
            word_count = len(sentence.split())
            if 8 <= word_count <= 35:
                score *= 1.1
            elif word_count < 8:
                score *= 0.8
            elif word_count > 35:
                score *= 0.9
            
            scores.append(min(score, 1.0))  # Cap at 1.0
        
        return scores
    
    def analyze_topic_distribution(self, text: str) -> Dict[str, float]:
        """
        Analyze topic distribution in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of topics and their strength scores
        """
        topic_keywords = {
            'technology': ['algorithm', 'system', 'software', 'computer', 'digital', 'technology', 'programming', 'code'],
            'science': ['research', 'study', 'analysis', 'experiment', 'hypothesis', 'theory', 'scientific', 'method'],
            'business': ['company', 'market', 'customer', 'product', 'service', 'business', 'strategy', 'revenue'],
            'education': ['learning', 'student', 'teacher', 'education', 'knowledge', 'skill', 'training', 'course'],
            'health': ['health', 'medical', 'patient', 'treatment', 'disease', 'therapy', 'clinical', 'medicine'],
            'finance': ['money', 'financial', 'investment', 'market', 'economic', 'profit', 'cost', 'budget'],
        }
        
        text_lower = text.lower()
        word_set = set(re.findall(r'\b[a-z]+\b', text_lower))
        
        topic_scores = {}
        for topic, keywords in topic_keywords.items():
            keyword_set = set(keywords)
            overlap = len(word_set.intersection(keyword_set))
            if keyword_set:
                topic_scores[topic] = overlap / len(keyword_set)
        
        # Normalize scores
        max_score = max(topic_scores.values()) if topic_scores.values() else 1
        if max_score > 0:
            topic_scores = {topic: score / max_score for topic, score in topic_scores.items()}
        
        # Return only topics with significant presence
        return {topic: score for topic, score in topic_scores.items() if score > 0.1}
    
    def detect_sentiment(self, text: str) -> Dict[str, float]:
        """
        Basic sentiment analysis.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome',
            'positive', 'beneficial', 'effective', 'successful', 'improve', 'better',
            'advantage', 'benefit', 'gain', 'progress', 'achievement', 'success'
        }
        
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'negative', 'harmful', 'damaging',
            'problem', 'issue', 'difficulty', 'challenge', 'fail', 'failure', 'error',
            'disadvantage', 'loss', 'decline', 'decrease', 'worse', 'worsen'
        }
        
        neutral_words = {
            'analysis', 'study', 'research', 'method', 'approach', 'system', 'process',
            'information', 'data', 'result', 'finding', 'conclusion', 'summary'
        }
        
        words = re.findall(r'\b[a-z]+\b', text.lower())
        if not words:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        neu_count = sum(1 for word in words if word in neutral_words)
        
        total = len(words)
        return {
            'positive': pos_count / total,
            'negative': neg_count / total,
            'neutral': max(neu_count / total, 1.0 - (pos_count + neg_count) / total)
        }
    
    def extract_named_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Basic named entity recognition using patterns.
        
        Args:
            text: Input text
            
        Returns:
            List of (entity, type) tuples
        """
        entities = []
        
        # Simple pattern-based entity recognition
        patterns = {
            'PERSON': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            'ORGANIZATION': r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|Corporation|Institute|University)\b',
            'LOCATION': r'\b[A-Z][a-z]+ (?:City|State|Country|Avenue|Street|Road|Boulevard)?\b',
            'TECHNOLOGY': r'\b[A-Z][a-z]*(?:AI|ML|API|SDK|CPU|GPU|RAM|OS|DB)\b',
            'DATE': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            'NUMBER': r'\b\d+\.?\d*\s*(?:%|percent|million|billion|thousand)?\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match.strip()) > 2:  # Filter very short matches
                    entities.append((match.strip(), entity_type))
        
        return entities
    
    def get_content_complexity(self, text: str) -> Dict[str, float]:
        """
        Analyze content complexity.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with complexity metrics
        """
        if not text or not text.strip():
            return {'lexical': 0.0, 'syntactic': 0.0, 'semantic': 0.0, 'overall': 0.0}
        
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        # Lexical complexity (vocabulary diversity)
        unique_words = set(word.lower() for word in words if word.isalpha())
        lexical_complexity = len(unique_words) / len(words) if words else 0
        
        # Syntactic complexity (average sentence length)
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        syntactic_complexity = min(avg_sentence_length / 20, 1.0)  # Normalize to 0-1
        
        # Semantic complexity (concept density)
        concepts = self.extract_concepts(text)
        semantic_complexity = len(concepts) / len(words) * 100 if words else 0
        semantic_complexity = min(semantic_complexity, 1.0)
        
        # Overall complexity
        overall = (lexical_complexity + syntactic_complexity + semantic_complexity) / 3
        
        return {
            'lexical': lexical_complexity,
            'syntactic': syntactic_complexity,
            'semantic': semantic_complexity,
            'overall': overall
        }
    
    def clear_cache(self):
        """Clear LRU caches."""
        self.extract_keywords.cache_clear()
        logger.info("ContentAnalyzer caches cleared")