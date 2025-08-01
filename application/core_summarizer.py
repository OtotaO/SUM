"""
Core summarizer - generates core summaries with maximum compression.
Focused on essential information preservation.
"""

import logging
from typing import Dict, List, Set
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize

from domain.services.sentence_scorer import SentenceScorer
from domain.services.summary_builder import SummaryBuilder

logger = logging.getLogger(__name__)


class CoreSummarizer:
    """Generate core summary with maximum compression while preserving essential information."""
    
    def __init__(self, stop_words: Set[str]):
        self.stop_words = stop_words
        self.sentence_scorer = SentenceScorer(stop_words)
        self.summary_builder = SummaryBuilder()
    
    def generate_summary(
        self, 
        text: str, 
        target_density: float = 0.15,
        max_tokens: int = 50,
        ensure_completeness: bool = True
    ) -> str:
        """Generate a core summary that preserves essential information."""
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            return text
        
        # Score sentences semantically
        sentence_scores = self._score_sentences_semantically(sentences, text)
        
        # Calculate tokens per sentence
        sentence_tokens = {sentence: len(word_tokenize(sentence)) for sentence in sentences}
        
        # Build compressed summary
        summary = self._compress_to_summary(sentences, sentence_scores, sentence_tokens, max_tokens)
        
        # Validate completeness if requested
        if ensure_completeness:
            summary = self._validate_completeness(text, summary, target_density)
        
        return summary
    
    def _score_sentences_semantically(self, sentences: List[str], full_text: str) -> Dict[str, float]:
        """Score sentences using semantic importance."""
        scores = {}
        
        # Get document word frequencies
        doc_words = word_tokenize(full_text.lower())
        doc_word_freq = Counter(doc_words)
        
        for i, sentence in enumerate(sentences):
            sent_words = word_tokenize(sentence.lower())
            
            if len(sent_words) < 3:
                scores[sentence] = 0
                continue
            
            # Semantic importance factors
            freq_score = sum(doc_word_freq.get(word, 0) for word in sent_words) / len(sent_words)
            position_score = 1.0 - (0.2 * (i / len(sentences)))
            length_penalty = 1.0 if len(sent_words) <= 25 else 0.8
            
            # Importance boost
            importance_boost = self.sentence_scorer.score_by_importance(sentence)
            
            final_score = (freq_score * 0.4) + (position_score * 0.2) + (importance_boost * 0.4)
            scores[sentence] = final_score * length_penalty
        
        return scores
    
    def _compress_to_summary(
        self, 
        sentences: List[str], 
        scores: Dict[str, float], 
        sentence_tokens: Dict[str, int],
        max_tokens: int
    ) -> str:
        """Compress sentences to summary form."""
        sorted_sentences = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        selected_sentences = []
        current_tokens = 0
        
        for sentence, score in sorted_sentences:
            tokens = sentence_tokens[sentence]
            
            if current_tokens + tokens <= max_tokens:
                selected_sentences.append((sentence, sentences.index(sentence)))
                current_tokens += tokens
            elif not selected_sentences:  # Ensure at least one sentence
                # Truncate the best sentence if needed
                words = word_tokenize(sentence)
                truncated = ' '.join(words[:max_tokens-1])
                if len(words) > max_tokens-1:
                    truncated += '...'
                return truncated
        
        # Sort by original order
        selected_sentences.sort(key=lambda x: x[1])
        return ' '.join(sentence for sentence, _ in selected_sentences)
    
    def _validate_completeness(self, original: str, summary: str, target_density: float) -> str:
        """Validate that essential information is preserved."""
        # Simple validation: check if summary meets minimum density
        current_density = len(summary.split()) / len(original.split())
        
        if current_density < target_density * 0.5:
            # Summary is too short, might be missing essential information
            logger.warning(f"Summary density too low: {current_density:.2f} vs target {target_density}")
        
        return summary