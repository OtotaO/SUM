"""
Basic summarization engine - orchestrates simple frequency-based summarization.
Application layer that combines domain services.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from concurrent.futures import ThreadPoolExecutor
import os

from domain.interfaces.summarization_engine import SummarizationEngine
from domain.services.text_validator import TextValidator
from domain.services.word_frequency import WordFrequencyCalculator
from domain.services.sentence_scorer import SentenceScorer
from domain.services.summary_builder import SummaryBuilder
from domain.services.concept_extractor import ConceptExtractor
from infrastructure.nltk_manager import NLTKResourceManager

logger = logging.getLogger(__name__)


class BasicSummarizationEngine(SummarizationEngine):
    """Frequency-based extractive summarization with optimizations."""
    
    def __init__(self):
        """Initialize the engine with required resources."""
        self.nltk_manager = NLTKResourceManager()
        self.nltk_manager.initialize_resources(['punkt', 'stopwords'])
        
        self.stop_words = self.nltk_manager.get_english_stopwords()
        self.text_validator = TextValidator()
        self.word_freq_calculator = WordFrequencyCalculator(self.stop_words)
        self.sentence_scorer = SentenceScorer(self.stop_words)
        self.summary_builder = SummaryBuilder()
        self.concept_extractor = ConceptExtractor(self.stop_words)
        
        logger.info("BasicSummarizationEngine initialized successfully")
    
    def process_text(self, text: str, model_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process and summarize the input text."""
        if not self.text_validator.validate_input(text):
            return {'error': 'Empty or invalid text provided'}
        
        try:
            config = model_config or {}
            max_tokens = max(10, min(config.get('maxTokens', 100), 500))
            threshold = config.get('threshold', 0.3)
            
            sentences, words, word_freq = self._preprocess_text(text)
            
            if len(sentences) <= 2:
                return {
                    'summary': text, 
                    'sum': text, 
                    'tags': self.concept_extractor.extract_keywords(text)
                }
            
            # Calculate max frequency for normalization
            max_freq = max(word_freq.values()) if word_freq else 1
            
            # Score sentences
            sentence_scores = self._score_sentences(sentences, word_freq, max_freq)
            
            # Calculate tokens per sentence
            sentence_tokens = {sentence: len(word_tokenize(sentence)) for sentence in sentences}
            
            # Build summaries
            summary = self.summary_builder.build_summary(
                sentences, sentence_scores, sentence_tokens, max_tokens, threshold
            )
            
            condensed_max_tokens = max(10, min(config.get('maxTokens', 100) // 2, 100))
            condensed_summary = self.summary_builder.build_condensed_summary(
                sentences, sentence_scores, sentence_tokens, condensed_max_tokens
            )
            
            # Extract keywords
            tags = self.concept_extractor.extract_keywords(text)
            
            # Calculate metrics
            compression_ratio = len(word_tokenize(summary)) / len(words) if words else 1.0
            
            return {
                'summary': summary,
                'sum': condensed_summary,
                'tags': tags,
                'original_length': len(words),
                'compression_ratio': compression_ratio
            }
            
        except Exception as e:
            logger.error(f"Error during text processing: {str(e)}", exc_info=True)
            return {'error': f"Error during text processing: {str(e)}"}
    
    def _preprocess_text(self, text: str) -> Tuple[List[str], List[str], Dict[str, int]]:
        """Preprocess text for summarization."""
        sentences = sent_tokenize(text)
        words = []
        
        for sentence in sentences:
            sentence_words = word_tokenize(sentence.lower())
            words.extend(sentence_words)
        
        word_freq = self.word_freq_calculator.calculate_frequencies(words)
        
        return sentences, words, word_freq
    
    def _score_sentences(self, sentences: List[str], word_freq: Dict[str, int], max_freq: int) -> Dict[str, float]:
        """Score sentences based on word frequency."""
        if len(sentences) > 10:
            return self._score_sentences_parallel(sentences, word_freq, max_freq)
        return self.sentence_scorer.score_by_frequency(sentences, word_freq, max_freq)
    
    def _score_sentences_parallel(self, sentences: List[str], word_freq: Dict[str, int], max_freq: int) -> Dict[str, float]:
        """Score sentences in parallel for better performance."""
        sentence_scores = {}
        chunk_size = max(1, len(sentences) // (os.cpu_count() or 4))
        
        def score_chunk(chunk_sentences, start_idx):
            scorer = SentenceScorer(self.stop_words)
            # Create a temporary list with proper indices
            temp_sentences = [''] * len(sentences)
            for i, sent in enumerate(chunk_sentences):
                temp_sentences[start_idx + i] = sent
            return scorer.score_by_frequency(temp_sentences[start_idx:start_idx+len(chunk_sentences)], word_freq, max_freq)
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for i in range(0, len(sentences), chunk_size):
                chunk = sentences[i:i+chunk_size]
                futures.append(executor.submit(score_chunk, chunk, i))
            
            for future in futures:
                sentence_scores.update(future.result())
        
        return sentence_scores