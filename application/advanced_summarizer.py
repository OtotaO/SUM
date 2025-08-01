"""
Advanced summarization engine - comprehensive text analysis and summarization.
Orchestrates multiple analysis techniques.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

from domain.interfaces.summarization_engine import SummarizationEngine
from domain.services.text_validator import TextValidator
from domain.services.word_frequency import WordFrequencyCalculator
from domain.services.sentence_scorer import SentenceScorer
from domain.services.summary_builder import SummaryBuilder
from domain.services.concept_extractor import ConceptExtractor
from infrastructure.nltk_manager import NLTKResourceManager

logger = logging.getLogger(__name__)


class AdvancedSummarizationEngine(SummarizationEngine):
    """Comprehensive text analysis with multiple summarization approaches."""
    
    def __init__(self, num_tags: int = 5):
        """Initialize the advanced summarizer."""
        self.nltk_manager = NLTKResourceManager()
        self.nltk_manager.initialize_resources([
            'punkt', 'stopwords', 'averaged_perceptron_tagger',
            'maxent_ne_chunker', 'words', 'vader_lexicon'
        ])
        
        self.stop_words = self.nltk_manager.get_english_stopwords()
        self.sentiment_analyzer = self.nltk_manager.get_sentiment_analyzer()
        self.num_tags = num_tags
        
        # Initialize services
        self.text_validator = TextValidator()
        self.word_freq_calculator = WordFrequencyCalculator(self.stop_words)
        self.sentence_scorer = SentenceScorer(self.stop_words)
        self.summary_builder = SummaryBuilder()
        self.concept_extractor = ConceptExtractor(self.stop_words)
        
        logger.info("AdvancedSummarizationEngine initialized successfully")
    
    def process_text(self, text: str, model_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process text and generate comprehensive analysis results."""
        if not self.text_validator.validate_input(text):
            return {'error': 'Invalid input'}
        
        try:
            if len(text.split()) < 10:
                return {
                    'sum': text,
                    'summary': text,
                    'tags': self.concept_extractor.extract_keywords(text, self.num_tags)
                }
            
            config = model_config or {}
            
            # Generate summaries
            tag_summary = self.concept_extractor.extract_keywords(text, self.num_tags)
            sentence_summary = self.generate_sentence_summary(text, config)
            condensed_summary = self._generate_condensed_summary(text, config)
            
            result = {
                'tags': tag_summary,
                'sum': condensed_summary,
                'summary': sentence_summary,
            }
            
            # Add optional analysis
            if config.get('include_analysis', False):
                result.update({
                    'entities': self.identify_entities(text),
                    'main_concept': tag_summary[0] if tag_summary else "",
                    'sentiment': self.sentiment_analysis(text),
                    'keywords': tag_summary,
                    'language': self.detect_language(text)
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}", exc_info=True)
            return {'error': f'Processing failed: {str(e)}'}
    
    def identify_entities(self, text: str) -> List[Tuple[str, str]]:
        """Identify named entities in the text."""
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        entities = ne_chunk(tagged)
        
        result = []
        for chunk in entities:
            if hasattr(chunk, 'label'):
                entity = ' '.join(c[0] for c in chunk)
                result.append((entity, chunk.label()))
        return result
    
    def sentiment_analysis(self, text: str) -> str:
        """Analyze the sentiment of the text."""
        scores = self.sentiment_analyzer.polarity_scores(text)
        if scores['compound'] >= 0.05:
            return 'Positive'
        elif scores['compound'] <= -0.05:
            return 'Negative'
        return 'Neutral'
    
    def detect_language(self, text: str) -> str:
        """Simple language detection based on common words."""
        language_words = {
            'es': {'el', 'la', 'los', 'las', 'es', 'son', 'está', 'hola', 'mundo'},
            'fr': {'le', 'la', 'les', 'est', 'sont', 'bonjour', 'monde'},
            'de': {'der', 'die', 'das', 'ist', 'sind', 'hallo', 'welt'}
        }
        
        words = set(word_tokenize(text.lower()))
        
        for lang, markers in language_words.items():
            if len(words.intersection(markers)) > 0:
                return lang
        
        return 'en'
    
    def generate_sentence_summary(self, text: str, config: Dict[str, Any]) -> str:
        """Generate a summary by extracting key sentences."""
        num_sentences = config.get('num_sentences', 3)
        max_tokens = config.get('maxTokens', 200)
        
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            return text
        
        # Calculate word frequencies
        words = word_tokenize(text.lower())
        word_freq = self.word_freq_calculator.calculate_frequencies(words)
        max_freq = max(word_freq.values()) if word_freq else 1
        
        # Score sentences
        scores = self._score_sentences_with_importance(sentences, word_freq, max_freq)
        
        # Extract top sentences
        summary_sentences = self._extract_summary_sentences(sentences, scores, num_sentences)
        
        summary = ' '.join(summary_sentences)
        summary_tokens = word_tokenize(summary)
        if len(summary_tokens) > max_tokens:
            return ' '.join(summary_tokens[:max_tokens]) + '...'
        
        return summary
    
    def _generate_condensed_summary(self, text: str, config: Dict[str, Any]) -> str:
        """Generate a very concise summary."""
        max_tokens = config.get('maxTokens', 100) // 2
        
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            return text
        
        # Calculate word frequencies
        words = word_tokenize(text.lower())
        word_freq = self.word_freq_calculator.calculate_frequencies(words)
        max_freq = max(word_freq.values()) if word_freq else 1
        
        # Score sentences
        scores = self._score_sentences_with_importance(sentences, word_freq, max_freq)
        
        # Get top sentence
        top_sentence = max(scores.items(), key=lambda x: x[1])[0]
        
        tokens = word_tokenize(top_sentence)
        if len(tokens) > max_tokens:
            return ' '.join(tokens[:max_tokens]) + '...'
        
        return top_sentence
    
    def _score_sentences_with_importance(self, sentences: List[str], word_freq: Dict[str, int], max_freq: int) -> Dict[str, float]:
        """Score sentences combining frequency and importance."""
        freq_scores = self.sentence_scorer.score_by_frequency(sentences, word_freq, max_freq)
        
        # Add importance scoring
        for sentence in sentences:
            importance_boost = self.sentence_scorer.score_by_importance(sentence)
            freq_scores[sentence] = freq_scores.get(sentence, 0) * 0.7 + importance_boost * 0.3
        
        return freq_scores
    
    def _extract_summary_sentences(self, sentences: List[str], scores: Dict[str, float], num_sentences: int) -> List[str]:
        """Extract top scoring sentences while preserving order."""
        indexed_scores = [(i, sentence, scores[sentence]) for i, sentence in enumerate(sentences)]
        top_indexed = sorted(indexed_scores, key=lambda x: x[2], reverse=True)[:num_sentences]
        ordered_sentences = sorted(top_indexed, key=lambda x: x[0])
        
        return [sentence for _, sentence, _ in ordered_sentences]