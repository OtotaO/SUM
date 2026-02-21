"""
summarizer.py - Integration of topic modeling and summarization

This module connects the SUM summarization engine with the topic modeling
capabilities, providing a comprehensive text analysis solution.

Design principles:
- Clean integration (Fowler architecture)
- Performance optimization (Knuth)
- Code readability (Torvalds/van Rossum)
- Robust error handling (Stroustrup)
- Security focus (Schneier)

Author: ototao
License: Apache License 2.0
"""

import logging
import time
from typing import List, Dict, Any, Optional
from collections import Counter

from nltk.tokenize import sent_tokenize, word_tokenize

# Import our custom modules
from .topic_modeling import TopicModeler
from summarization_engine import SimpleSUM, MagnumOpusSUM

# Configure logging
logger = logging.getLogger(__name__)

class Summarizer:
    """
    Summarize text data using topic modeling and the SUM engine.
    
    This class combines topic modeling with extractive summarization
    to provide a comprehensive text analysis solution.
    """

    def __init__(self, data_file: Optional[str] = None, 
                data: Any = None, 
                num_topics: int = 5, 
                algorithm: str = 'lda',
                advanced: bool = False):
        """
        Initialize the summarizer with data and models.
        
        Args:
            data_file: Path to data file (optional)
            data: Pre-loaded data (optional)
            num_topics: Number of topics to model
            algorithm: Topic modeling algorithm ('lda', 'nmf', or 'lsa')
            advanced: Whether to use the advanced summarizer
        """
        if data_file is None and data is None:
            raise ValueError("Either data_file or data must be provided")
            
        # Initialize components
        from utils.data_loader import DataLoader
        self.data_loader = DataLoader(data_file=data_file, data=data)
        self.topic_modeler = TopicModeler(n_topics=num_topics, algorithm=algorithm)
        
        # Initialize appropriate summarizer
        self.summarizer = MagnumOpusSUM() if advanced else SimpleSUM()
        
        engine_type = "advanced" if advanced else "simple"
        logger.info(f"Initialized Summarizer with {algorithm} topic modeling and {engine_type} summarization")

    def analyze(self, max_tokens: int = 200, 
               include_topics: bool = True, 
               include_analysis: bool = False) -> Dict[str, Any]:
        """
        Analyze data using topic modeling and summarization.
        
        Args:
            max_tokens: Maximum tokens in generated summaries
            include_topics: Whether to include topic analysis
            include_analysis: Whether to include additional text analysis
            
        Returns:
            Dictionary with analysis results
        """
        try:
            start_time = time.time()
            
            # Load and extract text content
            data = self.data_loader.load_data()
            text = self._extract_text(data)
                
            # Process text with summarizer
            config = {
                'maxTokens': max_tokens,
                'include_analysis': include_analysis
            }
            
            summary_result = self.summarizer.process_text(text, config)
            
            # Add topic analysis if requested and text is substantial
            if include_topics and len(text.split()) > 50:
                topic_result = self._analyze_topics(text)
                summary_result['topics'] = topic_result
            
            # Add metadata
            metadata = self.data_loader.get_metadata()
            summary_result['metadata'] = metadata
            summary_result['processing_time'] = time.time() - start_time
            
            return summary_result
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}", exc_info=True)
            return {'error': f"Analysis failed: {str(e)}"}

    def _extract_text(self, data: Any) -> str:
        """Extract text content from various data structures."""
        if isinstance(data, str):
            return data
        elif isinstance(data, list):
            if all(isinstance(item, str) for item in data):
                return " ".join(data)
            else:
                return str(data)
        elif isinstance(data, dict):
            if 'content' in data:
                return data['content']
            elif 'text' in data:
                return data['text']
            else:
                return str(data)
        else:
            return str(data)

    def _analyze_topics(self, text: str) -> Dict[str, Any]:
        """Perform topic analysis on the text."""
        # Split text into paragraphs for document-level analysis
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # For very short texts or single paragraphs, use sentences
        if len(paragraphs) < 3:
            paragraphs = [s for s in sent_tokenize(text) if len(s.split()) > 5]
        
        # If still not enough documents, fall back to original text
        if len(paragraphs) < 3:
            paragraphs = [text]
            
        # Fit topic model
        self.topic_modeler.fit(paragraphs)
        
        # Get topic summary
        return self.topic_modeler.get_topics_summary()

    def generate_topic_based_summary(self, text: str, num_sentences: int = 3) -> Dict[str, Any]:
        """
        Generate a summary based on topic relevance.
        
        This method extracts sentences that best represent the main topics.
        
        Args:
            text: Text to summarize
            num_sentences: Number of sentences to include
            
        Returns:
            Dictionary with topic-based summary
        """
        # Extract sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return {'summary': text, 'topics': []}
            
        # Create document-term matrix for sentences
        self.topic_modeler.vectorizer.fit([text])
        
        # Fit topic model on the full text
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) < 3:
            paragraphs = [text]
            
        self.topic_modeler.fit(paragraphs)
        topics = self.topic_modeler.get_topics()
        
        # Score sentences based on topic relevance
        sentence_scores = self._score_sentences_by_topics(sentences, topics)
        
        # Select top sentences in original order
        ranked_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        selected_sentences = [sentence for sentence, _ in ranked_sentences[:num_sentences]]
        selected_with_position = [(s, sentences.index(s)) for s in selected_sentences]
        selected_with_position.sort(key=lambda x: x[1])
        
        # Create summary
        summary = ' '.join([s for s, _ in selected_with_position])
        
        return {
            'summary': summary,
            'topics': [', '.join(topic) for topic in topics]
        }
        
    def _score_sentences_by_topics(self, sentences: List[str], topics: List[List[str]]) -> Dict[str, float]:
        """Score sentences based on their relevance to identified topics."""
        scores = {}
        
        for i, sentence in enumerate(sentences):
            words = word_tokenize(sentence.lower())
            relevant_words = [w for w in words if len(w) > 2 and w.isalnum()]
            
            if not relevant_words:
                scores[sentence] = 0
                continue
                
            topic_score = 0
            for topic in topics:
                topic_words_in_sentence = sum(1 for word in relevant_words if word in topic)
                topic_score += topic_words_in_sentence / max(1, len(relevant_words))
                
            # Position bias - earlier sentences get slight preference
            position_score = 1.0 - (0.05 * (i / len(sentences)))
            scores[sentence] = topic_score * position_score
            
        return scores
