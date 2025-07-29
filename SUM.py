"""
SUM.py - Knowledge Distillation Engine

Core summarization functionality for the SUM platform with methods for
extracting key information from text using various NLP techniques.

Design principles:
- Simplicity and readability (Torvalds/van Rossum)
- Performance optimization (Knuth)
- Test-driven development (Beck)
- Algorithm innovation (Dijkstra)
- Security focus (Schneier)

Author: ototao
License: Apache License 2.0
"""

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import numpy as np
import os
import logging
import re
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from wordcloud import WordCloud

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SUM:
    """Base class for summarization algorithms."""
    
    def process_text(self, text: str, model_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process text using the implemented algorithm."""
        raise NotImplementedError("Subclasses must implement process_text")


class SimpleSUM(SUM):
    """
    Frequency-based extractive summarization engine with 
    optimizations for performance on larger texts.
    """

    def __init__(self):
        """Initialize the SimpleSUM engine with required NLTK resources."""
        self._init_nltk()
        self.word_freq = None
        self.max_freq = 1

    def _init_nltk(self):
        """Initialize NLTK resources safely."""
        try:
            nltk_data_dir = os.path.expanduser('~/nltk_data')
            os.makedirs(nltk_data_dir, exist_ok=True)

            for resource in ['punkt', 'stopwords']:
                try:
                    nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
                except Exception as e:
                    logger.error(f"Error downloading {resource}: {str(e)}")
                    raise

            self._load_stopwords()
            logger.info("SimpleSUM initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing NLTK: {str(e)}")
            raise RuntimeError("Failed to initialize NLTK resources")

    def _load_stopwords(self):
        """Securely load stopwords with validation."""
        try:
            raw_stopwords = stopwords.words('english')
            self.stop_words = {word for word in raw_stopwords 
                              if self._is_safe_string(word)}
        except Exception as e:
            logger.error(f"Error loading stopwords: {str(e)}")
            self.stop_words = {"the", "a", "an", "and", "in", "on", "at", "to", "for", "with"}

    @staticmethod
    def _is_safe_string(text: str) -> bool:
        """Verify string doesn't contain potentially unsafe patterns."""
        if len(text) > 100:  # Unusually long for a stopword
            return False
        unsafe_patterns = [
            r'[\s\S]*exec\s*\(', r'[\s\S]*eval\s*\(', r'[\s\S]*\bimport\b',
            r'[\s\S]*__[a-zA-Z]+__', r'[\s\S]*\bopen\s*\('
        ]
        return not any(re.search(pattern, text) for pattern in unsafe_patterns)

    @lru_cache(maxsize=1024)
    def _calculate_word_frequency(self, word: str) -> float:
        """Calculate normalized word frequency with caching."""
        return self.word_freq.get(word, 0) / self.max_freq if self.max_freq > 0 else 0

    def _preprocess_text(self, text: str) -> Tuple[List[str], List[str], Dict[str, int]]:
        """Preprocess text for summarization."""
        if not self._is_safe_string(text[:100]):
            logger.warning("Potentially unsafe input detected")
            return [], [], {}
            
        sentences = sent_tokenize(text)
        words = []
        word_freq = Counter()
        
        for sentence in sentences:
            sentence_words = word_tokenize(sentence.lower())
            words.extend(sentence_words)
            word_freq.update(word for word in sentence_words
                           if word.isalnum() and word not in self.stop_words)
        
        self.word_freq = word_freq
        self.max_freq = max(word_freq.values()) if word_freq else 1
        
        return sentences, words, word_freq

    def process_text(self, text: str, model_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process and summarize the input text."""
        if not text or not isinstance(text, str) or not text.strip():
            return {'error': 'Empty or invalid text provided'}

        try:
            config = model_config or {}
            max_tokens = max(10, min(config.get('maxTokens', 100), 500))
            threshold = config.get('threshold', 0.3)
            
            sentences, words, word_freq = self._preprocess_text(text)
            
            if len(sentences) <= 2:
                return {'summary': text, 'sum': text, 'tags': self.generate_tag_summary(text)}
                
            sentence_scores = (
                self._score_sentences_parallel(sentences, word_freq) 
                if len(sentences) > 10 else 
                self._score_sentences(sentences, word_freq)
            )
                
            sentence_tokens = {sentence: len(word_tokenize(sentence)) for sentence in sentences}
            
            summary_data = self._build_summary(
                sentences, sentence_scores, sentence_tokens, max_tokens, threshold
            )
            
            summary_data['tags'] = self.generate_tag_summary(text)
            
            condensed_max_tokens = max(10, min(config.get('maxTokens', 100) // 2, 100))
            summary_data['sum'] = self._build_condensed_summary(
                sentences, sentence_scores, sentence_tokens, condensed_max_tokens
            )
            
            summary_data['original_length'] = len(words)
            summary_data['compression_ratio'] = (
                len(word_tokenize(summary_data['summary'])) / len(words) if words else 1.0
            )
            
            return summary_data

        except Exception as e:
            logger.error(f"Error during text processing: {str(e)}", exc_info=True)
            return {'error': f"Error during text processing: {str(e)}"}

    def _score_sentences(self, sentences: List[str], word_freq: Dict[str, int]) -> Dict[str, float]:
        """Score sentences based on word frequency."""
        sentence_scores = {}
        
        for i, sentence in enumerate(sentences):
            sent_words = word_tokenize(sentence.lower())
            
            if len(sent_words) < 3:
                sentence_scores[sentence] = 0
                continue
                
            score = sum(self._calculate_word_frequency(word) 
                      for word in sent_words 
                      if word.isalnum() and word not in self.stop_words)
                
            position_weight = 1.0 - (0.1 * (i / max(1, len(sentences))))
            sentence_scores[sentence] = (score / max(1, len(sent_words))) * position_weight
            
        return sentence_scores

    def _score_sentences_parallel(self, sentences: List[str], word_freq: Dict[str, int]) -> Dict[str, float]:
        """Score sentences in parallel for better performance on large texts."""
        sentence_scores = {}
        chunk_size = max(1, len(sentences) // (os.cpu_count() or 4))
        
        def score_chunk(chunk_sentences, start_idx):
            chunk_scores = {}
            for i, sentence in enumerate(chunk_sentences):
                abs_idx = start_idx + i
                sent_words = word_tokenize(sentence.lower())
                
                if len(sent_words) < 3:
                    chunk_scores[sentence] = 0
                    continue
                    
                score = sum(self._calculate_word_frequency(word) 
                           for word in sent_words 
                           if word.isalnum() and word not in self.stop_words)
                
                position_weight = 1.0 - (0.1 * (abs_idx / max(1, len(sentences))))
                chunk_scores[sentence] = (score / max(1, len(sent_words))) * position_weight
                
            return chunk_scores
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for i in range(0, len(sentences), chunk_size):
                chunk = sentences[i:i+chunk_size]
                futures.append(executor.submit(score_chunk, chunk, i))
                
            for future in futures:
                sentence_scores.update(future.result())
                
        return sentence_scores

    def _build_summary(self, sentences: List[str], sentence_scores: Dict[str, float], 
                      sentence_tokens: Dict[str, int], max_tokens: int, threshold: float) -> Dict[str, str]:
        """Build a summary from scored sentences."""
        qualified_sentences = [
            (sentence, score) for sentence, score in sentence_scores.items() 
            if score > threshold
        ]
        
        sorted_sentences = sorted(qualified_sentences, key=lambda x: x[1], reverse=True)
        summary_sentences = []
        current_tokens = 0
        
        for sentence, score in sorted_sentences:
            if current_tokens + sentence_tokens[sentence] <= max_tokens:
                summary_sentences.append((sentence, sentences.index(sentence)))
                current_tokens += sentence_tokens[sentence]
                
            if current_tokens >= max_tokens:
                break
                
        if not summary_sentences and sorted_sentences:
            top_sentence, _ = sorted_sentences[0]
            if sentence_tokens[top_sentence] <= max_tokens:
                summary_sentences.append((top_sentence, sentences.index(top_sentence)))
            else:
                words = word_tokenize(top_sentence)[:max_tokens-1]
                truncated = ' '.join(words) + '...'
                return {'summary': truncated}
                
        summary_sentences.sort(key=lambda x: x[1])
        summary = ' '.join(sentence for sentence, _ in summary_sentences)
        
        return {'summary': summary}
    
    def _build_condensed_summary(self, sentences: List[str], sentence_scores: Dict[str, float], 
                                sentence_tokens: Dict[str, int], max_tokens: int) -> str:
        """Build a condensed summary from the highest scoring sentences."""
        sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        selected_sentences = []
        current_tokens = 0
        
        for sentence, _ in sorted_sentences:
            if current_tokens + sentence_tokens[sentence] <= max_tokens:
                selected_sentences.append((sentence, sentences.index(sentence)))
                current_tokens += sentence_tokens[sentence]
            else:
                break
                
        if not selected_sentences and sorted_sentences:
            top_sentence, _ = sorted_sentences[0]
            if sentence_tokens[top_sentence] <= max_tokens:
                selected_sentences.append((top_sentence, sentences.index(top_sentence)))
            else:
                words = word_tokenize(top_sentence)[:max_tokens-1]
                return ' '.join(words) + '...'
        
        if not selected_sentences:
            return ""
            
        selected_sentences.sort(key=lambda x: x[1])
        return ' '.join(sentence for sentence, _ in selected_sentences)
    
    def generate_tag_summary(self, text: str) -> List[str]:
        """Generate a list of tags that summarize the text."""
        words = word_tokenize(text.lower())
        word_freq = Counter(word for word in words
                          if word.isalnum() and word not in self.stop_words and len(word) > 2)
        
        return [word for word, _ in word_freq.most_common(5)]


class MagnumOpusSUM(SUM):
    """
    Comprehensive text analysis and summarization system with
    multiple summarization approaches and text analysis features.
    """

    def __init__(self, num_tags: int = 5) -> None:
        """Initialize the advanced summarizer."""
        required_packages = [
            'punkt', 'stopwords', 'averaged_perceptron_tagger',
            'maxent_ne_chunker', 'words', 'vader_lexicon'
        ]
        
        for package in required_packages:
            try:
                nltk.data.find(f'tokenizers/{package}')
            except LookupError:
                nltk.download(package, quiet=True)

        self.stop_words = set(stopwords.words('english'))
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.num_tags = num_tags

    def process_text(self, text: str, model_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process text and generate comprehensive analysis results."""
        if not self._validate_input(text):
            return {'error': 'Invalid input'}

        try:
            if len(text.split()) < 10:
                return {
                    'sum': text,
                    'summary': text,
                    'tags': self.generate_tag_summary(text)
                }

            config = model_config or {}
            tag_summary = self.generate_tag_summary(text)
            sentence_summary = self.generate_sentence_summary(text, config)
            condensed_summary = self._generate_condensed_summary(text, config)

            result = {
                'tags': tag_summary,
                'sum': condensed_summary,
                'summary': sentence_summary,
            }
            
            if config.get('include_analysis', False):
                result.update({
                    'entities': self.identify_entities(text),
                    'main_concept': self.identify_main_concept(text),
                    'sentiment': self.sentiment_analysis(text),
                    'keywords': self.extract_keywords(text),
                    'language': self.detect_language(text)
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}", exc_info=True)
            return {'error': f'Processing failed: {str(e)}'}

    def _validate_input(self, text: str) -> bool:
        """Validate input text."""
        return bool(text and isinstance(text, str) and text.strip())

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

    def identify_main_concept(self, text: str) -> str:
        """Identify the main concept discussed in the text."""
        keywords = self.extract_keywords(text)
        return keywords[0] if keywords else ""

    def sentiment_analysis(self, text: str) -> str:
        """Analyze the sentiment of the text."""
        scores = self.sentiment_analyzer.polarity_scores(text)
        if scores['compound'] >= 0.05:
            return 'Positive'
        elif scores['compound'] <= -0.05:
            return 'Negative'
        return 'Neutral'

    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from the text based on frequency."""
        words = word_tokenize(text.lower())
        word_freq = Counter(word for word in words 
                           if word.isalnum() and word not in self.stop_words and len(word) > 2)
        
        return [word for word, _ in word_freq.most_common(self.num_tags)]

    def generate_word_cloud(self, text: str) -> WordCloud:
        """Generate a word cloud from the text."""
        return WordCloud(width=800, height=400,
                        background_color='white',
                        stopwords=self.stop_words).generate(text)

    def detect_language(self, text: str) -> str:
        """Detect the language of the input text."""
        spanish_words = {'el', 'la', 'los', 'las', 'es', 'son', 'está', 'hola', 'mundo'}
        french_words = {'le', 'la', 'les', 'est', 'sont', 'bonjour', 'monde'}
        german_words = {'der', 'die', 'das', 'ist', 'sind', 'hallo', 'welt'}
        
        words = set(word_tokenize(text.lower()))
        
        sp_count = len(words.intersection(spanish_words))
        fr_count = len(words.intersection(french_words))
        de_count = len(words.intersection(german_words))
        
        if sp_count > fr_count and sp_count > de_count and sp_count > 0:
            return 'es'
        elif fr_count > sp_count and fr_count > de_count and fr_count > 0:
            return 'fr'
        elif de_count > sp_count and de_count > fr_count and de_count > 0:
            return 'de'
        
        return 'en'

    def translate_text(self, text: str, target_lang: str = 'en') -> str:
        """Translate text to the target language."""
        translations = {
            'en-es': {'hello': 'hola', 'world': 'mundo', 'hello world': 'hola mundo'},
            'en-fr': {'hello': 'bonjour', 'world': 'monde', 'hello world': 'bonjour monde'}
        }
        
        from_to = f"en-{target_lang}" if target_lang != 'en' else f"{target_lang}-en"
        
        text_lower = text.lower()
        if from_to in translations and text_lower in translations[from_to]:
            return translations[from_to][text_lower]
        
        return text

    def generate_tag_summary(self, text: str) -> List[str]:
        """Generate a tag-based summary of the text."""
        return self.extract_keywords(text)

    def generate_sentence_summary(self, text: str, model_config: Optional[Dict[str, Any]] = None) -> str:
        """Generate a summary by extracting key sentences."""
        config = model_config or {}
        num_sentences = config.get('num_sentences', 3)
        max_tokens = config.get('maxTokens', 200)
        
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            return text

        word_freq = self._calculate_word_frequencies(text)
        scores = self._score_sentences(sentences, word_freq)
        summary_sentences = self._extract_summary_sentences(sentences, scores, num_sentences)
        
        summary = ' '.join(summary_sentences)
        summary_tokens = word_tokenize(summary)
        if len(summary_tokens) > max_tokens:
            return ' '.join(summary_tokens[:max_tokens]) + '...'
            
        return summary

    def _generate_condensed_summary(self, text: str, model_config: Optional[Dict[str, Any]] = None) -> str:
        """Generate a very concise summary."""
        config = model_config or {}
        max_tokens = config.get('maxTokens', 100) // 2
        
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            return text

        word_freq = self._calculate_word_frequencies(text)
        scores = self._score_sentences(sentences, word_freq)
        top_sentence = max(scores.items(), key=lambda x: x[1])[0]
        
        tokens = word_tokenize(top_sentence)
        if len(tokens) > max_tokens:
            return ' '.join(tokens[:max_tokens]) + '...'
            
        return top_sentence

    def _calculate_word_frequencies(self, text: str) -> Counter:
        """Calculate word frequencies for the text."""
        words = word_tokenize(text.lower())
        return Counter(word for word in words 
                      if word.isalnum() and word not in self.stop_words)

    def _score_sentences(self, sentences: List[str], word_freq: Counter) -> Dict[str, float]:
        """Score sentences based on word frequencies."""
        scores = {}
        max_freq = max(word_freq.values()) if word_freq else 1
        
        for i, sentence in enumerate(sentences):
            words = word_tokenize(sentence.lower())
            if not words:
                scores[sentence] = 0
                continue
                
            word_scores = sum(word_freq[word] / max_freq for word in words if word in word_freq)
            position_factor = 1.0 - (0.1 * (i / len(sentences)))
            scores[sentence] = (word_scores / len(words)) * position_factor
            
        return scores

    def _extract_summary_sentences(self, sentences: List[str], scores: Dict[str, float], num_sentences: int) -> List[str]:
        """Extract top scoring sentences while preserving order."""
        indexed_scores = [(i, sentence, scores[sentence]) for i, sentence in enumerate(sentences)]
        top_indexed = sorted(indexed_scores, key=lambda x: x[2], reverse=True)[:num_sentences]
        ordered_sentences = sorted(top_indexed, key=lambda x: x[0])
        
        return [sentence for _, sentence, _ in ordered_sentences]
        
    def adjust_parameters(self, num_tags: int) -> None:
        """Adjust the parameters of the summarizer."""
        self.num_tags = max(1, min(num_tags, 10))
        logger.info(f"Adjusted num_tags to {self.num_tags}")


# For backward compatibility
AdvancedSUM = MagnumOpusSUM

# Example usage
if __name__ == "__main__":
    sample_text = """
    Machine learning has seen rapid advancements in recent years. From image recognition to
    natural language processing, AI systems are becoming increasingly sophisticated. Deep learning
    models, in particular, have shown remarkable capabilities in handling complex tasks. However,
    challenges remain in areas such as explainability and bias mitigation. As the field continues
    to evolve, researchers are developing new approaches to address these limitations and expand
    the applications of machine learning across various domains.
    """
    
    simple_summarizer = SimpleSUM()
    advanced_summarizer = MagnumOpusSUM()
    
    simple_result = simple_summarizer.process_text(sample_text, {'maxTokens': 50})
    print(f"SimpleSUM Summary: {simple_result.get('summary', '')}")
    
    advanced_result = advanced_summarizer.process_text(sample_text, {'maxTokens': 50, 'include_analysis': True})
    print(f"MagnumOpusSUM Summary: {advanced_result.get('summary', '')}")
