"""
Language Detection and Multi-language Support for SUM

Provides automatic language detection and language-specific processing
for optimal summarization across different languages.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
import re
import unicodedata

# Language detection libraries
try:
    import langdetect
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    import langid
    LANGID_AVAILABLE = True
except ImportError:
    LANGID_AVAILABLE = False

# NLTK language-specific resources
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

logger = logging.getLogger(__name__)


class LanguageDetector:
    """
    Advanced language detection with multiple fallback methods.
    """
    
    # Language names mapping
    LANGUAGE_NAMES = {
        'en': 'English',
        'es': 'Spanish', 
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'ja': 'Japanese',
        'zh': 'Chinese',
        'ko': 'Korean',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'nl': 'Dutch',
        'sv': 'Swedish',
        'pl': 'Polish',
        'tr': 'Turkish',
        'vi': 'Vietnamese',
        'th': 'Thai',
        'he': 'Hebrew',
        'id': 'Indonesian'
    }
    
    # Supported languages for summarization
    SUPPORTED_LANGUAGES = {
        'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'nl', 'sv', 'pl', 'tr', 'id'
    }
    
    def __init__(self):
        """Initialize language detector with available libraries."""
        self.methods = []
        
        if LANGDETECT_AVAILABLE:
            self.methods.append('langdetect')
            # Configure langdetect
            langdetect.seed = 0  # For consistent results
            
        if LANGID_AVAILABLE:
            self.methods.append('langid')
            self.langid_identifier = langid.langid.LanguageIdentifier.from_modelstring(
                langid.langid.model, norm_probs=True
            )
            
        # Character-based detection patterns
        self.script_patterns = {
            'arabic': re.compile(r'[\u0600-\u06FF]'),
            'chinese': re.compile(r'[\u4E00-\u9FFF]'),
            'japanese': re.compile(r'[\u3040-\u309F\u30A0-\u30FF]'),
            'korean': re.compile(r'[\uAC00-\uD7AF]'),
            'hebrew': re.compile(r'[\u0590-\u05FF]'),
            'thai': re.compile(r'[\u0E00-\u0E7F]'),
            'cyrillic': re.compile(r'[\u0400-\u04FF]'),
            'devanagari': re.compile(r'[\u0900-\u097F]')
        }
        
        self._download_nltk_resources()
        
    def _download_nltk_resources(self):
        """Download NLTK resources for multiple languages."""
        resources = [
            'stopwords',
            'punkt',
            'averaged_perceptron_tagger',
            'maxent_ne_chunker',
            'words'
        ]
        
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except:
                    logger.warning(f"Could not download NLTK resource: {resource}")
    
    def detect_language(self, text: str, 
                       min_confidence: float = 0.7) -> Dict[str, Any]:
        """
        Detect language using multiple methods with confidence scoring.
        
        Args:
            text: Input text
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary with language code, name, confidence, and method
        """
        if not text or len(text.strip()) < 10:
            return {
                'language': 'en',
                'language_name': 'English',
                'confidence': 0.5,
                'method': 'default',
                'supported': True
            }
        
        # Try each detection method
        results = []
        
        # Method 1: langdetect
        if 'langdetect' in self.methods:
            try:
                langs = detect_langs(text[:5000])  # Limit text for performance
                if langs:
                    best = langs[0]
                    results.append({
                        'language': best.lang,
                        'confidence': best.prob,
                        'method': 'langdetect'
                    })
            except Exception as e:
                logger.debug(f"langdetect failed: {e}")
        
        # Method 2: langid
        if 'langid' in self.methods:
            try:
                lang, confidence = self.langid_identifier.classify(text[:5000])
                results.append({
                    'language': lang,
                    'confidence': confidence,
                    'method': 'langid'
                })
            except Exception as e:
                logger.debug(f"langid failed: {e}")
        
        # Method 3: Script-based detection
        script_result = self._detect_by_script(text)
        if script_result:
            results.append(script_result)
        
        # Method 4: Stopwords-based detection
        stopwords_result = self._detect_by_stopwords(text)
        if stopwords_result:
            results.append(stopwords_result)
        
        # Combine results
        if not results:
            return {
                'language': 'en',
                'language_name': 'English',
                'confidence': 0.5,
                'method': 'fallback',
                'supported': True
            }
        
        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        best_result = results[0]
        
        # Validate and enhance result
        lang_code = best_result['language']
        
        return {
            'language': lang_code,
            'language_name': self.LANGUAGE_NAMES.get(lang_code, lang_code.title()),
            'confidence': best_result['confidence'],
            'method': best_result['method'],
            'supported': lang_code in self.SUPPORTED_LANGUAGES,
            'all_results': results[:3]  # Top 3 results
        }
    
    def _detect_by_script(self, text: str) -> Optional[Dict[str, Any]]:
        """Detect language by character script."""
        char_counts = Counter()
        total_chars = 0
        
        for char in text[:1000]:  # Sample first 1000 chars
            if char.isalpha():
                total_chars += 1
                for script, pattern in self.script_patterns.items():
                    if pattern.search(char):
                        char_counts[script] += 1
                        break
        
        if not total_chars:
            return None
        
        # Map scripts to languages
        script_to_lang = {
            'arabic': 'ar',
            'chinese': 'zh',
            'japanese': 'ja',
            'korean': 'ko',
            'hebrew': 'he',
            'thai': 'th',
            'cyrillic': 'ru',
            'devanagari': 'hi'
        }
        
        if char_counts:
            most_common_script = char_counts.most_common(1)[0]
            script, count = most_common_script
            confidence = count / total_chars
            
            if confidence > 0.3:
                return {
                    'language': script_to_lang.get(script, 'unknown'),
                    'confidence': confidence,
                    'method': 'script'
                }
        
        return None
    
    def _detect_by_stopwords(self, text: str) -> Optional[Dict[str, Any]]:
        """Detect language by matching stopwords."""
        try:
            # Tokenize text
            words = word_tokenize(text.lower())[:200]  # First 200 words
            
            if not words:
                return None
            
            # Check each language's stopwords
            matches = {}
            available_langs = ['english', 'spanish', 'french', 'german', 
                             'italian', 'portuguese', 'russian', 'dutch',
                             'swedish', 'turkish', 'indonesian']
            
            lang_map = {
                'english': 'en', 'spanish': 'es', 'french': 'fr',
                'german': 'de', 'italian': 'it', 'portuguese': 'pt',
                'russian': 'ru', 'dutch': 'nl', 'swedish': 'sv',
                'turkish': 'tr', 'indonesian': 'id'
            }
            
            for lang in available_langs:
                try:
                    stops = set(stopwords.words(lang))
                    match_count = sum(1 for word in words if word in stops)
                    matches[lang] = match_count / len(words)
                except:
                    continue
            
            if matches:
                best_lang = max(matches, key=matches.get)
                confidence = matches[best_lang]
                
                if confidence > 0.15:  # At least 15% stopwords
                    return {
                        'language': lang_map.get(best_lang, 'en'),
                        'confidence': min(confidence * 2, 1.0),  # Scale up
                        'method': 'stopwords'
                    }
            
        except Exception as e:
            logger.debug(f"Stopwords detection failed: {e}")
        
        return None
    
    def get_language_config(self, language: str) -> Dict[str, Any]:
        """
        Get language-specific configuration for summarization.
        
        Args:
            language: Language code
            
        Returns:
            Configuration dictionary
        """
        configs = {
            'en': {
                'sentence_tokenizer': 'punkt',
                'word_tokenizer': 'word_tokenize',
                'stopwords': 'english',
                'stemmer': 'PorterStemmer',
                'min_sentence_length': 5,
                'sentence_delimiter': '. '
            },
            'es': {
                'sentence_tokenizer': 'punkt',
                'word_tokenizer': 'word_tokenize',
                'stopwords': 'spanish',
                'stemmer': 'SnowballStemmer',
                'stemmer_lang': 'spanish',
                'min_sentence_length': 5,
                'sentence_delimiter': '. '
            },
            'fr': {
                'sentence_tokenizer': 'punkt',
                'word_tokenizer': 'word_tokenize', 
                'stopwords': 'french',
                'stemmer': 'SnowballStemmer',
                'stemmer_lang': 'french',
                'min_sentence_length': 5,
                'sentence_delimiter': '. '
            },
            'de': {
                'sentence_tokenizer': 'punkt',
                'word_tokenizer': 'word_tokenize',
                'stopwords': 'german',
                'stemmer': 'SnowballStemmer',
                'stemmer_lang': 'german',
                'min_sentence_length': 5,
                'sentence_delimiter': '. '
            },
            'zh': {
                'sentence_tokenizer': 'chinese',
                'word_tokenizer': 'jieba',
                'stopwords': None,
                'stemmer': None,
                'min_sentence_length': 3,
                'sentence_delimiter': '。'
            },
            'ja': {
                'sentence_tokenizer': 'japanese',
                'word_tokenizer': 'mecab',
                'stopwords': None,
                'stemmer': None,
                'min_sentence_length': 3,
                'sentence_delimiter': '。'
            }
        }
        
        # Return specific config or default to English
        return configs.get(language, configs['en'])


class MultilingualSummarizer:
    """
    Summarizer with multi-language support.
    """
    
    def __init__(self):
        """Initialize multilingual summarizer."""
        self.detector = LanguageDetector()
        self._language_processors = {}
        
    def summarize(self, text: str, 
                 config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Summarize text with automatic language detection.
        
        Args:
            text: Input text
            config: Summarization configuration
            
        Returns:
            Summary with language information
        """
        # Detect language
        lang_info = self.detector.detect_language(text)
        language = lang_info['language']
        
        # Get language-specific configuration
        lang_config = self.detector.get_language_config(language)
        
        # Prepare result
        result = {
            'detected_language': lang_info,
            'language_config': lang_config
        }
        
        # If language is not supported, fall back to English
        if not lang_info['supported']:
            logger.warning(f"Language {language} not fully supported, using English")
            language = 'en'
            result['fallback_language'] = 'en'
        
        # Apply language-specific preprocessing
        processed_text = self._preprocess_text(text, language, lang_config)
        result['preprocessed'] = True
        
        # Return enhanced configuration
        return {
            **result,
            'text': processed_text,
            'original_text': text,
            'language': language,
            'language_name': lang_info['language_name']
        }
    
    def _preprocess_text(self, text: str, language: str, 
                        config: Dict[str, Any]) -> str:
        """Apply language-specific preprocessing."""
        # Basic preprocessing for all languages
        text = text.strip()
        
        # Language-specific preprocessing
        if language == 'zh':
            # Chinese: Add spaces between sentences
            text = re.sub(r'。', '。 ', text)
        elif language == 'ja':
            # Japanese: Normalize characters
            text = unicodedata.normalize('NFKC', text)
        elif language == 'ar':
            # Arabic: Right-to-left handling
            text = text.strip()
        
        return text


# Global instance
language_detector = LanguageDetector()
multilingual_summarizer = MultilingualSummarizer()


def detect_language(text: str) -> Dict[str, Any]:
    """
    Detect language of text.
    
    Args:
        text: Input text
        
    Returns:
        Language detection results
    """
    return language_detector.detect_language(text)


def summarize_multilingual(text: str, 
                          config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Summarize text with language detection.
    
    Args:
        text: Input text
        config: Configuration
        
    Returns:
        Summary with language information
    """
    return multilingual_summarizer.summarize(text, config)