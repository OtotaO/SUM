"""
text_preprocessing.py - Centralized text preprocessing utilities

This module provides standardized text preprocessing functions for the SUM platform,
ensuring consistent text handling across all components.

Design principles:
- Consistent preprocessing
- Efficient implementation
- Configurable options
- Robust error handling

Author: ototao
License: Apache License 2.0
"""

import re
import logging
import string
from typing import List, Set, Dict, Tuple, Optional, Any, Union
import threading

# Import centralized NLTK utilities
from Utils.nltk_utils import get_stopwords, get_lemmatizer, initialize_nltk

# Configure logging
logger = logging.getLogger(__name__)

# Thread safety
_preprocessing_lock = threading.Lock()

class TextPreprocessor:
    """
    Centralized text preprocessing utility.
    
    This class provides standardized methods for text preprocessing,
    including tokenization, normalization, and cleaning.
    """
    
    def __init__(self, 
                initialize_resources: bool = True,
                custom_stopwords: Optional[Set[str]] = None):
        """
        Initialize the text preprocessor.
        
        Args:
            initialize_resources: Whether to initialize NLTK resources
            custom_stopwords: Custom stopwords to add to the standard set
        """
        # Initialize NLTK resources if requested
        if initialize_resources:
            initialize_nltk()
            
        # Get standard stopwords
        self.stopwords = get_stopwords()
        
        # Add custom stopwords if provided
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
            
        # Get lemmatizer
        self.lemmatizer = get_lemmatizer()
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.special_char_pattern = re.compile(r'[^\w\s]')
        self.multiple_spaces_pattern = re.compile(r'\s+')
        self.numbers_pattern = re.compile(r'\b\d+\b')
        
        logger.debug("TextPreprocessor initialized")
    
    def preprocess_text(self, 
                      text: str, 
                      lowercase: bool = True,
                      remove_urls: bool = True,
                      remove_emails: bool = True,
                      remove_special_chars: bool = False,
                      remove_numbers: bool = False,
                      remove_stopwords: bool = True,
                      lemmatize: bool = False) -> str:
        """
        Preprocess text with configurable options.
        
        Args:
            text: Input text to preprocess
            lowercase: Convert to lowercase
            remove_urls: Remove URLs
            remove_emails: Remove email addresses
            remove_special_chars: Remove special characters
            remove_numbers: Remove numbers
            remove_stopwords: Remove stopwords
            lemmatize: Apply lemmatization
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Make a copy of the text
        processed_text = text
        
        # Convert to lowercase if requested
        if lowercase:
            processed_text = processed_text.lower()
            
        # Remove URLs if requested
        if remove_urls:
            processed_text = self.url_pattern.sub(' ', processed_text)
            
        # Remove emails if requested
        if remove_emails:
            processed_text = self.email_pattern.sub(' ', processed_text)
            
        # Remove special characters if requested
        if remove_special_chars:
            processed_text = self.special_char_pattern.sub(' ', processed_text)
            
        # Remove numbers if requested
        if remove_numbers:
            processed_text = self.numbers_pattern.sub(' ', processed_text)
            
        # Normalize whitespace
        processed_text = self.multiple_spaces_pattern.sub(' ', processed_text).strip()
        
        # Tokenize, remove stopwords, and lemmatize if requested
        if remove_stopwords or lemmatize:
            tokens = processed_text.split()
            
            # Process tokens
            processed_tokens = []
            for token in tokens:
                # Skip stopwords if requested
                if remove_stopwords and token in self.stopwords:
                    continue
                    
                # Apply lemmatization if requested
                if lemmatize and self.lemmatizer:
                    try:
                        token = self.lemmatizer.lemmatize(token)
                    except Exception as e:
                        logger.warning(f"Lemmatization error for token '{token}': {e}")
                
                processed_tokens.append(token)
                
            # Rejoin tokens
            processed_text = ' '.join(processed_tokens)
            
        return processed_text
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Tokenize text into sentences using NLTK's sentence tokenizer.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        if not text or not isinstance(text, str):
            return []
            
        try:
            from nltk.tokenize import sent_tokenize
            return sent_tokenize(text)
        except Exception as e:
            logger.error(f"Error tokenizing sentences: {e}")
            # Fallback to simple sentence splitting
            return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    def tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize text into words using NLTK's word tokenizer.
        
        Args:
            text: Input text
            
        Returns:
            List of words
        """
        if not text or not isinstance(text, str):
            return []
            
        try:
            from nltk.tokenize import word_tokenize
            return word_tokenize(text)
        except Exception as e:
            logger.error(f"Error tokenizing words: {e}")
            # Fallback to simple word splitting
            return [w for w in re.split(r'\W+', text) if w]
    
    def extract_ngrams(self, text: str, n: int = 2) -> List[str]:
        """
        Extract n-grams from text.
        
        Args:
            text: Input text
            n: Size of n-grams
            
        Returns:
            List of n-grams
        """
        if not text or not isinstance(text, str) or n < 1:
            return []
            
        words = self.tokenize_words(text)
        
        # Generate n-grams
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(' '.join(words[i:i+n]))
            
        return ngrams
    
    def calculate_word_frequencies(self, text: str, remove_stopwords: bool = True) -> Dict[str, int]:
        """
        Calculate word frequencies in text.
        
        Args:
            text: Input text
            remove_stopwords: Whether to remove stopwords
            
        Returns:
            Dictionary mapping words to their frequencies
        """
        if not text or not isinstance(text, str):
            return {}
            
        # Tokenize words
        words = self.tokenize_words(text.lower())
        
        # Calculate frequencies
        frequencies = {}
        for word in words:
            if word.isalnum() and (not remove_stopwords or word not in self.stopwords):
                frequencies[word] = frequencies.get(word, 0) + 1
                
        return frequencies
    
    def is_safe_string(self, text: str) -> bool:
        """
        Check if a string is safe (no potentially harmful patterns).
        
        Args:
            text: String to check
            
        Returns:
            True if the string is safe, False otherwise
        """
        if not text or not isinstance(text, str):
            return False
            
        # Check for unusually long strings
        if len(text) > 10000:
            return False
            
        # Check for potentially harmful patterns
        unsafe_patterns = [
            r'[\s\S]*exec\s*\(', 
            r'[\s\S]*eval\s*\(', 
            r'[\s\S]*\bimport\b',
            r'[\s\S]*__[a-zA-Z]+__', 
            r'[\s\S]*\bopen\s*\('
        ]
        
        return not any(re.search(pattern, text) for pattern in unsafe_patterns)


# Create a default instance for easy import
default_preprocessor = TextPreprocessor()

# Convenience functions that use the default preprocessor
def preprocess_text(text: str, **kwargs) -> str:
    """
    Preprocess text using the default preprocessor.
    
    Args:
        text: Input text
        **kwargs: Additional arguments to pass to preprocess_text
        
    Returns:
        Preprocessed text
    """
    return default_preprocessor.preprocess_text(text, **kwargs)

def tokenize_sentences(text: str) -> List[str]:
    """
    Tokenize text into sentences using the default preprocessor.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    return default_preprocessor.tokenize_sentences(text)

def tokenize_words(text: str) -> List[str]:
    """
    Tokenize text into words using the default preprocessor.
    
    Args:
        text: Input text
        
    Returns:
        List of words
    """
    return default_preprocessor.tokenize_words(text)

def calculate_word_frequencies(text: str, remove_stopwords: bool = True) -> Dict[str, int]:
    """
    Calculate word frequencies using the default preprocessor.
    
    Args:
        text: Input text
        remove_stopwords: Whether to remove stopwords
        
    Returns:
        Dictionary mapping words to their frequencies
    """
    return default_preprocessor.calculate_word_frequencies(text, remove_stopwords)
