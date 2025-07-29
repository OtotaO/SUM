"""test_text_preprocessing.py - Unit tests for the text preprocessing utilities

This module provides comprehensive unit tests for the text preprocessing utilities,
ensuring proper functionality for text cleaning, tokenization, and normalization.

Author: ototao
License: Apache License 2.0
"""

import unittest
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Utils.text_preprocessing import (
    TextPreprocessor,
    preprocess_text,
    tokenize_sentences,
    tokenize_words,
    calculate_word_frequencies
)

class TestTextPreprocessing(unittest.TestCase):
    """Tests for the text preprocessing utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test preprocessor
        self.preprocessor = TextPreprocessor(initialize_resources=False)
        
        # Sample texts for testing
        self.simple_text = "This is a simple test text with some basic punctuation!"
        self.complex_text = """This is a more complex text with URLs (https://example.com), 
        email addresses (user@example.com), special characters (#$%^&*), 
        and numbers (123456). It also has multiple sentences. And some stopwords."""
        
        # Mock stopwords for testing without NLTK
        self.preprocessor.stopwords = {'a', 'an', 'the', 'is', 'are', 'and', 'with', 'some', 'it', 'also', 'has'}
        
    def test_initialization(self):
        """Test TextPreprocessor initialization."""
        # Test initialization with default parameters
        preprocessor = TextPreprocessor(initialize_resources=False)
        
        # Check that preprocessor was initialized
        self.assertIsNotNone(preprocessor)
        self.assertIsNotNone(preprocessor.stopwords)
        
        # Test initialization with custom stopwords
        custom_stopwords = {'custom', 'words'}
        preprocessor = TextPreprocessor(initialize_resources=False, custom_stopwords=custom_stopwords)
        
        # Check that custom stopwords were added
        self.assertTrue(custom_stopwords.issubset(preprocessor.stopwords))
    
    def test_preprocess_text_basic(self):
        """Test basic text preprocessing."""
        # Test with default parameters
        result = self.preprocessor.preprocess_text(self.simple_text)
        
        # Check that stopwords were removed
        self.assertNotIn('is', result)
        self.assertNotIn('a', result)
        self.assertNotIn('with', result)
        
        # Check that other words were preserved
        self.assertIn('simple', result)
        self.assertIn('test', result)
        self.assertIn('text', result)
        
        # Check that text was lowercased
        self.assertNotIn('This', result)
        self.assertIn('this', result)
    
    def test_preprocess_text_options(self):
        """Test text preprocessing with various options."""
        # Test with remove_stopwords=False
        result = self.preprocessor.preprocess_text(
            self.simple_text,
            remove_stopwords=False
        )
        
        # Check that stopwords were preserved
        self.assertIn('is', result)
        self.assertIn('a', result)
        self.assertIn('with', result)
        
        # Test with lowercase=False
        result = self.preprocessor.preprocess_text(
            self.simple_text,
            lowercase=False,
            remove_stopwords=False
        )
        
        # Check that case was preserved
        self.assertIn('This', result)
        self.assertNotIn('this', result)
        
        # Test with remove_special_chars=True
        result = self.preprocessor.preprocess_text(
            self.simple_text,
            remove_special_chars=True
        )
        
        # Check that punctuation was removed
        self.assertNotIn('!', result)
        self.assertNotIn('.', result)
    
    def test_preprocess_text_complex(self):
        """Test preprocessing of complex text."""
        # Test with various options
        result = self.preprocessor.preprocess_text(
            self.complex_text,
            remove_urls=True,
            remove_emails=True,
            remove_special_chars=True,
            remove_numbers=True
        )
        
        # Check that URLs were removed
        self.assertNotIn('https://example.com', result)
        self.assertNotIn('example.com', result)
        
        # Check that email addresses were removed
        self.assertNotIn('user@example.com', result)
        
        # Check that special characters were removed
        self.assertNotIn('#', result)
        self.assertNotIn('$', result)
        self.assertNotIn('%', result)
        
        # Check that numbers were removed
        self.assertNotIn('123456', result)
    
    def test_tokenize_sentences(self):
        """Test sentence tokenization."""
        # Test with simple text
        sentences = self.preprocessor.tokenize_sentences("This is a sentence. This is another sentence!")
        
        # Check that sentences were tokenized correctly
        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0], "This is a sentence.")
        self.assertEqual(sentences[1], "This is another sentence!")
        
        # Test with empty text
        sentences = self.preprocessor.tokenize_sentences("")
        self.assertEqual(len(sentences), 0)
        
        # Test with None
        sentences = self.preprocessor.tokenize_sentences(None)
        self.assertEqual(len(sentences), 0)
    
    def test_tokenize_words(self):
        """Test word tokenization."""
        # Test with simple text
        words = self.preprocessor.tokenize_words("This is a simple test.")
        
        # Check that words were tokenized correctly
        self.assertEqual(len(words), 5)
        self.assertIn("This", words)
        self.assertIn("is", words)
        self.assertIn("a", words)
        self.assertIn("simple", words)
        self.assertIn("test", words)
        
        # Test with empty text
        words = self.preprocessor.tokenize_words("")
        self.assertEqual(len(words), 0)
        
        # Test with None
        words = self.preprocessor.tokenize_words(None)
        self.assertEqual(len(words), 0)
    
    def test_extract_ngrams(self):
        """Test n-gram extraction."""
        # Test with simple text
        bigrams = self.preprocessor.extract_ngrams("This is a simple test.", n=2)
        
        # Check that bigrams were extracted correctly
        self.assertIn("This is", bigrams)
        self.assertIn("is a", bigrams)
        self.assertIn("a simple", bigrams)
        self.assertIn("simple test", bigrams)
        
        # Test with n=3
        trigrams = self.preprocessor.extract_ngrams("This is a simple test.", n=3)
        
        # Check that trigrams were extracted correctly
        self.assertIn("This is a", trigrams)
        self.assertIn("is a simple", trigrams)
        self.assertIn("a simple test", trigrams)
        
        # Test with empty text
        ngrams = self.preprocessor.extract_ngrams("", n=2)
        self.assertEqual(len(ngrams), 0)
        
        # Test with None
        ngrams = self.preprocessor.extract_ngrams(None, n=2)
        self.assertEqual(len(ngrams), 0)
    
    def test_calculate_word_frequencies(self):
        """Test word frequency calculation."""
        # Test with simple text
        frequencies = self.preprocessor.calculate_word_frequencies("This is a test. This is another test.")
        
        # Check that frequencies were calculated correctly
        self.assertEqual(frequencies["this"], 2)
        self.assertEqual(frequencies["test"], 2)
        self.assertEqual(frequencies["another"], 1)
        
        # Check that stopwords were removed
        self.assertNotIn("is", frequencies)
        self.assertNotIn("a", frequencies)
        
        # Test with remove_stopwords=False
        frequencies = self.preprocessor.calculate_word_frequencies(
            "This is a test. This is another test.",
            remove_stopwords=False
        )
        
        # Check that stopwords were included
        self.assertIn("is", frequencies)
        self.assertIn("a", frequencies)
    
    def test_is_safe_string(self):
        """Test string safety checking."""
        # Test with safe strings
        self.assertTrue(self.preprocessor.is_safe_string("This is a safe string."))
        self.assertTrue(self.preprocessor.is_safe_string("123456"))
        
        # Test with potentially unsafe strings
        self.assertFalse(self.preprocessor.is_safe_string("exec('import os')"))
        self.assertFalse(self.preprocessor.is_safe_string("eval('2 + 2')"))
        self.assertFalse(self.preprocessor.is_safe_string("import sys"))
        self.assertFalse(self.preprocessor.is_safe_string("__dict__"))
        self.assertFalse(self.preprocessor.is_safe_string("open('file.txt')"))
        
        # Test with empty string
        self.assertFalse(self.preprocessor.is_safe_string(""))
        
        # Test with None
        self.assertFalse(self.preprocessor.is_safe_string(None))
        
        # Test with very long string
        self.assertFalse(self.preprocessor.is_safe_string("a" * 11000))
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        # Test preprocess_text
        result = preprocess_text("This is a test.")
        self.assertNotIn("is", result)
        self.assertNotIn("a", result)
        self.assertIn("this", result)
        self.assertIn("test", result)
        
        # Test tokenize_sentences
        sentences = tokenize_sentences("This is a sentence. This is another sentence!")
        self.assertEqual(len(sentences), 2)
        
        # Test tokenize_words
        words = tokenize_words("This is a simple test.")
        self.assertEqual(len(words), 5)
        
        # Test calculate_word_frequencies
        frequencies = calculate_word_frequencies("This is a test. This is another test.")
        self.assertEqual(frequencies["this"], 2)
        self.assertEqual(frequencies["test"], 2)


if __name__ == '__main__':
    unittest.main()