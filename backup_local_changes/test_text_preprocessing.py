"""
test_text_preprocessing.py - Unit tests for text preprocessing utilities

This module provides comprehensive unit tests for the text preprocessing utilities,
ensuring proper functionality for text cleaning, tokenization, and analysis.

Author: ototao
License: Apache License 2.0
"""

import unittest
import os
import sys
from unittest.mock import patch, MagicMock

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
        # Sample texts for testing
        self.simple_text = "This is a simple test text."
        self.complex_text = """
        This is a more complex text with URLs like https://example.com and 
        email addresses like test@example.com. It also has special characters 
        like !@#$%^&*() and numbers like 12345. This text has multiple sentences.
        Some words appear multiple times, like text, text, text.
        """
        
        # Mock NLTK resources
        self.patcher1 = patch('Utils.nltk_utils.initialize_nltk')
        self.patcher2 = patch('Utils.nltk_utils.get_stopwords')
        self.patcher3 = patch('Utils.nltk_utils.get_lemmatizer')
        
        self.mock_initialize_nltk = self.patcher1.start()
        self.mock_get_stopwords = self.patcher2.start()
        self.mock_get_lemmatizer = self.patcher3.start()
        
        # Configure mocks
        self.mock_initialize_nltk.return_value = True
        self.mock_get_stopwords.return_value = {'a', 'an', 'the', 'is', 'are', 'and', 'with', 'this', 'it', 'also', 'has', 'like', 'some'}
        
        mock_lemmatizer = MagicMock()
        mock_lemmatizer.lemmatize.side_effect = lambda word: word.rstrip('s')
        self.mock_get_lemmatizer.return_value = mock_lemmatizer
        
    def tearDown(self):
        """Clean up test fixtures."""
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()
    
    def test_text_preprocessor_initialization(self):
        """Test TextPreprocessor initialization."""
        preprocessor = TextPreprocessor()
        
        # Check that NLTK resources were initialized
        self.mock_initialize_nltk.assert_called_once()
        self.mock_get_stopwords.assert_called_once()
        self.mock_get_lemmatizer.assert_called_once()
        
        # Check that stopwords and lemmatizer were set
        self.assertEqual(preprocessor.stopwords, self.mock_get_stopwords.return_value)
        self.assertEqual(preprocessor.lemmatizer, self.mock_get_lemmatizer.return_value)
    
    def test_text_preprocessor_with_custom_stopwords(self):
        """Test TextPreprocessor with custom stopwords."""
        custom_stopwords = {'custom', 'stopword'}
        preprocessor = TextPreprocessor(custom_stopwords=custom_stopwords)
        
        # Check that custom stopwords were added
        for word in custom_stopwords:
            self.assertIn(word, preprocessor.stopwords)
    
    def test_preprocess_text_basic(self):
        """Test basic text preprocessing."""
        preprocessor = TextPreprocessor()
        result = preprocessor.preprocess_text(self.simple_text)
        
        # Check that text was preprocessed
        self.assertIsInstance(result, str)
        self.assertTrue(result)  # Not empty
        
        # Check that text was lowercased
        self.assertEqual(result, result.lower())
        
        # Check that stopwords were removed
        for word in preprocessor.stopwords:
            if word in self.simple_text.lower():
                self.assertNotIn(f" {word} ", f" {result} ")
    
    def test_preprocess_text_with_options(self):
        """Test text preprocessing with various options."""
        preprocessor = TextPreprocessor()
        
        # Test with URLs and emails
        result = preprocessor.preprocess_text(
            self.complex_text,
            remove_urls=True,
            remove_emails=True
        )
        self.assertNotIn("https://example.com", result)
        self.assertNotIn("test@example.com", result)
        
        # Test with special characters
        result = preprocessor.preprocess_text(
            self.complex_text,
            remove_special_chars=True
        )
        for char in "!@#$%^&*()":
            self.assertNotIn(char, result)
        
        # Test with numbers
        result = preprocessor.preprocess_text(
            self.complex_text,
            remove_numbers=True
        )
        self.assertNotIn("12345", result)
        
        # Test with lemmatization
        result = preprocessor.preprocess_text(
            "Running and jumps are examples of words",
            lemmatize=True
        )
        self.assertIn("running", result)
        self.assertIn("jump", result)
        
        # Test without lowercase
        result = preprocessor.preprocess_text(
            "This Has Mixed CASE",
            lowercase=False
        )
        self.assertIn("This", result)
        self.assertIn("Has", result)
        self.assertIn("Mixed", result)
        self.assertIn("CASE", result)
    
    def test_tokenize_sentences(self):
        """Test sentence tokenization."""
        preprocessor = TextPreprocessor()
        
        # Test with simple text
        sentences = preprocessor.tokenize_sentences(self.simple_text)
        self.assertIsInstance(sentences, list)
        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0], self.simple_text)
        
        # Test with complex text
        sentences = preprocessor.tokenize_sentences(self.complex_text)
        self.assertIsInstance(sentences, list)
        self.assertGreater(len(sentences), 1)
        
        # Test with empty text
        sentences = preprocessor.tokenize_sentences("")
        self.assertEqual(sentences, [])
        
        # Test with non-string
        sentences = preprocessor.tokenize_sentences(None)
        self.assertEqual(sentences, [])
    
    def test_tokenize_words(self):
        """Test word tokenization."""
        preprocessor = TextPreprocessor()
        
        # Test with simple text
        words = preprocessor.tokenize_words(self.simple_text)
        self.assertIsInstance(words, list)
        self.assertGreater(len(words), 1)
        self.assertIn("This", words)
        self.assertIn("simple", words)
        self.assertIn("test", words)
        
        # Test with empty text
        words = preprocessor.tokenize_words("")
        self.assertEqual(words, [])
        
        # Test with non-string
        words = preprocessor.tokenize_words(None)
        self.assertEqual(words, [])
    
    def test_extract_ngrams(self):
        """Test n-gram extraction."""
        preprocessor = TextPreprocessor()
        
        # Test with bigrams
        bigrams = preprocessor.extract_ngrams(self.simple_text, n=2)
        self.assertIsInstance(bigrams, list)
        self.assertGreater(len(bigrams), 0)
        self.assertIn("This is", bigrams)
        self.assertIn("is a", bigrams)
        
        # Test with trigrams
        trigrams = preprocessor.extract_ngrams(self.simple_text, n=3)
        self.assertIsInstance(trigrams, list)
        self.assertGreater(len(trigrams), 0)
        self.assertIn("This is a", trigrams)
        
        # Test with invalid n
        ngrams = preprocessor.extract_ngrams(self.simple_text, n=0)
        self.assertEqual(ngrams, [])
        
        # Test with empty text
        ngrams = preprocessor.extract_ngrams("", n=2)
        self.assertEqual(ngrams, [])
    
    def test_calculate_word_frequencies(self):
        """Test word frequency calculation."""
        preprocessor = TextPreprocessor()
        
        # Test with simple text
        frequencies = preprocessor.calculate_word_frequencies(self.simple_text)
        self.assertIsInstance(frequencies, dict)
        self.assertIn("simple", frequencies)
        self.assertIn("test", frequencies)
        self.assertEqual(frequencies["simple"], 1)
        self.assertEqual(frequencies["test"], 1)
        
        # Test with repeated words
        frequencies = preprocessor.calculate_word_frequencies("test test test")
        self.assertIn("test", frequencies)
        self.assertEqual(frequencies["test"], 3)
        
        # Test with stopwords
        frequencies = preprocessor.calculate_word_frequencies(
            "the test is a test",
            remove_stopwords=True
        )
        self.assertIn("test", frequencies)
        self.assertEqual(frequencies["test"], 2)
        self.assertNotIn("the", frequencies)
        self.assertNotIn("is", frequencies)
        self.assertNotIn("a", frequencies)
        
        # Test with empty text
        frequencies = preprocessor.calculate_word_frequencies("")
        self.assertEqual(frequencies, {})
    
    def test_is_safe_string(self):
        """Test string safety validation."""
        preprocessor = TextPreprocessor()
        
        # Test with safe strings
        self.assertTrue(preprocessor.is_safe_string("This is a safe string"))
        self.assertTrue(preprocessor.is_safe_string("123456"))
        
        # Test with unsafe strings
        self.assertFalse(preprocessor.is_safe_string("exec('import os')"))
        self.assertFalse(preprocessor.is_safe_string("eval('2+2')"))
        self.assertFalse(preprocessor.is_safe_string("import sys"))
        self.assertFalse(preprocessor.is_safe_string("__dict__"))
        self.assertFalse(preprocessor.is_safe_string("open('file.txt')"))
        
        # Test with very long string
        very_long_string = "a" * 20000
        self.assertFalse(preprocessor.is_safe_string(very_long_string))
        
        # Test with empty string
        self.assertFalse(preprocessor.is_safe_string(""))
        
        # Test with non-string
        self.assertFalse(preprocessor.is_safe_string(None))
    
    def test_module_level_functions(self):
        """Test module-level convenience functions."""
        # Test preprocess_text
        result = preprocess_text(self.simple_text)
        self.assertIsInstance(result, str)
        self.assertTrue(result)
        
        # Test tokenize_sentences
        sentences = tokenize_sentences(self.simple_text)
        self.assertIsInstance(sentences, list)
        self.assertEqual(len(sentences), 1)
        
        # Test tokenize_words
        words = tokenize_words(self.simple_text)
        self.assertIsInstance(words, list)
        self.assertGreater(len(words), 1)
        
        # Test calculate_word_frequencies
        frequencies = calculate_word_frequencies(self.simple_text)
        self.assertIsInstance(frequencies, dict)
        self.assertGreater(len(frequencies), 0)


if __name__ == '__main__':
    unittest.main()
