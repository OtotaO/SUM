"""
test_nltk_utils.py - Unit tests for NLTK utilities

This module provides comprehensive unit tests for the NLTK utilities,
ensuring proper functionality for resource initialization and access.

Author: ototao
License: Apache License 2.0
"""

import unittest
import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Utils.nltk_utils import (
    initialize_nltk,
    get_stopwords,
    get_lemmatizer,
    is_initialized,
    download_specific_resource,
    REQUIRED_RESOURCES
)

class TestNLTKUtils(unittest.TestCase):
    """Tests for the NLTK utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Reset initialization state before each test
        import Utils.nltk_utils
        Utils.nltk_utils._initialized = False
        Utils.nltk_utils._stopwords_cache = None
        Utils.nltk_utils._lemmatizer_cache = None
        
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    @patch('nltk.download')
    def test_initialize_nltk(self, mock_download):
        """Test NLTK initialization."""
        # Configure mock
        mock_download.return_value = True
        
        # Test initialization
        result = initialize_nltk(download_dir=self.temp_dir)
        
        # Check result
        self.assertTrue(result)
        self.assertTrue(is_initialized())
        
        # Check that download was called for each required resource
        self.assertEqual(mock_download.call_count, len(REQUIRED_RESOURCES))
        
        # Check that download was called with correct arguments
        for resource in REQUIRED_RESOURCES:
            mock_download.assert_any_call(
                resource, 
                download_dir=self.temp_dir, 
                quiet=True
            )
    
    @patch('nltk.download')
    def test_initialize_with_custom_resources(self, mock_download):
        """Test NLTK initialization with custom resources."""
        # Configure mock
        mock_download.return_value = True
        
        # Custom resources
        custom_resources = ['punkt', 'stopwords']
        
        # Test initialization
        result = initialize_nltk(
            resources=custom_resources,
            download_dir=self.temp_dir
        )
        
        # Check result
        self.assertTrue(result)
        
        # Check that download was called for each custom resource
        self.assertEqual(mock_download.call_count, len(custom_resources))
        
        # Check that download was called with correct arguments
        for resource in custom_resources:
            mock_download.assert_any_call(
                resource, 
                download_dir=self.temp_dir, 
                quiet=True
            )
    
    @patch('nltk.download')
    def test_initialize_failure(self, mock_download):
        """Test NLTK initialization failure."""
        # Configure mock to fail
        mock_download.side_effect = Exception("Download failed")
        
        # Test initialization
        result = initialize_nltk(download_dir=self.temp_dir)
        
        # Check result
        self.assertFalse(result)
        self.assertFalse(is_initialized())
    
    @patch('nltk.corpus.stopwords.words')
    @patch('nltk.download')
    def test_get_stopwords(self, mock_download, mock_stopwords_words):
        """Test getting stopwords."""
        # Configure mocks
        mock_download.return_value = True
        mock_stopwords_words.return_value = ['the', 'a', 'an']
        
        # Test getting stopwords
        stopwords = get_stopwords()
        
        # Check result
        self.assertIsNotNone(stopwords)
        self.assertIsInstance(stopwords, set)
        self.assertIn('the', stopwords)
        self.assertIn('a', stopwords)
        self.assertIn('an', stopwords)
        
        # Check that initialization happened
        self.assertTrue(is_initialized())
    
    @patch('nltk.stem.WordNetLemmatizer')
    @patch('nltk.download')
    def test_get_lemmatizer(self, mock_download, mock_lemmatizer_class):
        """Test getting lemmatizer."""
        # Configure mocks
        mock_download.return_value = True
        mock_lemmatizer = MagicMock()
        mock_lemmatizer_class.return_value = mock_lemmatizer
        
        # Test getting lemmatizer
        lemmatizer = get_lemmatizer()
        
        # Check result
        self.assertIsNotNone(lemmatizer)
        self.assertEqual(lemmatizer, mock_lemmatizer)
        
        # Check that initialization happened
        self.assertTrue(is_initialized())
    
    @patch('nltk.download')
    def test_download_specific_resource(self, mock_download):
        """Test downloading a specific resource."""
        # Configure mock
        mock_download.return_value = True
        
        # Test downloading a specific resource
        result = download_specific_resource(
            'wordnet',
            download_dir=self.temp_dir
        )
        
        # Check result
        self.assertTrue(result)
        
        # Check that download was called with correct arguments
        mock_download.assert_called_once_with(
            'wordnet',
            download_dir=self.temp_dir,
            quiet=True
        )
    
    @patch('nltk.download')
    def test_download_specific_resource_failure(self, mock_download):
        """Test downloading a specific resource failure."""
        # Configure mock to fail
        mock_download.side_effect = Exception("Download failed")
        
        # Test downloading a specific resource
        result = download_specific_resource(
            'wordnet',
            download_dir=self.temp_dir
        )
        
        # Check result
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
