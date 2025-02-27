"""test_nltk_utils.py - Unit tests for the NLTK utilities

This module provides comprehensive unit tests for the NLTK utilities,
ensuring proper functionality for resource management and initialization.

Author: ototao
License: Apache License 2.0
"""

import unittest
import os
import sys
import tempfile
from pathlib import Path
import shutil

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
        # Create a temporary directory for NLTK data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.nltk_data_dir = Path(self.temp_dir.name)
        
        # Reset initialization state
        import Utils.nltk_utils as nltk_utils
        nltk_utils._initialized = False
        nltk_utils._stopwords_cache = None
        nltk_utils._lemmatizer_cache = None
        
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        self.temp_dir.cleanup()
        
        # Reset initialization state
        import Utils.nltk_utils as nltk_utils
        nltk_utils._initialized = False
        nltk_utils._stopwords_cache = None
        nltk_utils._lemmatizer_cache = None
    
    def test_initialization(self):
        """Test NLTK resource initialization."""
        # Test initialization with minimal resources for faster tests
        result = initialize_nltk(
            resources=['punkt', 'stopwords'],
            download_dir=str(self.nltk_data_dir),
            quiet=True
        )
        
        # Check that initialization was successful
        self.assertTrue(result)
        self.assertTrue(is_initialized())
        
        # Check that resources were downloaded
        self.assertTrue((self.nltk_data_dir / 'corpora' / 'stopwords').exists())
        self.assertTrue((self.nltk_data_dir / 'tokenizers' / 'punkt').exists())
    
    def test_get_stopwords(self):
        """Test getting stopwords."""
        # Initialize with minimal resources
        initialize_nltk(
            resources=['stopwords'],
            download_dir=str(self.nltk_data_dir),
            quiet=True
        )
        
        # Get stopwords
        stopwords = get_stopwords()
        
        # Check that stopwords is a set
        self.assertIsInstance(stopwords, set)
        
        # Check that common stopwords are included
        common_stopwords = {'the', 'a', 'an', 'and', 'in', 'on', 'at'}
        self.assertTrue(common_stopwords.issubset(stopwords))
    
    def test_get_lemmatizer(self):
        """Test getting lemmatizer."""
        # Initialize with minimal resources
        initialize_nltk(
            resources=['wordnet'],
            download_dir=str(self.nltk_data_dir),
            quiet=True
        )
        
        # Get lemmatizer
        lemmatizer = get_lemmatizer()
        
        # Check that lemmatizer is not None
        self.assertIsNotNone(lemmatizer)
        
        # Check that lemmatizer works
        self.assertEqual(lemmatizer.lemmatize('running', 'v'), 'run')
        self.assertEqual(lemmatizer.lemmatize('better', 'a'), 'good')
    
    def test_download_specific_resource(self):
        """Test downloading a specific resource."""
        # Download a specific resource
        result = download_specific_resource(
            resource='punkt',
            download_dir=str(self.nltk_data_dir),
            quiet=True
        )
        
        # Check that download was successful
        self.assertTrue(result)
        
        # Check that resource was downloaded
        self.assertTrue((self.nltk_data_dir / 'tokenizers' / 'punkt').exists())
    
    def test_required_resources(self):
        """Test that required resources are defined."""
        # Check that REQUIRED_RESOURCES is a list
        self.assertIsInstance(REQUIRED_RESOURCES, list)
        
        # Check that common resources are included
        common_resources = {'punkt', 'stopwords', 'wordnet'}
        self.assertTrue(common_resources.issubset(set(REQUIRED_RESOURCES)))
    
    def test_initialization_with_existing_resources(self):
        """Test initialization with existing resources."""
        # First initialization
        initialize_nltk(
            resources=['punkt'],
            download_dir=str(self.nltk_data_dir),
            quiet=True
        )
        
        # Reset initialization state
        import Utils.nltk_utils as nltk_utils
        nltk_utils._initialized = False
        
        # Second initialization
        result = initialize_nltk(
            resources=['punkt'],
            download_dir=str(self.nltk_data_dir),
            quiet=True
        )
        
        # Check that initialization was successful
        self.assertTrue(result)
    
    def test_initialization_with_invalid_resource(self):
        """Test initialization with an invalid resource."""
        # Initialize with an invalid resource
        result = initialize_nltk(
            resources=['nonexistent_resource'],
            download_dir=str(self.nltk_data_dir),
            quiet=True
        )
        
        # Check that initialization failed
        self.assertFalse(result)
        
        # Check that initialization state is still False
        self.assertFalse(is_initialized())
    
    def test_thread_safety(self):
        """Test thread safety of initialization."""
        import threading
        
        # Define a function to initialize NLTK in a thread
        def initialize_in_thread(results, index):
            result = initialize_nltk(
                resources=['punkt'],
                download_dir=str(self.nltk_data_dir),
                quiet=True
            )
            results[index] = result
        
        # Create threads
        num_threads = 5
        threads = []
        results = [False] * num_threads
        
        for i in range(num_threads):
            thread = threading.Thread(target=initialize_in_thread, args=(results, i))
            threads.append(thread)
        
        # Start threads
        for thread in threads:
            thread.start()
        
        # Wait for threads to finish
        for thread in threads:
            thread.join()
        
        # Check that all initializations were successful
        self.assertTrue(all(results))
        
        # Check that initialization state is True
        self.assertTrue(is_initialized())


if __name__ == '__main__':
    unittest.main()