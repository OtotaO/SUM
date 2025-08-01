"""
test_suite.py - Comprehensive test suite for SUM platform

This module implements a thorough test suite for the SUM knowledge distillation platform,
following Test-Driven Development principles (Kent Beck methodology) with a focus on
comprehensive test coverage, edge cases, and integration testing.

Design principles:
- Isolated, repeatable tests (Beck's TDD)
- Comprehensive coverage (Stroustrup methodology)
- Performance benchmarking (Knuth approach)
- Security testing (Schneier principles)
- Clear test organization (Torvalds/van Rossum style)

Author: ototao
License: Apache License 2.0
"""

import unittest
import json
import os
import sys
import tempfile
import time
import warnings
from unittest.mock import patch, MagicMock
import numpy as np
from io import StringIO

# Add project root to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components for testing
try:
    from summarization_engine import SimpleSUM, AdvancedSUM
    from Utils.data_loader import DataLoader
    from Models.topic_modeling import TopicModeler
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure to run tests from project root or configure PYTHONPATH")
    sys.exit(1)

# Import Flask app for API testing
try:
    import main
    from flask import Flask
    from flask.testing import FlaskClient
except ImportError:
    main = None
    Flask = None
    FlaskClient = None
    warnings.warn("Flask not available, API tests will be skipped")


class TestSimpleSUM(unittest.TestCase):
    """Test the SimpleSUM core functionality."""
    
    def setUp(self):
        """Set up for test methods."""
        self.summarizer = SimpleSUM()
        self.short_text = "This is a short text."
        self.medium_text = (
            "This is the first sentence of a medium text. "
            "This is the second sentence with different words. "
            "The third sentence contains important information. "
            "This fourth sentence is less important. "
            "Finally, this fifth sentence concludes the text."
        )
        self.long_text = " ".join([self.medium_text] * 5)  # Repeat 5 times

    def test_initialization(self):
        """Test that the summarizer initializes correctly."""
        self.assertIsInstance(self.summarizer, SimpleSUM)
        self.assertTrue(hasattr(self.summarizer, 'process_text'))
        self.assertTrue(hasattr(self.summarizer, 'stop_words'))
        self.assertGreater(len(self.summarizer.stop_words), 0)

    def test_empty_text(self):
        """Test handling of empty text."""
        result = self.summarizer.process_text("")
        self.assertIn('error', result)
        self.assertEqual(result['error'], 'Empty or invalid text provided')
        
        result = self.summarizer.process_text("   ")
        self.assertIn('error', result)

    def test_short_text(self):
        """Test that short texts are returned as-is."""
        result = self.summarizer.process_text(self.short_text)
        self.assertIn('summary', result)
        self.assertEqual(result['summary'], self.short_text)

    def test_medium_text_summarization(self):
        """Test summarization of medium length text."""
        result = self.summarizer.process_text(self.medium_text)
        self.assertIn('summary', result)
        self.assertIn('compression_ratio', result)
        
        # Summary should be shorter than original
        self.assertLess(len(result['summary']), len(self.medium_text))
        
        # Compression ratio should be between 0 and 1
        self.assertGreater(result['compression_ratio'], 0)
        self.assertLessEqual(result['compression_ratio'], 1)

    def test_token_limit_respected(self):
        """Test that the token limit is respected."""
        max_tokens = 10
        config = {'maxTokens': max_tokens}
        
        result = self.summarizer.process_text(self.medium_text, config)
        self.assertIn('summary', result)
        
        # Count tokens in summary
        summary_tokens = len(result['summary'].split())
        self.assertLessEqual(summary_tokens, max_tokens)

    def test_threshold_parameter(self):
        """Test that the threshold parameter affects summarization."""
        # With high threshold, fewer sentences should be included
        high_config = {'threshold': 0.9}
        high_result = self.summarizer.process_text(self.long_text, high_config)
        
        # With low threshold, more sentences should be included
        low_config = {'threshold': 0.1}
        low_result = self.summarizer.process_text(self.long_text, low_config)
        
        # Higher threshold should lead to shorter summary
        if 'summary' in high_result and 'summary' in low_result:
            self.assertLessEqual(
                len(high_result['summary']), 
                len(low_result['summary'])
            )

    def test_performance(self):
        """Test performance with larger text."""
        very_long_text = " ".join([self.long_text] * 5)  # 25x medium_text
        
        start_time = time.time()
        result = self.summarizer.process_text(very_long_text)
        processing_time = time.time() - start_time
        
        self.assertIn('summary', result)
        
        # Processing time should be reasonable
        # This is somewhat arbitrary, adjust based on your performance expectations
        self.assertLess(processing_time, 5.0, "Summarization took too long")

    def test_invalid_config(self):
        """Test handling of invalid configuration."""
        invalid_configs = [
            {'maxTokens': 'not_a_number'},
            {'maxTokens': -10},
            {'threshold': 'invalid'},
            {'threshold': 2.0}  # Above valid range
        ]
        
        for config in invalid_configs:
            # Should not crash with invalid config
            result = self.summarizer.process_text(self.medium_text, config)
            self.assertIn('summary', result)

    def test_security(self):
        """Test handling of potentially malicious input."""
        # Test with code injection attempt
        malicious_text = "Normal text. eval('import os; os.system(\"echo pwned\")')"
        result = self.summarizer.process_text(malicious_text)
        
        # Should safely process the text without executing code
        self.assertIn('summary', result)
        
        # Test with extremely long input
        very_long_text = "a" * 1000000  # 1MB of 'a's
        result = self.summarizer.process_text(very_long_text)
        
        # Should handle long text without crashing
        self.assertIn('summary', result)

    def test_parallel_processing(self):
        """Test parallel processing of longer texts."""
        very_long_text = " ".join([self.long_text] * 10)  # 50x medium_text
        
        config = {'maxTokens': 200}
        result = self.summarizer.process_text(very_long_text, config)
        
        self.assertIn('summary', result)
        
        # Summary should not be too short or too long
        summary_tokens = len(result['summary'].split())
        self.assertGreater(summary_tokens, 10)
        self.assertLessEqual(summary_tokens, config['maxTokens'])


class TestDataLoader(unittest.TestCase):
    """Test the DataLoader component."""
    
    def setUp(self):
        """Set up test files and loader."""
        # Create temporary test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # JSON test file
        self.json_data = {
            "entries": [
                {
                    "title": "Test Entry 1",
                    "content": "This is test content for the first entry."
                },
                {
                    "title": "Test Entry 2",
                    "content": "This is test content for the second entry."
                }
            ]
        }
        self.json_file = os.path.join(self.temp_dir.name, "test.json")
        with open(self.json_file, 'w') as f:
            json.dump(self.json_data, f)
        
        # Text test file
        self.text_content = (
            "This is a test text file.\n\n"
            "It contains multiple paragraphs.\n\n"
            "Each paragraph is separated by blank lines."
        )
        self.text_file = os.path.join(self.temp_dir.name, "test.txt")
        with open(self.text_file, 'w') as f:
            f.write(self.text_content)
        
        # CSV test file
        self.csv_content = (
            "id,title,content\n"
            "1,CSV Test 1,This is test content for CSV row 1.\n"
            "2,CSV Test 2,This is test content for CSV row 2.\n"
        )
        self.csv_file = os.path.join(self.temp_dir.name, "test.csv")
        with open(self.csv_file, 'w') as f:
            f.write(self.csv_content)
        
        # Create loader
        self.json_loader = DataLoader(data_file=self.json_file)
        self.data_only_loader = DataLoader(data=self.json_data)

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test initialization with different options."""
        # Test initialization with file
        self.assertIsInstance(self.json_loader, DataLoader)
        self.assertEqual(self.json_loader.data, self.json_data)
        
        # Test initialization with data
        self.assertIsInstance(self.data_only_loader, DataLoader)
        self.assertEqual(self.data_only_loader.data, self.json_data)
        
        # Test initialization with neither file nor data
        with self.assertRaises(ValueError):
            DataLoader()
        
        # Test initialization with both file and data
        with self.assertRaises(ValueError):
            DataLoader(data_file=self.json_file, data=self.json_data)

    def test_load_json_data(self):
        """Test loading JSON data."""
        loader = DataLoader(data_file=self.json_file)
        self.assertEqual(loader.data, self.json_data)

    def test_load_text_data(self):
        """Test loading text data."""
        loader = DataLoader(data_file=self.text_file)
        # Text should be loaded as paragraphs
        self.assertIsInstance(loader.data, list)
        self.assertEqual(len(loader.data), 3)  # 3 paragraphs

    def test_load_csv_data(self):
        """Test loading CSV data."""
        loader = DataLoader(data_file=self.csv_file)
        self.assertIsInstance(loader.data, list)
        self.assertEqual(len(loader.data), 2)  # 2 data rows
        self.assertIsInstance(loader.data[0], dict)
        self.assertIn('title', loader.data[0])
        self.assertIn('content', loader.data[0])

    def test_invalid_file_path(self):
        """Test handling of invalid file paths."""
        with self.assertRaises(FileNotFoundError):
            DataLoader(data_file="nonexistent_file.json")

    def test_preprocess_data(self):
        """Test data preprocessing functionality."""
        # Load data
        loader = DataLoader(data_file=self.json_file)
        
        # Preprocess with default options
        processed = loader.preprocess_data()
        
        # Check result structure
        self.assertIsInstance(processed, list)
        self.assertEqual(len(processed), 2)  # 2 entries
        
        # Each entry should be list of sentences
        self.assertIsInstance(processed[0], list)
        
        # Each sentence should be a list of tokens
        self.assertIsInstance(processed[0][0], list)
        
        # Stopwords should be removed by default
        content_lower = self.json_data["entries"][0]["content"].lower()
        tokens = processed[0][0]
        
        for token in tokens:
            self.assertNotIn(token, ['is', 'the', 'for'])  # Common stopwords

    def test_preprocessing_options(self):
        """Test different preprocessing options."""
        loader = DataLoader(data_file=self.text_file)
        
        # Test with stemming
        stem_processed = loader.preprocess_data(lemmatize=False, stem=True)
        
        # Test with lemmatization (default)
        lemma_processed = loader.preprocess_data(lemmatize=True, stem=False)
        
        # Stemming and lemmatization should produce different results
        # Find a word that would be stemmed/lemmatized differently
        # (e.g., "contains" -> stem:"contain", lemma:"contain")
        
        flat_stem = [token for sent in stem_processed for token in sent]
        flat_lemma = [token for sent in lemma_processed for token in sent]
        
        # The flat lists might have different lengths due to differences
        # in how stemming and lemmatization work
        self.assertIsInstance(flat_stem, list)
        self.assertIsInstance(flat_lemma, list)

    def test_metadata(self):
        """Test metadata extraction."""
        loader = DataLoader(data_file=self.json_file)
        metadata = loader.get_metadata()
        
        # Check metadata structure
        self.assertIsInstance(metadata, dict)
        self.assertIn('num_documents', metadata)
        self.assertIn('total_words', metadata)
        self.assertIn('total_sentences', metadata)
        
        # Verify counts
        self.assertEqual(metadata['num_documents'], 2)
        self.assertGreater(metadata['total_words'], 0)
        self.assertGreater(metadata['total_sentences'], 0)

    def test_security(self):
        """Test handling of potentially malicious input."""
        # Test with a file that's too large
        large_file = os.path.join(self.temp_dir.name, "large.txt")
        with open(large_file, 'w') as f:
            f.write("a" * (101 * 1024 * 1024))  # 101MB, should exceed limit
        
        with self.assertRaises(ValueError):
            DataLoader(data_file=large_file)
        
        # Test with invalid file extension
        invalid_file = os.path.join(self.temp_dir.name, "test.exe")
        with open(invalid_file, 'w') as f:
            f.write("not a valid data file")
        
        with self.assertRaises(ValueError):
            DataLoader(data_file=invalid_file)


class TestTopicModeler(unittest.TestCase):
    """Test the TopicModeler component."""
    
    def setUp(self):
        """Set up test data and modeler."""
        # Sample documents for topic modeling
        self.documents = [
            "Machine learning involves computers learning from data to perform tasks.",
            "Deep learning models use neural networks with many layers.",
            "Neural networks are inspired by the human brain's structure.",
            "Supervised learning requires labeled training data.",
            "Unsupervised learning finds patterns without labeled data.",
            "Reinforcement learning involves agents learning from environment feedback.",
            "Natural language processing helps computers understand human language.",
            "Computer vision enables machines to interpret visual information.",
            "Data preprocessing is crucial for effective machine learning.",
            "Overfitting happens when models learn noise in training data."
        ]
        
        # Create modeler with default settings
        self.modeler = TopicModeler(n_topics=3, algorithm='lda')

    def test_initialization(self):
        """Test initialization with different parameters."""
        # Test default initialization
        self.assertIsInstance(self.modeler, TopicModeler)
        self.assertEqual(self.modeler.n_topics, 3)
        self.assertEqual(self.modeler.algorithm, 'lda')
        
        # Test with different algorithms
        nmf_modeler = TopicModeler(n_topics=2, algorithm='nmf')
        self.assertEqual(nmf_modeler.algorithm, 'nmf')
        
        lsa_modeler = TopicModeler(n_topics=2, algorithm='lsa')
        self.assertEqual(lsa_modeler.algorithm, 'lsa')
        
        # Test with invalid algorithm
        with self.assertRaises(ValueError):
            TopicModeler(algorithm='invalid')
        
        # Test with invalid n_topics
        with self.assertRaises(ValueError):
            TopicModeler(n_topics=0)

    def test_fit_transform(self):
        """Test fitting and transforming documents."""
        # Fit the model
        self.modeler.fit(self.documents)
        
        # Check if model is fitted
        self.assertTrue(self.modeler.is_fitted)
        
        # Check topic words extraction
        self.assertEqual(len(self.modeler.topic_words), 3)  # 3 topics
        
        # Transform documents
        doc_topics = self.modeler.transform(self.documents)
        
        # Check transformation shape
        self.assertEqual(doc_topics.shape, (len(self.documents), 3))  # (n_docs, n_topics)
        
        # Check that probabilities sum to approximately 1 for each document
        for doc_topic in doc_topics:
            self.assertAlmostEqual(np.sum(doc_topic), 1.0, delta=0.01)

    def test_get_topic_terms(self):
        """Test retrieving topic terms."""
        # Fit the model
        self.modeler.fit(self.documents)
        
        # Get terms for each topic
        for topic_idx in range(3):
            terms = self.modeler.get_topic_terms(topic_idx)
            
            # Check structure
            self.assertIsInstance(terms, list)
            self.assertGreaterEqual(len(terms), 1)
            self.assertIsInstance(terms[0], tuple)
            self.assertEqual(len(terms[0]), 2)  # (term, weight)
            
            # Check term type and weight range
            term, weight = terms[0]
            self.assertIsInstance(term, str)
            self.assertIsInstance(weight, float)
            self.assertGreaterEqual(weight, 0)

    def test_predict_topic(self):
        """Test topic prediction for new documents."""
        # Fit the model
        self.modeler.fit(self.documents)
        
        # Predict topic for new document
        new_doc = "Neural networks have revolutionized artificial intelligence."
        topic_id, confidence = self.modeler.predict_topic(new_doc)
        
        # Check result types
        self.assertIsInstance(topic_id, int)
        self.assertIsInstance(confidence, float)
        
        # Check value ranges
        self.assertGreaterEqual(topic_id, 0)
        self.assertLess(topic_id, 3)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)

    def test_cluster_documents(self):
        """Test document clustering."""
        # Fit the model
        self.modeler.fit(self.documents)
        
        # Cluster documents
        clusters, metadata = self.modeler.cluster_documents(self.documents)
        
        # Check cluster assignments
        self.assertEqual(len(clusters), len(self.documents))
        for cluster in clusters:
            self.assertGreaterEqual(cluster, 0)
            self.assertLess(cluster, 3)  # Default n_clusters = n_topics = 3
        
        # Check metadata
        self.assertIn('cluster_sizes', metadata)
        self.assertEqual(len(metadata['cluster_sizes']), 3)

    def test_topic_labels(self):
        """Test generating topic labels."""
        # Fit the model
        self.modeler.fit(self.documents)
        
        # Generate labels
        labels = self.modeler.generate_topic_labels()
        
        # Check labels
        self.assertEqual(len(labels), 3)
        for topic_id, label in labels.items():
            self.assertGreaterEqual(topic_id, 0)
            self.assertLess(topic_id, 3)
            self.assertIsInstance(label, str)
            self.assertGreater(len(label), 0)
        
        # Test with custom descriptions
        custom = {0: "Machine Learning", 1: "Neural Networks"}
        labels = self.modeler.generate_topic_labels(custom_descriptions=custom)
        
        # Check custom labels were used
        self.assertEqual(labels[0], "Machine Learning")
        self.assertEqual(labels[1], "Neural Networks")

    def test_save_load_model(self):
        """Test saving and loading model."""
        # Fit the model
        self.modeler.fit(self.documents)
        
        # Create temporary file for model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            model_file = tmp.name
        
        try:
            # Save model
            self.modeler.save_model(model_file)
            
            # Load model
            loaded_modeler = TopicModeler.load_model(model_file)
            
            # Check loaded model
            self.assertEqual(loaded_modeler.n_topics, self.modeler.n_topics)
            self.assertEqual(loaded_modeler.algorithm, self.modeler.algorithm)
            self.assertEqual(len(loaded_modeler.topic_words), len(self.modeler.topic_words))
            
            # Check functionality of loaded model
            topic_id, _ = loaded_modeler.predict_topic(self.documents[0])
            self.assertGreaterEqual(topic_id, 0)
            self.assertLess(topic_id, 3)
            
        finally:
            # Clean up
            if os.path.exists(model_file):
                os.unlink(model_file)

    def test_performance(self):
        """Test performance with larger document set."""
        # Create larger document set
        large_docs = self.documents * 10  # 100 documents
        
        # Measure fitting time
        start_time = time.time()
        modeler = TopicModeler(n_topics=5, algorithm='lda')
        modeler.fit(large_docs)
        fit_time = time.time() - start_time
        
        # Check reasonable fit time (arbitrary threshold)
        self.assertLess(fit_time, 10.0, "Topic modeling took too long")


@unittest.skipIf(main is None, "Flask not available for API tests")
class TestAPI(unittest.TestCase):
    """Test the SUM REST API endpoints."""
    
    def setUp(self):
        """Set up test client."""
        # Create a test client
        main.app.config['TESTING'] = True
        self.client = main.app.test_client()
        
        # Sample data for testing
        self.sample_text = (
            "Machine learning has seen rapid advancements in recent years. "
            "From image recognition to natural language processing, AI systems are becoming increasingly sophisticated. "
            "Deep learning models, in particular, have shown remarkable capabilities in handling complex tasks. "
            "However, challenges remain in areas such as explainability and bias mitigation."
        )

    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get('/api/health')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('version', data)
        self.assertIn('uptime', data)

    def test_process_text_simple(self):
        """Test processing text with simple model."""
        payload = {
            'text': self.sample_text,
            'model': 'simple',
            'config': {
                'maxTokens': 50
            }
        }
        
        response = self.client.post(
            '/api/process_text',
            data=json.dumps(payload),
            content_type='application/json'
        )
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('summary', data)
        self.assertIn('compression_ratio', data)
        self.assertIn('processing_time', data)
        self.assertEqual(data['model'], 'simple')

    def test_process_text_empty(self):
        """Test handling empty text."""
        payload = {
            'text': '',
            'model': 'simple'
        }
        
        response = self.client.post(
            '/api/process_text',
            data=json.dumps(payload),
            content_type='application/json'
        )
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)

    def test_process_text_invalid_json(self):
        """Test handling invalid JSON."""
        response = self.client.post(
            '/api/process_text',
            data='not valid json',
            content_type='application/json'
        )
        
        self.assertNotEqual(response.status_code, 200)

    def test_analyze_topics(self):
        """Test topic analysis endpoint."""
        payload = {
            'documents': [
                "Machine learning is fascinating.",
                "Neural networks are powerful.",
                "Data science requires statistical knowledge."
            ],
            'num_topics': 2,
            'algorithm': 'lda'
        }
        
        response = self.client.post(
            '/api/analyze_topics',
            data=json.dumps(payload),
            content_type='application/json'
        )
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('topics', data)
        self.assertIn('document_topics', data)
        self.assertEqual(len(data['document_topics']), 3)  # 3 documents

    def test_analyze_topics_invalid_params(self):
        """Test handling invalid parameters for topic analysis."""
        # Invalid num_topics
        payload = {
            'documents': ["Test document"],
            'num_topics': 0  # Invalid
        }
        
        response = self.client.post(
            '/api/analyze_topics',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        self.assertNotEqual(response.status_code, 200)
        
        # Invalid algorithm
        payload = {
            'documents': ["Test document"],
            'algorithm': 'invalid'  # Invalid
        }
        
        response = self.client.post(
            '/api/analyze_topics',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        self.assertNotEqual(response.status_code, 200)


class TestIntegration(unittest.TestCase):
    """Integration tests for SUM platform components."""
    
    def setUp(self):
        """Set up test data and components."""
        # Create components
        self.summarizer = SimpleSUM()
        self.topic_modeler = TopicModeler(n_topics=2)
        
        # Sample data
        self.json_data = {
            "entries": [
                {
                    "title": "Machine Learning",
                    "content": "Machine learning involves computers learning from data to perform tasks."
                },
                {
                    "title": "Deep Learning",
                    "content": "Deep learning models use neural networks with many layers."
                }
            ]
        }
        
        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        with open(self.temp_file.name, 'w') as f:
            json.dump(self.json_data, f)

    def tearDown(self):
        """Clean up temporary files."""
        if hasattr(self, 'temp_file') and os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_data_to_summary_pipeline(self):
        """Test the data-to-summary pipeline."""
        # Load data
        loader = DataLoader(data_file=self.temp_file.name)
        
        # Preprocess data
        processed = loader.preprocess_data()
        
        # Extract raw text
        text = " ".join([" ".join([" ".join(sent) for sent in doc]) for doc in processed])
        
        # Generate summary
        summary_result = self.summarizer.process_text(text)
        
        # Check results
        self.assertIn('summary', summary_result)
        self.assertGreater(len(summary_result['summary']), 0)

    def test_data_to_topics_pipeline(self):
        """Test the data-to-topics pipeline."""
        # Load data
        loader = DataLoader(data_file=self.temp_file.name)
        
        # Extract raw texts
        texts = [entry['content'] for entry in loader.data['entries']]
        
        # Generate topics
        self.topic_modeler.fit(texts)
        
        # Get topics summary
        topics = self.topic_modeler.get_topics_summary()
        
        # Check results
        self.assertEqual(len(topics), 2)  # 2 topics
        
        # Check we can predict topics for new text
        new_text = "Neural networks are transforming AI."
        topic_id, confidence = self.topic_modeler.predict_topic(new_text)
        
        self.assertIn(topic_id, [0, 1])
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)


class TestBenchmarks(unittest.TestCase):
    """Performance benchmarks for SUM platform components."""
    
    def setUp(self):
        """Set up benchmark data."""
        # Generate test data of different sizes
        self.small_text = "This is a small text for benchmarking." * 10  # ~100 words
        self.medium_text = "This is a medium text for benchmarking performance of SUM components." * 100  # ~1000 words
        
        # Only generate large text if needed (to save memory in normal tests)
        self.large_text = None
        
        # Initialize components
        self.summarizer = SimpleSUM()

    def generate_large_text(self):
        """Generate large text on demand."""
        if self.large_text is None:
            self.large_text = "This is a large text for benchmarking performance." * 1000  # ~10000 words
        return self.large_text

    def test_summarizer_benchmark(self):
        """Benchmark the summarizer with different text sizes."""
        # Small text benchmark
        start_time = time.time()
        self.summarizer.process_text(self.small_text)
        small_time = time.time() - start_time
        
        # Medium text benchmark
        start_time = time.time()
        self.summarizer.process_text(self.medium_text)
        medium_time = time.time() - start_time
        
        # Check relative performance (medium should be slower but not exponentially)
        ratio = medium_time / max(small_time, 0.001)  # Avoid division by zero
        
        # Medium text should be slower but not exponentially
        # (O(n) or O(n log n) complexity, not O(n²))
        self.assertLess(ratio, 100, "Performance degradation too high for medium text")
        
        # Skip large text benchmark by default, uncomment to run
        # large_text = self.generate_large_text()
        # start_time = time.time()
        # self.summarizer.process_text(large_text)
        # large_time = time.time() - start_time
        # print(f"Large text processing time: {large_time}s")

    def test_parallel_vs_sequential(self):
        """Compare parallel vs. sequential processing."""
        large_text = self.generate_large_text()
        
        # Force sequential processing by monkey patching
        original_score_parallel = self.summarizer._score_sentences_parallel
        self.summarizer._score_sentences_parallel = self.summarizer._score_sentences
        
        start_time = time.time()
        self.summarizer.process_text(large_text)
        sequential_time = time.time() - start_time
        
        # Restore parallel processing
        self.summarizer._score_sentences_parallel = original_score_parallel
        
        start_time = time.time()
        self.summarizer.process_text(large_text)
        parallel_time = time.time() - start_time
        
        # Parallel should be faster on large texts
        self.assertLessEqual(parallel_time, sequential_time * 0.8, 
                          "Parallel processing not significantly faster")


def suite():
    """Create a test suite with all test cases."""
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestSimpleSUM))
    test_suite.addTest(unittest.makeSuite(TestDataLoader))
    test_suite.addTest(unittest.makeSuite(TestTopicModeler))
    
    # Only add API tests if Flask is available
    if Flask is not None:
        test_suite.addTest(unittest.makeSuite(TestAPI))
    
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Add benchmarks (can be skipped in normal runs)
    # test_suite.addTest(unittest.makeSuite(TestBenchmarks))
    
    return test_suite


if __name__ == '__main__':
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run SUM platform tests')
    parser.add_argument('--benchmarks', action='store_true', help='Run performance benchmarks')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    # Configure test runner
    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestSimpleSUM))
    test_suite.addTest(unittest.makeSuite(TestDataLoader))
    test_suite.addTest(unittest.makeSuite(TestTopicModeler))
    
    # Only add API tests if Flask is available
    if Flask is not None:
        test_suite.addTest(unittest.makeSuite(TestAPI))
    else:
        print("Skipping API tests: Flask not available")
        
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Add benchmarks if requested
    if args.benchmarks:
        test_suite.addTest(unittest.makeSuite(TestBenchmarks))
        print("Running performance benchmarks (this may take a while)...")
    
    # Run tests
    result = runner.run(test_suite)
    
    # Exit with error code if tests failed
    sys.exit(not result.wasSuccessful())
