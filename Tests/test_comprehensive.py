"""
test_comprehensive.py - Comprehensive test suite for SUM

This module provides a comprehensive suite of unit tests and integration tests
for the SUM knowledge distillation platform.

Design principles:
- Test-driven development (Beck)
- Robust test coverage (Stroustrup)
- Clean test structure (Torvalds/van Rossum)
- Security testing (Schneier)
- Performance benchmarking (Knuth)

Author: ototao
License: Apache License 2.0
"""

import unittest
import sys
import os
import json
import time
import logging
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import SUM components
from SUM import SimpleSUM, MagnumOpusSUM
from Utils.data_loader import DataLoader
from Models.topic_modeling import TopicModeler

# Disable unnecessary logging during tests
logging.disable(logging.CRITICAL)

class TestSimpleSUM(unittest.TestCase):
    """Tests for the SimpleSUM class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.summarizer = SimpleSUM()
        self.test_text = """
        Machine learning has seen rapid advancements in recent years. From image recognition to 
        natural language processing, AI systems are becoming increasingly sophisticated. Deep learning 
        models, in particular, have shown remarkable capabilities in handling complex tasks. However, 
        challenges remain in areas such as explainability and bias mitigation. As the field continues 
        to evolve, researchers are developing new approaches to address these limitations and expand 
        the applications of machine learning across various domains.
        """
        
    def test_initialization(self):
        """Test that summarizer initializes properly."""
        self.assertIsInstance(self.summarizer, SimpleSUM)
        self.assertTrue(hasattr(self.summarizer, 'stop_words'))
        self.assertTrue(len(self.summarizer.stop_words) > 0)
        
    def test_preprocessing(self):
        """Test text preprocessing."""
        sentences, words, word_freq = self.summarizer._preprocess_text(self.test_text)
        
        # Check that sentences were identified
        self.assertTrue(len(sentences) > 0)
        
        # Check that words were tokenized
        self.assertTrue(len(words) > 0)
        
        # Check that frequencies were calculated
        self.assertTrue(len(word_freq) > 0)
        self.assertIn('learning', word_freq)
        
    def test_empty_text(self):
        """Test handling of empty text."""
        result = self.summarizer.process_text("")
        self.assertIn('error', result)
        
    def test_short_text(self):
        """Test handling of short text."""
        short_text = "This is a short text."
        result = self.summarizer.process_text(short_text)
        self.assertEqual(result['summary'], short_text)
        
    def test_parameter_validation(self):
        """Test validation of parameters."""
        # Test with very high max_tokens (should be capped)
        result = self.summarizer.process_text(self.test_text, {'maxTokens': 1000})
        self.assertIn('summary', result)
        
        # Test with very low max_tokens
        result = self.summarizer.process_text(self.test_text, {'maxTokens': 5})
        self.assertIn('summary', result)
        
    def test_summarization(self):
        """Test the actual summarization functionality."""
        result = self.summarizer.process_text(self.test_text, {'maxTokens': 50})
        
        # Check that we got a summary
        self.assertIn('summary', result)
        self.assertTrue(len(result['summary']) > 0)
        
        # Check that we got tags
        self.assertIn('tags', result)
        self.assertTrue(len(result['tags']) > 0)
        
        # Check that summary is shorter than input
        self.assertLess(len(result['summary']), len(self.test_text))
        
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        result = self.summarizer.process_text(self.test_text)
        self.assertIn('compression_ratio', result)
        self.assertTrue(0 < result['compression_ratio'] <= 1)
        
    def test_parallel_processing(self):
        """Test parallel sentence scoring for longer texts."""
        # Create a longer text by repeating
        long_text = self.test_text * 5
        
        start_time = time.time()
        result = self.summarizer.process_text(long_text, {'maxTokens': 100})
        processing_time = time.time() - start_time
        
        # Ensure we got a result
        self.assertIn('summary', result)
        
        # This is not a strict performance test, just a sanity check
        self.assertLess(processing_time, 5.0, "Processing took too long")
        

class TestMagnumOpusSUM(unittest.TestCase):
    """Tests for the MagnumOpusSUM class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.summarizer = MagnumOpusSUM()
        self.test_text = """
        Climate change is having a profound impact on global biodiversity. Rising temperatures and 
        changing weather patterns are altering ecosystems worldwide. Many species are struggling 
        to adapt, leading to population declines and potential extinctions. Conservation efforts 
        are crucial to mitigate these effects and preserve Earth's biodiversity. Scientists are 
        monitoring these changes and developing strategies to protect vulnerable species.
        """
        
    def test_initialization(self):
        """Test that summarizer initializes properly."""
        self.assertIsInstance(self.summarizer, MagnumOpusSUM)
        
    def test_empty_text(self):
        """Test handling of empty text."""
        result = self.summarizer.process_text("")
        self.assertIn('error', result)
        
    def test_tag_summary(self):
        """Test tag-based summary generation."""
        tags = self.summarizer.generate_tag_summary(self.test_text)
        self.assertTrue(len(tags) > 0)
        self.assertIsInstance(tags, list)
        
    def test_sentence_summary(self):
        """Test sentence-based summary generation."""
        summary = self.summarizer.generate_sentence_summary(self.test_text)
        self.assertTrue(len(summary) > 0)
        self.assertIsInstance(summary, str)
        self.assertLess(len(summary), len(self.test_text))
        
    def test_entity_recognition(self):
        """Test entity recognition."""
        entities = self.summarizer.identify_entities(self.test_text)
        self.assertIsInstance(entities, list)
        
    def test_sentiment_analysis(self):
        """Test sentiment analysis."""
        sentiment = self.summarizer.sentiment_analysis(self.test_text)
        self.assertIn(sentiment, ['Positive', 'Negative', 'Neutral'])
        
    def test_comprehensive_processing(self):
        """Test comprehensive text processing."""
        result = self.summarizer.process_text(self.test_text, {'include_analysis': True})
        
        # Check essential outputs
        self.assertIn('summary', result)
        self.assertIn('sum', result)
        self.assertIn('tags', result)
        
        # Check that we got a summary
        self.assertTrue(len(result['summary']) > 0)
        
        # Check that condensed summary is shorter
        self.assertLessEqual(len(result['sum']), len(result['summary']))


class TestDataLoader(unittest.TestCase):
    """Tests for the DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test files
        self.create_test_files()
        
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory and files
        shutil.rmtree(self.temp_dir)
        
    def create_test_files(self):
        """Create test files for different formats."""
        # JSON file
        self.json_file = os.path.join(self.temp_dir, 'test.json')
        json_data = {
            "entries": [
                {
                    "title": "Test Entry 1",
                    "content": "This is test content for entry 1."
                },
                {
                    "title": "Test Entry 2",
                    "content": "This is test content for entry 2."
                }
            ]
        }
        with open(self.json_file, 'w') as f:
            json.dump(json_data, f)
            
        # TXT file
        self.txt_file = os.path.join(self.temp_dir, 'test.txt')
        with open(self.txt_file, 'w') as f:
            f.write("This is a test text file.\nIt has multiple lines.\nEach line contains text.")
            
        # CSV file
        self.csv_file = os.path.join(self.temp_dir, 'test.csv')
        with open(self.csv_file, 'w') as f:
            f.write("id,name,value\n1,test1,100\n2,test2,200\n3,test3,300")
            
    def test_load_json(self):
        """Test loading JSON data."""
        loader = DataLoader(data_file=self.json_file)
        data = loader.load_data()
        
        self.assertIsInstance(data, dict)
        self.assertIn('entries', data)
        self.assertEqual(len(data['entries']), 2)
        
    def test_load_txt(self):
        """Test loading TXT data."""
        loader = DataLoader(data_file=self.txt_file)
        data = loader.load_data()
        
        self.assertIsInstance(data, str)
        self.assertTrue('test text file' in data)
        
    def test_load_csv(self):
        """Test loading CSV data."""
        loader = DataLoader(data_file=self.csv_file)
        data = loader.load_data()
        
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 4)  # Header + 3 rows
        
    def test_preprocess_data(self):
        """Test preprocessing data."""
        loader = DataLoader(data_file=self.txt_file)
        processed_data = loader.preprocess_data()
        
        self.assertIsInstance(processed_data, list)
        self.assertTrue(len(processed_data) > 0)
        
    def test_get_metadata(self):
        """Test extracting metadata."""
        loader = DataLoader(data_file=self.json_file)
        metadata = loader.get_metadata()
        
        self.assertIsInstance(metadata, dict)
        self.assertIn('source', metadata)
        self.assertIn('type', metadata)
        self.assertIn('size', metadata)
        
    def test_invalid_file(self):
        """Test handling of invalid file."""
        non_existent_file = os.path.join(self.temp_dir, 'non_existent.txt')
        
        with self.assertRaises(FileNotFoundError):
            loader = DataLoader(data_file=non_existent_file)
            loader.load_data()


class TestTopicModeler(unittest.TestCase):
    """Tests for the TopicModeler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_texts = [
            "Machine learning uses algorithms to parse data, learn from it, and make informed decisions.",
            "Deep learning is a subset of machine learning that uses neural networks with many layers.",
            "Neural networks are computing systems inspired by the biological neural networks in brains.",
            "Reinforcement learning is training algorithms to make suitable actions to maximize reward.",
            "Supervised learning algorithms build a model based on labeled training data.",
            "Climate change is a long-term change in Earth's average weather patterns.",
            "Global warming is the long-term heating of Earth's climate system due to human activities.",
            "Renewable energy comes from sources that are naturally replenished on a human timescale.",
            "Solar power is the conversion of energy from sunlight into electricity.",
            "Wind power is the use of air flow to rotate wind turbines to generate electricity."
        ]
        
    def test_lda_modeling(self):
        """Test LDA topic modeling."""
        tm = TopicModeler(n_topics=2, algorithm='lda')
        tm.fit(self.test_texts)
        
        # Check topics
        topics = tm.get_topics()
        self.assertEqual(len(topics), 2)
        self.assertTrue(all(len(topic) > 0 for topic in topics))
        
        # Check document-topic distribution
        doc_topics = tm.transform(self.test_texts)
        self.assertEqual(doc_topics.shape, (len(self.test_texts), 2))
        
    def test_nmf_modeling(self):
        """Test NMF topic modeling."""
        tm = TopicModeler(n_topics=2, algorithm='nmf')
        tm.fit(self.test_texts)
        
        # Check topics
        topics = tm.get_topics()
        self.assertEqual(len(topics), 2)
        
    def test_lsa_modeling(self):
        """Test LSA topic modeling."""
        tm = TopicModeler(n_topics=2, algorithm='lsa')
        tm.fit(self.test_texts)
        
        # Check topics
        topics = tm.get_topics()
        self.assertEqual(len(topics), 2)
        
    def test_topic_summary(self):
        """Test topic summary generation."""
        tm = TopicModeler(n_topics=2, algorithm='lda')
        tm.fit(self.test_texts)
        
        summary = tm.get_topics_summary()
        self.assertIn('algorithm', summary)
        self.assertIn('num_topics', summary)
        self.assertIn('topics', summary)
        self.assertEqual(len(summary['topics']), 2)
        
    def test_invalid_algorithm(self):
        """Test handling of invalid algorithm."""
        with self.assertRaises(ValueError):
            tm = TopicModeler(algorithm='invalid_algorithm')
            

class TestWebService(unittest.TestCase):
    """Tests for the web service API."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Import Flask app
        from main import app
        
        # Configure app for testing
        app.config['TESTING'] = True
        self.app = app.test_client()
        
    @patch('main.simple_summarizer.process_text')
    def test_process_text_api(self, mock_process_text):
        """Test the process_text API endpoint."""
        # Mock summarizer response
        mock_process_text.return_value = {
            'summary': 'Test summary',
            'sum': 'Test condensed',
            'tags': ['tag1', 'tag2']
        }
        
        # Test request
        response = self.app.post('/api/process_text',
                               json={
                                   'text': 'Test text',
                                   'model': 'simple',
                                   'config': {'maxTokens': 100}
                               })
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('summary', data)
        self.assertIn('processing_time', data)
        
    def test_health_check_api(self):
        """Test the health check API endpoint."""
        response = self.app.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        

class BenchmarkTests(unittest.TestCase):
    """Performance benchmark tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simple_sum = SimpleSUM()
        self.magnum_opus_sum = MagnumOpusSUM()
        
        # Generate a long text for benchmarking
        paragraph = """
        Machine learning has seen rapid advancements in recent years. From image recognition to 
        natural language processing, AI systems are becoming increasingly sophisticated. Deep learning 
        models, in particular, have shown remarkable capabilities in handling complex tasks. However, 
        challenges remain in areas such as explainability and bias mitigation. As the field continues 
        to evolve, researchers are developing new approaches to address these limitations and expand 
        the applications of machine learning across various domains.
        """
        self.long_text = paragraph * 10
        
    def test_simple_sum_performance(self):
        """Benchmark SimpleSUM performance."""
        start_time = time.time()
        result = self.simple_sum.process_text(self.long_text, {'maxTokens': 100})
        duration = time.time() - start_time
        
        print(f"\nSimpleSUM processed {len(self.long_text)} chars in {duration:.4f} seconds")
        self.assertLess(duration, 5.0, "SimpleSUM processing took too long")
        
    def test_magnum_opus_sum_performance(self):
        """Benchmark MagnumOpusSUM performance."""
        start_time = time.time()
        result = self.magnum_opus_sum.process_text(self.long_text, {'maxTokens': 100})
        duration = time.time() - start_time
        
        print(f"MagnumOpusSUM processed {len(self.long_text)} chars in {duration:.4f} seconds")
        self.assertLess(duration, 10.0, "MagnumOpusSUM processing took too long")
        

if __name__ == '__main__':
    unittest.main()
