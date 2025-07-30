"""
Test suite for SUM summarization engines.

This module contains comprehensive tests for all SUM engine components
including SimpleSUM, MagnumOpusSUM, and HierarchicalDensificationEngine.

Author: ototao
License: Apache License 2.0
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SUM import SimpleSUM, MagnumOpusSUM, HierarchicalDensificationEngine
from wordcloud import WordCloud


class TestSimpleSUM(unittest.TestCase):
    """Test cases for SimpleSUM engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.summarizer = SimpleSUM()
        self.test_text = """
        Machine learning and artificial intelligence have revolutionized the technology industry. 
        Companies like Google, Microsoft, and Amazon are leading the way in developing AI solutions. 
        Deep learning, a subset of machine learning, has enabled breakthroughs in computer vision, 
        natural language processing, and speech recognition. These technologies are transforming 
        industries from healthcare to finance, creating new opportunities and challenges.
        """
    
    def test_process_text_basic(self):
        """Test basic text processing functionality."""
        result = self.summarizer.process_text(self.test_text)
        
        # Check required keys are present
        required_keys = ['summary', 'sum', 'tags']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check types
        self.assertIsInstance(result['summary'], str)
        self.assertIsInstance(result['sum'], str)
        self.assertIsInstance(result['tags'], list)
        
        # Check content is not empty
        self.assertTrue(len(result['summary']) > 0)
        self.assertTrue(len(result['tags']) > 0)
    
    def test_empty_text_handling(self):
        """Test handling of empty or invalid text."""
        # Test empty string
        result = self.summarizer.process_text("")
        self.assertIn('error', result)
        
        # Test None
        result = self.summarizer.process_text(None)
        self.assertIn('error', result)
        
        # Test non-string
        result = self.summarizer.process_text(123)
        self.assertIn('error', result)
    
    def test_short_text_handling(self):
        """Test handling of very short text."""
        short_text = "Hello world."
        result = self.summarizer.process_text(short_text)
        
        # Short text should be returned as-is
        self.assertEqual(result['summary'], short_text)
        self.assertIsInstance(result['tags'], list)
    
    def test_configuration_parameters(self):
        """Test processing with different configuration parameters."""
        config = {
            'maxTokens': 50,
            'threshold': 0.5
        }
        
        result = self.summarizer.process_text(self.test_text, config)
        self.assertIn('summary', result)
        
        # Check if summary respects token limit roughly
        summary_tokens = len(result['summary'].split())
        self.assertLessEqual(summary_tokens, 60)  # Allow some tolerance


class TestMagnumOpusSUM(unittest.TestCase):
    """Test cases for MagnumOpusSUM engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.summarizer = MagnumOpusSUM()
        self.test_text = """
        Machine learning and artificial intelligence have revolutionized the technology industry. 
        Companies like Google, Microsoft, and Amazon are leading the way in developing AI solutions. 
        Deep learning, a subset of machine learning, has enabled breakthroughs in computer vision, 
        natural language processing, and speech recognition. These technologies are transforming 
        industries from healthcare to finance, creating new opportunities and challenges.
        """
    
    def test_process_text_with_analysis(self):
        """Test advanced processing with analysis enabled."""
        config = {'include_analysis': True}
        result = self.summarizer.process_text(self.test_text, config)
        
        # Check all expected keys are present
        expected_keys = ['tags', 'sum', 'summary', 'entities', 'main_concept',
                        'sentiment', 'keywords', 'language']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check types of returned values
        self.assertIsInstance(result['tags'], list)
        self.assertIsInstance(result['sum'], str)
        self.assertIsInstance(result['summary'], str)
        self.assertIsInstance(result['entities'], list)
        self.assertIsInstance(result['main_concept'], str)
        self.assertIsInstance(result['sentiment'], str)
        self.assertIsInstance(result['keywords'], list)
        self.assertIsInstance(result['language'], str)
    
    def test_identify_entities(self):
        """Test named entity recognition."""
        entities = self.summarizer.identify_entities(self.test_text)
        self.assertIsInstance(entities, list)
        self.assertTrue(all(isinstance(entity, tuple) for entity in entities))
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis."""
        sentiment = self.summarizer.sentiment_analysis(self.test_text)
        self.assertIn(sentiment, ['Positive', 'Negative', 'Neutral'])
        
        # Test with clearly positive text
        positive_text = "This is excellent! I love it. Amazing work!"
        self.assertEqual(self.summarizer.sentiment_analysis(positive_text), 'Positive')
        
        # Test with clearly negative text
        negative_text = "This is terrible! I hate it. Awful work!"
        self.assertEqual(self.summarizer.sentiment_analysis(negative_text), 'Negative')
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        keywords = self.summarizer.extract_keywords(self.test_text)
        self.assertIsInstance(keywords, list)
        self.assertEqual(len(keywords), 5)  # default num_tags is 5
    
    def test_detect_language(self):
        """Test language detection."""
        lang = self.summarizer.detect_language(self.test_text)
        self.assertEqual(lang, 'en')
        
        # Test with non-English text
        spanish_text = "Hola mundo. Esto es una prueba."
        self.assertEqual(self.summarizer.detect_language(spanish_text), 'es')
    
    def test_generate_summaries(self):
        """Test various summary generation methods."""
        # Test tag summary
        tags = self.summarizer.generate_tag_summary(self.test_text)
        self.assertIsInstance(tags, list)
        self.assertTrue(len(tags) > 0)
        
        # Test sentence summary
        summary = self.summarizer.generate_sentence_summary(self.test_text)
        self.assertIsInstance(summary, str)
        self.assertTrue(len(summary) < len(self.test_text))


class TestHierarchicalDensificationEngine(unittest.TestCase):
    """Test cases for HierarchicalDensificationEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = HierarchicalDensificationEngine()
        self.test_text = """
        The essence of wisdom lies not in the accumulation of knowledge, but in understanding the 
        nature of reality itself. Truth is like a mirror - it reflects not what we wish to see, 
        but what actually is. In seeking knowledge, we often find that the more we learn, the less 
        we realize we know. This paradox is fundamental to human consciousness and the eternal quest 
        for meaning. Love and wisdom are interconnected; one cannot truly exist without the other.
        """
    
    def test_hierarchical_processing(self):
        """Test the three-level hierarchical processing."""
        config = {
            'max_concepts': 5,
            'max_summary_tokens': 50,
            'max_insights': 3
        }
        
        result = self.engine.process_text(self.test_text, config)
        
        # Check hierarchical structure
        self.assertIn('hierarchical_summary', result)
        hierarchical = result['hierarchical_summary']
        
        self.assertIn('level_1_concepts', hierarchical)
        self.assertIn('level_2_core', hierarchical)
        self.assertIn('level_3_expanded', hierarchical)
        
        # Check types
        self.assertIsInstance(hierarchical['level_1_concepts'], list)
        self.assertIsInstance(hierarchical['level_2_core'], str)
        
        # Check concepts are extracted
        self.assertTrue(len(hierarchical['level_1_concepts']) > 0)
        self.assertTrue(len(hierarchical['level_2_core']) > 0)
    
    def test_insight_extraction(self):
        """Test insight extraction functionality."""
        config = {'max_insights': 5, 'min_insight_score': 0.3}
        result = self.engine.process_text(self.test_text, config)
        
        self.assertIn('key_insights', result)
        insights = result['key_insights']
        
        self.assertIsInstance(insights, list)
        
        # Check insight structure if any insights found
        if insights:
            insight = insights[0]
            self.assertIn('text', insight)
            self.assertIn('score', insight)
            self.assertIn('type', insight)
    
    def test_metadata_generation(self):
        """Test metadata generation."""
        result = self.engine.process_text(self.test_text)
        
        self.assertIn('metadata', result)
        metadata = result['metadata']
        
        expected_metadata_keys = ['processing_time', 'compression_ratio', 'concept_density']
        for key in expected_metadata_keys:
            self.assertIn(key, metadata)
    
    def test_backward_compatibility(self):
        """Test backward compatibility with legacy interfaces."""
        result = self.engine.process_text(self.test_text)
        
        # Should have backward compatibility keys
        self.assertIn('summary', result)
        self.assertIn('tags', result)
        self.assertIn('sum', result)
    
    def test_empty_text_handling(self):
        """Test handling of empty text."""
        result = self.engine.process_text("")
        self.assertIn('error', result)
        
        result = self.engine.process_text(None)
        self.assertIn('error', result)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)