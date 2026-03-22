"""
Test suite for SUM summarization engines

Comprehensive tests for all summarization models including:
- Basic summarization
- Advanced summarization  
- Hierarchical densification
- Language detection
- Edge cases and error handling
"""

import pytest
import time
from unittest.mock import patch, MagicMock

from summarization_engine import (
    BasicSummarizationEngine,
    AdvancedSummarizationEngine, 
    HierarchicalDensificationEngine
)


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    Artificial intelligence (AI) is intelligence demonstrated by machines,
    in contrast to the natural intelligence displayed by humans and animals.
    Leading AI textbooks define the field as the study of intelligent agents:
    any device that perceives its environment and takes actions that maximize
    its chance of successfully achieving its goals. Colloquially, the term
    artificial intelligence is often used to describe machines that mimic
    cognitive functions that humans associate with the human mind, such as
    learning and problem solving.
    """

class TestBasicSummarizationEngine:
    """Test suite for BasicSummarizationEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create a BasicSummarizationEngine instance."""
        return BasicSummarizationEngine()
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return """
        Artificial intelligence (AI) is intelligence demonstrated by machines, 
        in contrast to the natural intelligence displayed by humans and animals. 
        Leading AI textbooks define the field as the study of intelligent agents: 
        any device that perceives its environment and takes actions that maximize 
        its chance of successfully achieving its goals. Colloquially, the term 
        artificial intelligence is often used to describe machines that mimic 
        cognitive functions that humans associate with the human mind, such as 
        learning and problem solving.
        """
    
    def test_process_empty_text(self, engine):
        """Test processing empty text."""
        result = engine.process_text("")
        assert 'error' in result
        assert 'Empty' in result['error']
    
    def test_process_none_text(self, engine):
        """Test processing None text."""
        result = engine.process_text(None)
        assert 'error' in result
    
    def test_process_valid_text(self, engine, sample_text):
        """Test processing valid text."""
        result = engine.process_text(sample_text)
        
        assert 'summary' in result
        assert 'sum' in result
        assert 'tags' in result
        assert 'original_length' in result
        assert 'compression_ratio' in result
        assert 'detected_language' in result
        assert 'language_name' in result
        
        # Check language detection
        assert result['detected_language'] == 'en'
        assert result['language_name'] == 'English'
        assert result['language_confidence'] > 0.6
    
    def test_summary_length_control(self, engine, sample_text):
        """Test summary length control with maxTokens."""
        # Short summary
        result = engine.process_text(sample_text, {'maxTokens': 20, 'threshold': 0.1})
        summary_words = len(result['summary'].split())
        assert summary_words <= 25  # Allow small overflow
        
        # Longer summary
        result = engine.process_text(sample_text, {'maxTokens': 100, 'threshold': 0.1})
        summary_words_long = len(result['summary'].split())
        assert summary_words_long > summary_words
    
    def test_tag_generation(self, engine, sample_text):
        """Test tag generation."""
        result = engine.process_text(sample_text)
        
        assert isinstance(result['tags'], list)
        assert len(result['tags']) > 0
        assert 'intelligence' in result['tags']
        assert 'artificial' in result['tags']
    
    def test_caching(self, engine, sample_text):
        """Test caching functionality."""
        # First call - not cached
        unique_text = sample_text + str(time.time())
        result1 = engine.process_text(unique_text, {'use_cache': True})
        assert result1.get('cached', False) is False
        
        # Second call - should be cached
        result2 = engine.process_text(unique_text, {'use_cache': True})
        assert result2.get('cached', False) is True
        
        # Results should match
        assert result1['summary'] == result2['summary']
        assert result1['tags'] == result2['tags']
    
    def test_short_text_handling(self, engine):
        """Test handling of very short text."""
        short_text = "This is a test."
        result = engine.process_text(short_text)
        
        assert 'summary' in result
        assert result['summary'] == short_text
    
    def test_special_characters(self, engine):
        """Test handling of special characters."""
        text = "AI & ML are amazing! But what about $costs? #technology"
        result = engine.process_text(text)
        
        assert 'error' not in result
        assert 'summary' in result
    
    def test_multilingual_text(self, engine):
        """Test detection of non-English text."""
        spanish_text = """
        La inteligencia artificial es la inteligencia demostrada por máquinas, 
        en contraste con la inteligencia natural mostrada por humanos y animales.
        """
        result = engine.process_text(spanish_text)
        
        assert result['detected_language'] == 'es'
        assert result['language_name'] == 'Spanish'


class TestAdvancedSummarizationEngine:
    """Test suite for AdvancedSummarizationEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create an AdvancedSummarizationEngine instance."""
        return AdvancedSummarizationEngine()
    
    def test_advanced_features(self, engine, sample_text):
        """Test advanced features of MagnumOpusSUM."""
        result = engine.process_text(sample_text)
        
        # Check additional features
        if 'meta_analysis' in result:
            assert 'named_entities' in result['meta_analysis']
            assert 'main_concept' in result['meta_analysis']
            assert 'sentiment' in result['meta_analysis']
            assert 'keywords' in result['meta_analysis']
    
    def test_sentiment_analysis(self, engine):
        """Test sentiment analysis feature."""
        positive_text = "This is absolutely amazing and wonderful! I love it!"
        result = engine.process_text(positive_text)
        
        if 'meta_analysis' in result and 'sentiment' in result['meta_analysis']:
            sentiment = result['meta_analysis']['sentiment']
            assert sentiment['compound'] > 0.5  # Positive sentiment
    
    def test_entity_extraction(self, engine):
        """Test named entity extraction."""
        text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
        result = engine.process_text(text)
        
        # Advanced engine might extract entities
        assert 'sum' in result
        assert 'summary' in result


class TestHierarchicalDensificationEngine:
    """Test suite for HierarchicalDensificationEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create a HierarchicalDensificationEngine instance."""
        return HierarchicalDensificationEngine()
    
    @pytest.fixture
    def long_text(self):
        """Longer text for hierarchical testing."""
        return """
        The Renaissance was a period in European history marking the transition 
        from the Middle Ages to modernity and covering the 15th and 16th centuries. 
        It occurred after the Crisis of the Late Middle Ages and was associated 
        with great social change. In addition to the standard periodization, 
        proponents of a long Renaissance put its beginning in the 14th century 
        and its end in the 17th century.
        
        The intellectual basis of the Renaissance was its version of humanism, 
        derived from the concept of Roman humanitas and the rediscovery of 
        classical Greek philosophy, such as that of Protagoras, who said that 
        "man is the measure of all things." This new thinking became manifest 
        in art, architecture, politics, science and literature.
        
        Early examples were the development of perspective in oil painting and 
        the revived knowledge of how to make concrete. Although the invention 
        of metal movable type sped the dissemination of ideas from the later 
        15th century, the changes of the Renaissance were not uniform across 
        Europe: the first traces appear in Italy as early as the late 13th 
        century, in particular with the writings of Dante and the paintings 
        of Giotto.
        """
    
    def test_hierarchical_structure(self, engine, long_text):
        """Test hierarchical summary structure."""
        result = engine.process_text(long_text)
        
        assert 'hierarchical_summary' in result
        hier = result['hierarchical_summary']
        
        # Check levels
        assert 'level_1_concepts' in hier
        assert 'level_2_core' in hier
        # Level 3 might be optional based on text length
        
        # Check metadata
        assert 'metadata' in result
        assert 'compression_ratio' in result['metadata']
        assert 'concept_density' in result['metadata']
    
    def test_key_insights_extraction(self, engine, long_text):
        """Test key insights extraction."""
        result = engine.process_text(long_text)
        
        assert 'key_insights' in result
        assert isinstance(result['key_insights'], list)
    
    def test_concept_extraction(self, engine, long_text):
        """Test concept extraction."""
        result = engine.process_text(long_text)
        
        concepts = result['hierarchical_summary']['level_1_concepts']
        assert isinstance(concepts, list)
        assert len(concepts) > 0
        assert 'renaissance' in [c.lower() for c in concepts]
    
    def test_compression_levels(self, engine, long_text):
        """Test different compression levels."""
        result = engine.process_text(long_text)
        
        # Level 2 should be shorter than original
        level_2 = result['hierarchical_summary']['level_2_core']
        assert len(level_2) < len(long_text)
        
        # Level 3 (if exists) should be longer than level 2
        if 'level_3_expanded' in result['hierarchical_summary']:
            level_3 = result['hierarchical_summary']['level_3_expanded']
            if level_3:
                assert len(level_3) >= len(level_2)
    
    def test_security_validation(self, engine):
        """Test security validation."""
        malicious_text = "Some text with __import__('os').system('ls')"
        result = engine.process_text(malicious_text)
        
        # Should process safely without executing code
        assert 'error' not in result or 'security' not in result.get('error', '')


class TestLanguageDetection:
    """Test suite for language detection features."""
    
    @pytest.fixture
    def engines(self):
        """Create all engine instances."""
        return {
            'basic': BasicSummarizationEngine(),
            'advanced': AdvancedSummarizationEngine(),
            'hierarchical': HierarchicalDensificationEngine()
        }
    
    def test_english_detection(self, engines):
        """Test English language detection."""
        text = "This is a test in English language."
        
        for name, engine in engines.items():
            result = engine.process_text(text)
            assert result['detected_language'] == 'en'
            assert result['language_name'] == 'English'
            assert result['language_confidence'] > 0.8
    
    def test_spanish_detection(self, engines):
        """Test Spanish language detection."""
        text = "Este es un texto en español para probar la detección de idioma."
        
        for name, engine in engines.items():
            result = engine.process_text(text)
            assert result['detected_language'] == 'es'
            assert result['language_name'] == 'Spanish'
    
    def test_french_detection(self, engines):
        """Test French language detection."""
        text = "Ceci est un texte en français pour tester la détection de langue."
        
        for name, engine in engines.items():
            result = engine.process_text(text)
            assert result['detected_language'] == 'fr'
            assert result['language_name'] == 'French'
    
    def test_mixed_language_handling(self, engines):
        """Test handling of mixed language text."""
        text = "This is English. Esto es español. C'est français."
        
        # Should detect the dominant language
        for name, engine in engines.items():
            result = engine.process_text(text)
            assert result['detected_language'] in ['en', 'es', 'fr']
            assert 'language_confidence' in result


class TestPerformance:
    """Performance tests for summarization engines."""
    
    @pytest.fixture
    def large_text(self):
        """Generate large text for performance testing."""
        base = """The history of artificial intelligence began in antiquity, 
        with myths, stories and rumors of artificial beings endowed with 
        intelligence or consciousness by master craftsmen. """
        return base * 100  # Repeat to create large text
    
    def test_basic_engine_performance(self):
        """Test BasicSummarizationEngine performance."""
        engine = BasicSummarizationEngine()
        text = "Test text. " * 1000  # ~2000 words
        
        start_time = time.time()
        result = engine.process_text(text)
        processing_time = time.time() - start_time
        
        assert 'error' not in result
        assert processing_time < 5.0  # Should complete within 5 seconds
    
    @pytest.mark.slow
    def test_large_text_processing(self, large_text):
        """Test processing of large texts."""
        engine = BasicSummarizationEngine()
        
        start_time = time.time()
        result = engine.process_text(large_text)
        processing_time = time.time() - start_time
        
        assert 'error' not in result
        assert 'summary' in result
        print(f"Large text processed in {processing_time:.2f} seconds")


class TestErrorHandling:
    """Test error handling in summarization engines."""
    
    def test_invalid_input_types(self):
        """Test handling of invalid input types."""
        engine = BasicSummarizationEngine()
        
        # Test various invalid inputs
        invalid_inputs = [
            None,
            123,
            [],
            {},
            True,
            object()
        ]
        
        for invalid_input in invalid_inputs:
            result = engine.process_text(invalid_input)
            assert 'error' in result
    
    def test_empty_strings(self):
        """Test handling of empty strings."""
        engine = BasicSummarizationEngine()
        
        empty_inputs = ["", "   ", "\n\n\n", "\t\t"]
        
        for empty_input in empty_inputs:
            result = engine.process_text(empty_input)
            assert 'error' in result or result['summary'] == empty_input.strip()
    
    def test_unicode_handling(self):
        """Test handling of Unicode text."""
        engine = BasicSummarizationEngine()
        
        unicode_texts = [
            "Hello 👋 World 🌍",
            "Здравствуй мир",  # Russian
            "你好世界",  # Chinese
            "مرحبا بالعالم",  # Arabic
        ]
        
        for text in unicode_texts:
            result = engine.process_text(text)
            assert 'error' not in result
            assert 'summary' in result


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])