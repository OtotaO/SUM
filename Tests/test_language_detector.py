"""
Test suite for language detection functionality

Tests language detection accuracy across multiple languages
and detection methods.
"""

import pytest
from language_detector import (
    LanguageDetector,
    detect_language,
    multilingual_summarizer
)


class TestLanguageDetector:
    """Test suite for LanguageDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create a LanguageDetector instance."""
        return LanguageDetector()
    
    def test_english_detection(self, detector):
        """Test English language detection."""
        texts = [
            "This is a simple English sentence.",
            "Artificial intelligence is revolutionizing technology.",
            "The quick brown fox jumps over the lazy dog."
        ]
        
        for text in texts:
            result = detector.detect_language(text)
            assert result['language'] == 'en'
            assert result['language_name'] == 'English'
            assert result['confidence'] > 0.8
    
    def test_spanish_detection(self, detector):
        """Test Spanish language detection."""
        texts = [
            "Esta es una oración simple en español.",
            "La inteligencia artificial está revolucionando la tecnología.",
            "El rápido zorro marrón salta sobre el perro perezoso."
        ]
        
        for text in texts:
            result = detector.detect_language(text)
            assert result['language'] == 'es'
            assert result['language_name'] == 'Spanish'
            assert result['confidence'] > 0.7
    
    def test_french_detection(self, detector):
        """Test French language detection."""
        texts = [
            "Ceci est une phrase simple en français.",
            "L'intelligence artificielle révolutionne la technologie.",
            "Le renard brun rapide saute par-dessus le chien paresseux."
        ]
        
        for text in texts:
            result = detector.detect_language(text)
            assert result['language'] == 'fr'
            assert result['language_name'] == 'French'
            assert result['confidence'] > 0.7
    
    def test_german_detection(self, detector):
        """Test German language detection."""
        texts = [
            "Dies ist ein einfacher deutscher Satz.",
            "Künstliche Intelligenz revolutioniert die Technologie.",
            "Der schnelle braune Fuchs springt über den faulen Hund."
        ]
        
        for text in texts:
            result = detector.detect_language(text)
            assert result['language'] == 'de'
            assert result['language_name'] == 'German'
    
    def test_italian_detection(self, detector):
        """Test Italian language detection."""
        text = "Questa è una semplice frase in italiano."
        result = detector.detect_language(text)
        assert result['language'] == 'it'
        assert result['language_name'] == 'Italian'
    
    def test_portuguese_detection(self, detector):
        """Test Portuguese language detection."""
        text = "Esta é uma frase simples em português."
        result = detector.detect_language(text)
        assert result['language'] == 'pt'
        assert result['language_name'] == 'Portuguese'
    
    def test_russian_detection(self, detector):
        """Test Russian language detection with Cyrillic script."""
        text = "Это простое предложение на русском языке."
        result = detector.detect_language(text)
        assert result['language'] == 'ru'
        assert result['language_name'] == 'Russian'
    
    def test_chinese_detection(self, detector):
        """Test Chinese language detection."""
        text = "这是一个简单的中文句子。"
        result = detector.detect_language(text)
        assert result['language'] == 'zh'
        assert result['language_name'] == 'Chinese'
    
    def test_japanese_detection(self, detector):
        """Test Japanese language detection."""
        texts = [
            "これは簡単な日本語の文です。",
            "人工知能は技術を革命化しています。"
        ]
        
        for text in texts:
            result = detector.detect_language(text)
            assert result['language'] == 'ja'
            assert result['language_name'] == 'Japanese'
    
    def test_arabic_detection(self, detector):
        """Test Arabic language detection."""
        text = "هذه جملة بسيطة باللغة العربية."
        result = detector.detect_language(text)
        assert result['language'] == 'ar'
        assert result['language_name'] == 'Arabic'
    
    def test_short_text_handling(self, detector):
        """Test handling of very short text."""
        result = detector.detect_language("Hello")
        assert result['language'] == 'en'
        assert result['confidence'] <= 0.5  # Low confidence expected
    
    def test_empty_text_handling(self, detector):
        """Test handling of empty text."""
        result = detector.detect_language("")
        assert result['language'] == 'en'  # Default
        assert result['method'] == 'default'
    
    def test_mixed_language_text(self, detector):
        """Test detection with mixed languages."""
        text = "Hello world! Bonjour le monde! Hola mundo!"
        result = detector.detect_language(text)
        
        # Should detect one of the languages
        assert result['language'] in ['en', 'fr', 'es']
        assert 'all_results' in result
    
    def test_detection_methods(self, detector):
        """Test different detection methods."""
        text = "This is a test to check which detection method is used."
        result = detector.detect_language(text)
        
        # Should have a detection method
        assert 'method' in result
        assert result['method'] in ['langdetect', 'langid', 'stopwords', 'script', 'fallback']
    
    def test_supported_languages(self, detector):
        """Test supported language checking."""
        # English should be supported
        result = detector.detect_language("This is English text.")
        assert result['supported'] is True
        
        # Some languages might not be fully supported
        result = detector.detect_language("这是中文文本。")  # Chinese
        assert 'supported' in result


class TestLanguageConfig:
    """Test language-specific configurations."""
    
    @pytest.fixture
    def detector(self):
        """Create a LanguageDetector instance."""
        return LanguageDetector()
    
    def test_english_config(self, detector):
        """Test English language configuration."""
        config = detector.get_language_config('en')
        
        assert config['sentence_tokenizer'] == 'punkt'
        assert config['word_tokenizer'] == 'word_tokenize'
        assert config['stopwords'] == 'english'
        assert config['stemmer'] == 'PorterStemmer'
        assert config['sentence_delimiter'] == '. '
    
    def test_spanish_config(self, detector):
        """Test Spanish language configuration."""
        config = detector.get_language_config('es')
        
        assert config['stopwords'] == 'spanish'
        assert config['stemmer'] == 'SnowballStemmer'
        assert config['stemmer_lang'] == 'spanish'
    
    def test_chinese_config(self, detector):
        """Test Chinese language configuration."""
        config = detector.get_language_config('zh')
        
        assert config['sentence_tokenizer'] == 'chinese'
        assert config['sentence_delimiter'] == '。'
        assert config['min_sentence_length'] == 3
    
    def test_unsupported_language_config(self, detector):
        """Test configuration for unsupported language."""
        config = detector.get_language_config('xyz')  # Unknown language
        
        # Should default to English config
        assert config['sentence_tokenizer'] == 'punkt'
        assert config['stopwords'] == 'english'


class TestMultilingualSummarizer:
    """Test multilingual summarization features."""
    
    def test_english_summarization(self):
        """Test English text summarization."""
        text = "Artificial intelligence is transforming industries worldwide."
        result = multilingual_summarizer.summarize(text)
        
        assert result['detected_language']['language'] == 'en'
        assert result['language'] == 'en'
        assert result['language_name'] == 'English'
        assert 'text' in result
        assert 'language_config' in result
    
    def test_spanish_summarization(self):
        """Test Spanish text summarization."""
        text = "La inteligencia artificial está transformando industrias en todo el mundo."
        result = multilingual_summarizer.summarize(text)
        
        assert result['language'] == 'es'
        assert result['language_name'] == 'Spanish'
    
    def test_unsupported_language_fallback(self):
        """Test fallback for unsupported languages."""
        # Use a language that might not be fully supported
        text = "ဤသည်မှာ မြန်မာဘာသာစကားဖြစ်သည်။"  # Burmese
        result = multilingual_summarizer.summarize(text)
        
        # Should have fallback handling
        if not result['detected_language']['supported']:
            assert 'fallback_language' in result
            assert result['fallback_language'] == 'en'
    
    def test_preprocessing(self):
        """Test language-specific preprocessing."""
        # Chinese text
        chinese_text = "这是中文。这是另一个句子。"
        result = multilingual_summarizer.summarize(chinese_text)
        
        # Check if preprocessing was applied
        assert result['preprocessed'] is True
        # Chinese sentences should have spaces added after periods
        if result['language'] == 'zh':
            assert '。 ' in result['text']


class TestGlobalFunctions:
    """Test global convenience functions."""
    
    def test_detect_language_function(self):
        """Test the global detect_language function."""
        result = detect_language("This is a test.")
        
        assert 'language' in result
        assert 'language_name' in result
        assert 'confidence' in result
        assert result['language'] == 'en'
    
    def test_summarize_multilingual_function(self):
        """Test the global summarize_multilingual function."""
        from language_detector import summarize_multilingual
        
        result = summarize_multilingual("Test text for summarization.")
        
        assert 'detected_language' in result
        assert 'language' in result
        assert 'text' in result


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    @pytest.fixture
    def detector(self):
        """Create a LanguageDetector instance."""
        return LanguageDetector()
    
    def test_numbers_only(self, detector):
        """Test text with only numbers."""
        result = detector.detect_language("123 456 789")
        assert result['language'] == 'en'  # Should default to English
    
    def test_special_characters(self, detector):
        """Test text with special characters."""
        result = detector.detect_language("!@#$%^&*()")
        assert result['language'] == 'en'  # Should default to English
    
    def test_mixed_scripts(self, detector):
        """Test text with mixed scripts."""
        text = "Hello 你好 مرحبا"
        result = detector.detect_language(text)
        
        # Should detect something
        assert 'language' in result
        assert result['confidence'] < 1.0  # Mixed text should have lower confidence
    
    def test_very_long_text(self, detector):
        """Test with very long text."""
        # Create a long English text
        long_text = "This is a test sentence. " * 1000
        
        result = detector.detect_language(long_text)
        assert result['language'] == 'en'
        assert result['confidence'] > 0.9
    
    def test_unicode_normalization(self, detector):
        """Test Unicode normalization handling."""
        # Text with different Unicode forms
        text = "Café"  # With combining accent
        result = detector.detect_language(text)
        
        # Should still detect properly
        assert result['language'] in ['en', 'fr', 'es']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])