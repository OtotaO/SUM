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

# Check if common entities are found
entity_texts = [entity[0].lower() for entity in entities]
self.assertTrue(any('google' in text for text in entity_texts))
self.assertTrue(any('microsoft' in text for text in entity_texts))

    def test_identify_main_concept(self):
"""Test main concept identification."""
main_concept = self.summarizer.identify_main_concept(self.test_text)
self.assertIsInstance(main_concept, str)
self.assertTrue('machine learning' in main_concept.lower() or 
           'ai' in main_concept.lower())

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

# Check if important terms are captured
all_keywords = ' '.join(keywords).lower()
self.assertTrue('machine learning' in all_keywords or 
           'deep learning' in all_keywords or 
           'ai' in all_keywords)

    def test_generate_word_cloud(self):
"""Test word cloud generation."""
wordcloud = self.summarizer.generate_word_cloud(self.test_text)
self.assertIsInstance(wordcloud, WordCloud)
self.assertTrue(len(wordcloud.words_) > 0)

    def test_detect_language(self):
"""Test language detection."""
lang = self.summarizer.detect_language(self.test_text)
self.assertEqual(lang, 'en')

# Test with non-English text
spanish_text = "Hola mundo. Esto es una prueba."
self.assertEqual(self.summarizer.detect_language(spanish_text), 'es')

    def test_translate_text(self):
"""Test text translation."""
text = "Hello world"
translated = self.summarizer.translate_text(text, target_lang='es')
self.assertNotEqual(translated, text)
self.assertTrue(translated.lower().startswith('hola'))

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

# Test with different sentence counts
summary_1 = self.summarizer.generate_sentence_summary(self.test_text, num_sentences=1)
summary_2 = self.summarizer.generate_sentence_summary(self.test_text, num_sentences=2)
self.assertTrue(len(summary_1) < len(summary_2))

    def test_adjust_parameters(self):
"""Test parameter adjustment."""
initial_num_tags = self.summarizer.num_tags

# Test increasing tags
self.summarizer.adjust_parameters(8)
self.assertEqual(self.summarizer.num_tags, 8)

# Test exceeding maximum
self.summarizer.adjust_parameters(15)
self.assertEqual(self.summarizer.num_tags, 10)  # max is 10

# Test below minimum
self.summarizer.adjust_parameters(0)
self.assertEqual(self.summarizer.num_tags, 1)  # min is 1

    def test_edge_cases(self):
"""Test edge cases and error handling."""
# Test empty text
result = self.summarizer.process_text("")
self.assertIn('error', result)

# Test very short text
short_text = "Hello world."
result = self.summarizer.process_text(short_text)
self.assertEqual(result['sum'], short_text)
self.assertEqual(result['summary'], short_text)

# Test with None
result = self.summarizer.process_text(None)
self.assertIn('error', result)

# Test with non-string input
result = self.summarizer.process_text(123)
self.assertIn('error', result)


if __name__ == '__main__':
    unittest.main()
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
        
        # Check if common entities are found
        entity_texts = [entity[0].lower() for entity in entities]
        self.assertTrue(any('google' in text for text in entity_texts))
        self.assertTrue(any('microsoft' in text for text in entity_texts))

    def test_identify_main_concept(self):
        """Test main concept identification."""
        main_concept = self.summarizer.identify_main_concept(self.test_text)
        self.assertIsInstance(main_concept, str)
        self.assertTrue('machine learning' in main_concept.lower() or 
                       'ai' in main_concept.lower())

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
        
        # Check if important terms are captured
        all_keywords = ' '.join(keywords).lower()
        self.assertTrue('machine learning' in all_keywords or 
                       'deep learning' in all_keywords or 
                       'ai' in all_keywords)

    def test_generate_word_cloud(self):
        """Test word cloud generation."""
        wordcloud = self.summarizer.generate_word_cloud(self.test_text)
        self.assertIsInstance(wordcloud, WordCloud)
        self.assertTrue(len(wordcloud.words_) > 0)

    def test_detect_language(self):
        """Test language detection."""
        lang = self.summarizer.detect_language(self.test_text)
        self.assertEqual(lang, 'en')
        
        # Test with non-English text
        spanish_text = "Hola mundo. Esto es una prueba."
        self.assertEqual(self.summarizer.detect_language(spanish_text), 'es')

    def test_translate_text(self):
        """Test text translation."""
        text = "Hello world"
        translated = self.summarizer.translate_text(text, target_lang='es')
        self.assertNotEqual(translated, text)
        self.assertTrue(translated.lower().startswith('hola'))

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
        
        # Test with different sentence counts
        summary_1 = self.summarizer.generate_sentence_summary(self.test_text, num_sentences=1)
        summary_2 = self.summarizer.generate_sentence_summary(self.test_text, num_sentences=2)
        self.assertTrue(len(summary_1) < len(summary_2))

    def test_adjust_parameters(self):
        """Test parameter adjustment."""
        initial_num_tags = self.summarizer.num_tags
        
        # Test increasing tags
        self.summarizer.adjust_parameters(8)
        self.assertEqual(self.summarizer.num_tags, 8)
        
        # Test exceeding maximum
        self.summarizer.adjust_parameters(15)
        self.assertEqual(self.summarizer.num_tags, 10)  # max is 10
        
        # Test below minimum
        self.summarizer.adjust_parameters(0)
        self.assertEqual(self.summarizer.num_tags, 1)  # min is 1

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test empty text
        result = self.summarizer.process_text("")
        self.assertIn('error', result)
        
        # Test very short text
        short_text = "Hello world."
        result = self.summarizer.process_text(short_text)
        self.assertEqual(result['sum'], short_text)
        self.assertEqual(result['summary'], short_text)
        
        # Test with None
        result = self.summarizer.process_text(None)
        self.assertIn('error', result)
        
        # Test with non-string input
        result = self.summarizer.process_text(123)
        self.assertIn('error', result)


if __name__ == '__main__':
    unittest.main()
