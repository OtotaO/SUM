import unittest
from SUM import MagnumOpusSUM
import json
import os
import tempfile

class TestSUM(unittest.TestCase):
    def setUp(self):
        self.summarizer = MagnumOpusSUM()
        self.test_text = "This is a test sentence. It contains multiple sentences. We will use it to test various functionalities."
        
        # Create a temporary JSON file for testing load_data
        self.temp_json = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json')
        json.dump({"text1": self.test_text}, self.temp_json)
        self.temp_json.close()

    def tearDown(self):
        os.unlink(self.temp_json.name)

    def test_preprocess_text(self):
        preprocessed = self.summarizer.preprocess_text(self.test_text)
        self.assertNotIn("is", preprocessed)
        self.assertNotIn("a", preprocessed)
        self.assertIn("test", preprocessed)

    def test_generate_summaries(self):
        summaries = self.summarizer.generate_summaries([self.test_text])
        self.assertEqual(len(summaries), 1)
        self.assertIsInstance(summaries[0], str)

    def test_identify_entities(self):
        entities = self.summarizer.identify_entities(self.test_text)
        self.assertIsInstance(entities, list)
        self.assertTrue(all(isinstance(entity, tuple) for entity in entities))

    def test_identify_main_concept(self):
        main_concept = self.summarizer.identify_main_concept(self.test_text)
        self.assertIsInstance(main_concept, str)

    def test_sentiment_analysis(self):
        sentiment = self.summarizer.sentiment_analysis(self.test_text)
        self.assertIn(sentiment, ['Positive', 'Negative', 'Neutral'])

    def test_extract_keywords(self):
        keywords = self.summarizer.extract_keywords(self.test_text)
        self.assertIsInstance(keywords, list)
        self.assertEqual(len(keywords), 5)  # default top_n is 5

    def test_generate_word_cloud(self):
        wordcloud = self.summarizer.generate_word_cloud(self.test_text)
        self.assertIsNotNone(wordcloud)

    def test_detect_language(self):
        lang = self.summarizer.detect_language(self.test_text)
        self.assertEqual(lang, 'en')

    def test_translate_text(self):
        translated = self.summarizer.translate_text(self.test_text, target_lang='es')
        self.assertNotEqual(translated, self.test_text)

    def test_adjust_parameters(self):
        initial_num_tags = self.summarizer.num_tags
        for _ in range(5):
            self.summarizer.adjust_parameters(5)
        self.assertNotEqual(initial_num_tags, self.summarizer.num_tags)

if __name__ == '__main__':
    unittest.main()
import unittest
from SUM import MagnumOpusSUM

class TestMagnumOpusSUM(unittest.TestCase):
    def setUp(self):
        self.summarizer = MagnumOpusSUM()
        self.test_text = "Machine learning has seen rapid advancements in recent years. From image recognition to natural language processing, AI systems are becoming increasingly sophisticated. Deep learning models, in particular, have shown remarkable capabilities in handling complex tasks."

    def test_preprocess_text(self):
        processed = self.summarizer.preprocess_text(self.test_text)
        self.assertIsInstance(processed, str)
        self.assertTrue(len(processed) > 0)

    def test_generate_tag_summary(self):
        tags = self.summarizer.generate_tag_summary(self.test_text)
        self.assertIsInstance(tags, list)
        self.assertTrue(len(tags) <= self.summarizer.num_tags)

    def test_generate_sentence_summary(self):
        summary = self.summarizer.generate_sentence_summary(self.test_text)
        self.assertIsInstance(summary, str)
        self.assertTrue(len(summary) < len(self.test_text))
        
    def test_process_text_categories(self):
        result = self.summarizer.process_text(self.test_text)
        self.assertIn('tags', result)
        self.assertIn('sum', result)
        self.assertIn('summary', result)
        self.assertTrue(len(result['tags']) > 0)
        self.assertTrue(len(result['sum']) > 0)
        self.assertTrue(len(result['summary']) > 0)
        
    def test_empty_text(self):
        result = self.summarizer.process_text("")
        self.assertIn('error', result)
        
    def test_short_text(self):
        short_text = "This is a very short text."
        result = self.summarizer.process_text(short_text)
        self.assertEqual(result['sum'], short_text)
        
    def test_long_text_compression(self):
        long_text = " ".join([self.test_text] * 5)  # Repeat test text 5 times
        result = self.summarizer.process_text(long_text)
        self.assertTrue(len(result['summary']) < len(long_text))
        self.assertTrue(len(result['sum']) < len(result['summary']))

if __name__ == '__main__':
    unittest.main()
