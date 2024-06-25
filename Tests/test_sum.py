import unittest
from SUM import SUM
import json
import os
import tempfile

class TestSUM(unittest.TestCase):
    def setUp(self):
        self.summarizer = SUM()
        self.test_text = "This is a test sentence. It contains multiple sentences. We will use it to test various functionalities."
        
        # Create a temporary JSON file for testing load_data
        self.temp_json = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json')
        json.dump({"text1": self.test_text}, self.temp_json)
        self.temp_json.close()

    def tearDown(self):
        os.unlink(self.temp_json.name)

    def test_load_data(self):
        data = self.summarizer.load_data(self.temp_json.name)
        self.assertEqual(data, {"text1": self.test_text})

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

    def test_identify_main_direction(self):
        main_direction = self.summarizer.identify_main_direction(self.test_text)
        self.assertIsInstance(main_direction, str)

    def test_calculate_similarity(self):
        similarity = self.summarizer.calculate_similarity(self.test_text, self.test_text)
        self.assertIsInstance(similarity, float)
        self.assertAlmostEqual(similarity, 1.0, places=5)

    def test_extract_keywords(self):
        keywords = self.summarizer.extract_keywords(self.test_text)
        self.assertIsInstance(keywords, list)
        self.assertEqual(len(keywords), 5)  # default top_n is 5

    def test_sentiment_analysis(self):
        sentiment = self.summarizer.sentiment_analysis(self.test_text)
        self.assertIn(sentiment, ['Positive', 'Negative', 'Neutral'])

    def test_generate_text_summary(self):
        summary = self.summarizer.generate_text_summary(self.test_text)
        self.assertIsInstance(summary, str)
        self.assertLess(len(summary), len(self.test_text))

    def test_export_summary(self):
        summary = "This is a test summary."
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as temp_file:
            self.summarizer.export_summary(summary, temp_file.name)
            temp_file.seek(0)
            content = temp_file.read()
        self.assertEqual(content, summary)
        os.unlink(temp_file.name)

if __name__ == '__main__':
    unittest.main()
