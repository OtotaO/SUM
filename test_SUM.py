import unittest
from summarization_engine import SimpleSUM  # Import the SimpleSUM class from your SUM.py file

class TestSimpleSUM(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.summarizer = SimpleSUM()

    def test_empty_text(self):
        """Test that the summarizer returns an error for empty text."""
        result = self.summarizer.process_text("")
        self.assertEqual(result, {'error': 'Empty text provided'})

    def test_short_text(self):
        """Test that the summarizer returns the original text for short texts (<= 2 sentences)."""
        text = "This is a short text."
        result = self.summarizer.process_text(text)
        self.assertEqual(result, {'summary': text})

    def test_summary_within_token_limit(self):
        """Test that the summarizer returns a summary within the token limit."""
        text = "This is a sample text. This text will be summarized. This is another sentence."
        config = {'maxTokens': 20}
        result = self.summarizer.process_text(text, config)
        self.assertLessEqual(len(result['summary'].split()), config['maxTokens'])

    def test_no_complete_sentence_fits(self):
        """Test that the summarizer returns a truncated first sentence when no complete sentence fits within the token limit."""
        text = "This is a very long first sentence that exceeds the token limit. This is another sentence."
        config = {'maxTokens': 10}
        result = self.summarizer.process_text(text, config)
        self.assertTrue(result['summary'].endswith('...'))
        self.assertLessEqual(len(result['summary'].split()), config['maxTokens'])

    def test_error_handling(self):
        """Test that the summarizer handles errors gracefully."""
        # Mocking an error during tokenization (example)
        text = "This is a text."
        # Monkey-patching the sent_tokenize function to raise an exception
        import nltk
        original_sent_tokenize = nltk.tokenize.sent_tokenize  # Store the original function
        def mock_sent_tokenize(text):
            raise Exception("Mocked error during tokenization")
        nltk.tokenize.sent_tokenize = mock_sent_tokenize

        result = self.summarizer.process_text(text)
        self.assertIn('error', result)

        nltk.tokenize.sent_tokenize = original_sent_tokenize  # Restore the original function

if __name__ == '__main__':
    unittest.main()
