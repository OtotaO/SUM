
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import os
import logging

logger = logging.getLogger(__name__)

class SimpleSUM:
    """A text summarization class that uses NLTK for processing."""
    
    def __init__(self):
        """Initialize NLTK resources and stopwords."""
        self._initialize_nltk()
        self.stop_words = set(stopwords.words('english'))
        
    def _initialize_nltk(self):
        """Download required NLTK resources."""
        try:
            nltk_data_dir = os.path.expanduser('~/nltk_data')
            os.makedirs(nltk_data_dir, exist_ok=True)
            
            # Download required NLTK resources
            resources = ['punkt', 'stopwords', 'wordnet']
            for resource in resources:
                try:
                    nltk.download(resource, quiet=True)
                except Exception as e:
                    logger.error(f"Error downloading {resource}: {str(e)}")
                    raise RuntimeError(f"Failed to download NLTK resource: {resource}")
        except Exception as e:
            logger.error(f"Error initializing NLTK: {str(e)}")
            raise RuntimeError("Failed to initialize NLTK resources")

    def process_text(self, text, model_config=None):
        """
        Process and summarize input text.
        
        Args:
            text (str): Input text to summarize
            model_config (dict, optional): Configuration parameters
            
        Returns:
            dict: Summary results or error message
        """
        if not isinstance(text, str) or not text.strip():
            return {'error': 'Invalid or empty text provided'}

        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= 2:
                return {'summary': text}

            max_tokens = model_config.get('maxTokens', 100) if model_config else 100
            return self._generate_summary(sentences, max_tokens)
        except Exception as e:
            logger.exception("Error processing text")
            return {'error': str(e)}

    def _generate_summary(self, sentences, max_tokens):
        """Generate summary based on sentence scoring."""
        word_freq = self._calculate_word_frequencies(sentences)
        sentence_scores = self._score_sentences(sentences, word_freq)
        
        return self._build_summary(sentences, sentence_scores, max_tokens)

    def _calculate_word_frequencies(self, sentences):
        """Calculate word frequencies across all sentences."""
        word_freq = defaultdict(int)
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            for word in words:
                if word.isalnum() and word not in self.stop_words:
                    word_freq[word] += 1
        return word_freq

    def _score_sentences(self, sentences, word_freq):
        """Score sentences based on word frequencies."""
        sentence_scores = defaultdict(int)
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            for word in words:
                if word in word_freq:
                    sentence_scores[sentence] += word_freq[word]
        return sentence_scores

    def _build_summary(self, sentences, sentence_scores, max_tokens):
        """Build summary within token limit."""
        sorted_sentences = sorted(sentences, key=lambda s: sentence_scores[s], reverse=True)
        summary_sentences = []
        current_tokens = 0
        
        for sentence in sorted_sentences:
            tokens = len(word_tokenize(sentence))
            if current_tokens + tokens <= max_tokens:
                summary_sentences.append(sentence)
                current_tokens += tokens
            else:
                break
                
        if not summary_sentences:
            first_sent_words = word_tokenize(sorted_sentences[0])[:max_tokens-1]
            return {'summary': ' '.join(first_sent_words) + '...'}
            
        summary_sentences.sort(key=sentences.index)
        return {'summary': ' '.join(summary_sentences)}
