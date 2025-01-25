"""
MVP Text Summarizer
------------------
A minimal yet extensible implementation of extractive text summarization.
Following the Unix philosophy: Do One Thing Well.

Author: Your Name
License: Your License
"""

from typing import Dict, List, Optional
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter


class MVPSummarizer:
    """
    A minimal extractive text summarizer that focuses on core functionality.
    Designed for easy extension and maintenance.
    """

    def __init__(self) -> None:
        """Initialize the summarizer with required NLTK resources."""
        # Download required resources only once
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)

        self.stop_words = set(stopwords.words('english'))

    def summarize(self, text: str, num_sentences: int = 3) -> Dict[str, str]:
        """
        Create a summary of the input text by extracting key sentences.

        Args:
            text: Input text to summarize
            num_sentences: Number of sentences to include in summary (default: 3)

        Returns:
            Dictionary containing the summary or error message
        """
        if not self._validate_input(text, num_sentences):
            return {'error': 'Invalid input'}

        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= 1:
                return {'summary': text}

            # Calculate sentence importance
            sentence_scores = self._score_sentences(sentences)

            # Extract top sentences while preserving order
            summary = self._extract_summary(sentences, sentence_scores,
                                            num_sentences)

            return {'summary': summary}

        except Exception as e:
            return {'error': f'Summarization failed: {str(e)}'}

    def _validate_input(self, text: str, num_sentences: int) -> bool:
        """Validate input parameters."""
        return bool(text and isinstance(text, str) and text.strip()
                    and isinstance(num_sentences, int) and num_sentences > 0)

    def _score_sentences(self, sentences: List[str]) -> List[float]:
        """
        Score sentences based on word frequency.
        Returns list of scores corresponding to input sentences.
        """
        # Create word frequency distribution
        words = word_tokenize(' '.join(sentences).lower())
        word_freq = Counter(word for word in words
                            if word.isalnum() and word not in self.stop_words)

        # Score each sentence
        scores = []
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            score = sum(word_freq[word] for word in words if word in word_freq)
            scores.append(score / len(words) if words else 0)

        return scores

    def _extract_summary(self, sentences: List[str], scores: List[float],
                         num_sentences: int) -> str:
        """
        Extract top scoring sentences while preserving original order.
        """
        # Get indices of top scoring sentences
        paired_scores = list(enumerate(scores))
        top_indices = sorted(sorted(paired_scores,
                                    key=lambda x: x[1],
                                    reverse=True)[:num_sentences],
                             key=lambda x: x[0])

        # Combine sentences in original order
        summary_sentences = [sentences[idx] for idx, _ in top_indices]
        return ' '.join(summary_sentences)


def main():
    """Example usage of the MVP summarizer."""
    sample_text = """
    Machine learning is a subset of artificial intelligence. It focuses on the use 
    of data and algorithms to mimic human learning. The process allows for the 
    automated learning of insights without explicit programming. Many companies 
    use machine learning today. It helps them improve their services and products.
    """

    summarizer = MVPSummarizer()
    result = summarizer.summarize(sample_text, num_sentences=2)
    print("Summary:", result['summary'])


if __name__ == "__main__":
    main()
