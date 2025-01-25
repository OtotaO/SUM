import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import os

class SimpleSUM:
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english'))

    def process_text(self, text, model_config=None):
        if not text.strip():
            return {'error': 'Empty text provided'}

        try:
            # Tokenize into sentences
            sentences = sent_tokenize(text)

            if len(sentences) <= 1:
                return {'summary': text}

            # Get sentence count from config
            max_sentences = int(model_config.get('maxSentences', 3)) if model_config else 3

            # Create word frequency distribution
            words = word_tokenize(text.lower())
            word_freq = Counter(word for word in words if word.isalnum() and word not in self.stop_words)

            # Score sentences based on word frequency
            sentence_scores = []
            for i, sentence in enumerate(sentences):
                score = 0
                words = word_tokenize(sentence.lower())
                for word in words:
                    if word in word_freq:
                        score += word_freq[word]
                # Normalize by sentence length
                score = score / (len(words) + 1)
                # Boost score of early sentences
                if i < 2:
                    score *= 1.5
                sentence_scores.append((score, sentence))

            # Select top N sentences while preserving order
            top_sentences = sorted(sentence_scores, key=lambda x: x[0], reverse=True)[:max_sentences]
            summary_sentences = sorted(top_sentences, key=lambda x: sentences.index(x[1]))

            summary = ' '.join(sentence[1] for sentence in summary_sentences)
            return {'summary': summary}

        except Exception as e:
            return {'error': str(e)}

def main():
    summarizer = SimpleSUM()
    text = "This is a sample text. This text will be summarized. This is another sentence."
    config = {'maxSentences': 2}
    result = summarizer.process_text(text, config)
    print(result['summary'])

if __name__ == "__main__":
    main()