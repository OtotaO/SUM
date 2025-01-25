import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import os

class SimpleSUM:
    def __init__(self):
        try:
            nltk_data_dir = os.path.expanduser('~/nltk_data')
            os.makedirs(nltk_data_dir, exist_ok=True)
            for resource in ['punkt', 'stopwords']:
                nltk.download(resource, quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Error initializing NLTK: {str(e)}")
            raise

    def process_text(self, text, model_config=None):
        if not text.strip():
            return {'error': 'Empty text provided'}

        try:
            # Basic text cleaning
            text = ' '.join(text.split())  # Remove extra whitespace
            sentences = sent_tokenize(text)

            if len(sentences) <= 2:
                return {'summary': text}

            # Get number of sentences to include
            max_sentences = int(model_config.get('maxSentences', 3)) if model_config else 3

            # Calculate word frequencies
            word_freq = defaultdict(int)
            for word in word_tokenize(text.lower()):
                if word.isalnum() and word not in self.stop_words:
                    word_freq[word] += 1

            # Score sentences
            sentence_scores = []
            for i, sentence in enumerate(sentences):
                score = 0
                words = word_tokenize(sentence.lower())

                # Score based on word frequency
                for word in words:
                    if word in word_freq:
                        score += word_freq[word]

                # Position bias - favor earlier sentences
                position_weight = 1.0 - (i / len(sentences))
                score *= (1 + position_weight)

                sentence_scores.append((score, sentence))

            # Select top sentences and maintain original order
            selected_pairs = sorted(sentence_scores, reverse=True)[:max_sentences]
            summary_sentences = []
            for _, sentence in sorted(selected_pairs, key=lambda x: sentences.index(x[1])):
                summary_sentences.append(sentence)

            return {'summary': ' '.join(summary_sentences)}

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