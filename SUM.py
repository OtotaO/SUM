import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict

import os

class SimpleSUM:
    def __init__(self):
        try:
            # Create nltk_data directory
            nltk_data_dir = os.path.expanduser('~/nltk_data')
            os.makedirs(nltk_data_dir, exist_ok=True)
            
            # Download required NLTK resources
            for resource in ['punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger', 'wordnet']:
                try:
                    nltk.download(resource, quiet=True, raise_on_error=False)
                except Exception as e:
                    print(f"Error downloading {resource}: {str(e)}")
            
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Error initializing NLTK: {str(e)}")
            raise RuntimeError("Failed to initialize NLTK resources")

    def process_text(self, text, model_config=None):
        if not text.strip():
            return {'error': 'Empty text provided'}
            
        if model_config and model_config.get('maxTokens'):
            max_length = model_config['maxTokens']
        else:
            max_length = 100

        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= 2:
                return {'summary': text}

            words = word_tokenize(text.lower())
            word_freq = defaultdict(int)

            for word in words:
                if word.isalnum() and word not in self.stop_words:
                    word_freq[word] += 1

            sentence_scores = defaultdict(int)
            for sentence in sentences:
                for word in word_tokenize(sentence.lower()):
                    if word in word_freq:
                        sentence_scores[sentence] += word_freq[word]

            # Make summary length proportional to input length, but always shorter
            summary_length = max(1, min(len(sentences) // 3, len(sentences) - 1))
            
            # Get highest scoring sentences
            summary_sentences = sorted(sentences, key=lambda s: sentence_scores[s], reverse=True)[:summary_length]
            # Restore original sentence order for better readability
            summary_sentences.sort(key=sentences.index)
            summary = ' '.join(summary_sentences)

            return {'summary': summary}
        except Exception as e:
            return {'error': str(e)}

def main():
    summarizer = SimpleSUM()
    text = "This is a sample text. This text will be summarized. This is another sentence."
    summary = summarizer.process_text(text)
    print(summary)

if __name__ == "__main__":
    main()