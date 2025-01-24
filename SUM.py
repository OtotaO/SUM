
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
            
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Error initializing NLTK: {str(e)}")
            raise RuntimeError("Failed to initialize NLTK resources")

    def process_text(self, text, model_config=None):
        if not text.strip():
            return {'error': 'Empty text provided'}

        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= 2:
                return {'summary': text}

            # Get max tokens from config
            max_tokens = model_config.get('maxTokens', 100) if model_config else 100

            # Approximate tokens (rough estimate: words + punctuation)
            words = word_tokenize(text)
            tokens_per_word = 1.3  # Average estimate for English
            estimated_tokens = len(words) * tokens_per_word

            words = word_tokenize(text.lower())
            word_freq = defaultdict(int)

            for word in words:
                if word.isalnum() and word not in self.stop_words:
                    word_freq[word] += 1

            sentence_scores = defaultdict(int)
            sentence_tokens = {}
            
            for sentence in sentences:
                sent_words = word_tokenize(sentence.lower())
                sentence_tokens[sentence] = len(sent_words)
                for word in sent_words:
                    if word in word_freq:
                        sentence_scores[sentence] += word_freq[word]

            # Sort sentences by score
            sorted_sentences = sorted(sentences, key=lambda s: sentence_scores[s], reverse=True)
            
            # Build summary within token limit
            summary_sentences = []
            current_tokens = 0
            
            for sentence in sorted_sentences:
                if current_tokens + sentence_tokens[sentence] <= max_tokens:
                    summary_sentences.append(sentence)
                    current_tokens += sentence_tokens[sentence]
                else:
                    break
                    
            if not summary_sentences:
                # If no complete sentence fits, take the first sentence and truncate
                first_sent_words = word_tokenize(sorted_sentences[0])[:max_tokens-1]
                return {'summary': ' '.join(first_sent_words) + '...'}
                
            # Restore original order
            summary_sentences.sort(key=sentences.index)
            return {'summary': ' '.join(summary_sentences)}

        except Exception as e:
            return {'error': str(e)}

def main():
    summarizer = SimpleSUM()
    text = "This is a sample text. This text will be summarized. This is another sentence. This is a fourth sentence to make it longer than the example in the previous version."
    config = {'maxTokens': 20}  #test with a low token limit
    summary = summarizer.process_text(text, config)
    print(summary)

if __name__ == "__main__":
    main()
