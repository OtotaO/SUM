import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleSUM:
    def __init__(self):
        try:
            # Create nltk_data directory
            nltk_data_dir = os.path.expanduser('~/nltk_data')
            os.makedirs(nltk_data_dir, exist_ok=True)

            # Download required NLTK resources
            for resource in ['punkt', 'stopwords']:
                try:
                    nltk.download(resource, download_dir=nltk_data_dir, quiet=True, raise_on_error=False)
                except Exception as e:
                    logging.error(f"Error downloading {resource}: {str(e)}")

            self.stop_words = set(stopwords.words('english'))
            logging.info("SimpleSUM initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing NLTK: {str(e)}")
            raise RuntimeError("Failed to initialize NLTK resources")

    def process_text(self, text, model_config=None):
        if not text.strip():
            logging.warning("Empty text provided.")
            return {'error': 'Empty text provided'}

        try:
            logging.info("Starting text processing.")
            sentences = sent_tokenize(text)
            if len(sentences) <= 2:
                logging.info("Text has <= 2 sentences, returning original text.")
                return {'summary': text}

            # Get max tokens from config
            max_tokens = model_config.get('maxTokens', 100) if model_config else 100
            logging.info(f"Max tokens set to: {max_tokens}")

            # Tokenize the text to get an accurate token count
            words = word_tokenize(text)
            estimated_tokens = len(words)
            logging.info(f"Estimated tokens: {estimated_tokens}")

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
            logging.debug(f"Sorted sentences: {sorted_sentences}")
            
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
                first_sentence = sorted_sentences[0]
                first_sent_words = word_tokenize(first_sentence)[:max_tokens-1]
                summary = ' '.join(first_sent_words) + '...'
                logging.info("No complete sentence fits, returning truncated first sentence.")
                return {'summary': summary}
                
            # Restore original order
            summary_sentences.sort(key=sentences.index)
            summary = ' '.join(summary_sentences)
            logging.info(f"Generated summary: {summary}")
            return {'summary': summary}

        except Exception as e:
            logging.error(f"Error during text processing: {str(e)}")
            return {'error': f"Error during text processing: {str(e)}"}

def main():
    summarizer = SimpleSUM()
    text = "This is a sample text. This text will be summarized. This is another sentence.  This is a fourth sentence to make it longer than the example in the previous version."
    config = {'maxTokens': 20} #test with a low token limit
    summary = summarizer.process_text(text, config)
    print(summary)

if __name__ == "__main__":
    main()