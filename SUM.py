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

        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= 2:
                return {'summary': text}

            # Get max tokens from config or calculate based on input length
            words = word_tokenize(text)
            input_length = len(words)
            default_tokens = min(input_length // 2, 200)  # Target 1/2 of input length, max 200 tokens
            max_tokens = model_config.get('maxTokens', default_tokens) if model_config else default_tokens

            # Approximate tokens (rough estimate: words + punctuation)
            tokens_per_word = 1.3  # Average estimate for English
            estimated_tokens = len(words) * tokens_per_word

            words = word_tokenize(text.lower())
            word_freq = defaultdict(int)

            for word in words:
                if word.isalnum() and word not in self.stop_words:
                    word_freq[word] += 1

            sentence_scores = defaultdict(int)
            sentence_tokens = {}
            
            # Track articles and sections for hierarchical summarization
            current_article = None
            current_section = None
            article_section_sentences = defaultdict(lambda: defaultdict(list))
            
            for sentence in sentences:
                sent_words = word_tokenize(sentence.lower())
                sentence_tokens[sentence] = len(sent_words)
                
                # Identify articles and sections
                if "article" in sentence.lower():
                    current_article = sentence
                    current_section = None
                    sentence_scores[sentence] = 15  # Boost article headers
                elif "section" in sentence.lower():
                    current_section = sentence
                    sentence_scores[sentence] = 10  # Boost section headers
                elif current_article and current_section:
                    article_section_sentences[current_article][current_section].append(sentence)
                
                # Score sentences with constitutional context
                base_score = sum(word_freq[word] for word in sent_words if word in word_freq)
                key_terms = ["congress", "senate", "house", "representatives", "president", 
                           "united states", "power", "legislative", "executive", "judicial"]
                term_matches = sum(term in sentence.lower() for term in key_terms)
                if term_matches > 0:
                    base_score *= (1 + 0.2 * term_matches)  # Proportional boost based on matches
                
                sentence_scores[sentence] = base_score
            
            # Balance section representation
            for article in article_section_sentences.values():
                for section_sentences in article.values():
                    if section_sentences:
                        # Boost the highest scoring sentence from each section
                        best_sentence = max(section_sentences, key=lambda s: sentence_scores[s])
                        sentence_scores[best_sentence] *= 2

            # Sort sentences by score
            sorted_sentences = sorted(sentences, key=lambda s: sentence_scores[s], reverse=True)
            
            # Build summary with strict sentence control
            summary_sentences = []
            max_sentences = int(model_config.get('maxSentences', 5)) if model_config else 5
            
            for sentence in sorted_sentences:
                if len(summary_sentences) < max_sentences:
                    summary_sentences.append(sentence)
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
    text = "This is a sample text. This text will be summarized. This is another sentence.  This is a fourth sentence to make it longer than the example in the previous version."
    config = {'maxTokens': 20} #test with a low token limit
    summary = summarizer.process_text(text, config)
    print(summary)

if __name__ == "__main__":
    main()