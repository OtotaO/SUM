import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict

class SimpleSUM:
    def __init__(self):
        try:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('wordnet')
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Error initializing NLTK: {str(e)}")
            raise RuntimeError("Failed to initialize NLTK resources")

    def process_text(self, text):
        if not text.strip():
            return {'error': 'Empty text provided'}

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

            summary_length = max(1, len(sentences) // 3)
            summary_sentences = sorted(sentences, key=lambda s: sentence_scores[s], reverse=True)[:summary_length]
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