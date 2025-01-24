import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from heapq import nlargest

class SimpleSUM:
    def __init__(self):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english'))

    def process_text(self, text, summary_type='sum'):
        if not text.strip():
            return {'error': 'Empty text provided'}

        sentences = sent_tokenize(text)
        word_freq = self._calculate_word_freq(text)
        sentence_scores = self._score_sentences(sentences, word_freq)

        summary = self._get_summary(sentences, sentence_scores, 3)
        return {'summary': summary}

    def _calculate_word_freq(self, text):
        word_freq = defaultdict(int)
        for word in word_tokenize(text.lower()):
            if word.isalnum() and word not in self.stop_words:
                word_freq[word] += 1
        return word_freq

    def _score_sentences(self, sentences, word_freq):
        sentence_scores = defaultdict(int)
        for sentence in sentences:
            for word in word_tokenize(sentence.lower()):
                if word in word_freq:
                    sentence_scores[sentence] += word_freq[word]
        return sentence_scores

    def _get_summary(self, sentences, sentence_scores, n):
        summary_sentences = nlargest(n, sentence_scores, key=sentence_scores.get)
        return ' '.join(summary_sentences)

#This is the only remaining part of the original code.  The rest is removed because it is not needed.
#This function is kept because it is used in the new SimpleSUM class.  All other functions are removed.
def main():
    summarizer = SimpleSUM()
    text = "This is a sample text. This text will be summarized.  This is another sentence."
    summary = summarizer.process_text(text)
    print(summary)

if __name__ == "__main__":
    main()