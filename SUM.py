import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

class SUM:
    def __init__(self):
        # Download necessary NLTK resources
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
        # Initialize the stop words, lemmatizer, vectorizer, and SpaCy model
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer()
        self.nlp = spacy.load('en_core_web_lg')
        
        # Initialize the running summary and poetic summary
        self.running_summary = ""
        self.poetic_summary = ""

    def load_data(self, data_source):
        """Load data from a JSON file."""
        with open(data_source, 'r') as file:
            data = json.load(file)
        return data

    def preprocess_text(self, text):
        """Preprocess text by tokenizing, removing stop words, and lemmatizing."""
        words = word_tokenize(text.lower())
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        preprocessed_text = ' '.join(words)
        return preprocessed_text

    def generate_summaries(self, texts, max_length=100, min_length=30):
        """Generate summaries from a list of texts."""
        summaries = []
        for text in texts:
            sentences = sent_tokenize(text)
            sentence_scores = self.calculate_sentence_scores(sentences)
            summary = self.get_top_sentences(sentences, sentence_scores, max_length, min_length)
            summaries.append(summary)
            self.update_running_summary(summary)
            self.update_poetic_summary(summary)
            print("Current Running Summary:")
            print(self.running_summary)
            print("\nCurrent Poetic Summary:")
            print(self.poetic_summary)
        return summaries

    def update_running_summary(self, summary):
        """Update the running summary."""
        self.running_summary += summary + " "

    def update_poetic_summary(self, summary):
        """Update the poetic summary using poetic compression."""
        poetic_lines = self.generate_poetic_compression(summary)
        self.poetic_summary += "\n".join(poetic_lines) + "\n"

    def generate_poetic_compression(self, text):
        """Generate a poetic compression of the text."""
        doc = self.nlp(text)
        poetic_lines = []
        for sent in doc.sents:
            poetic_line = ' '.join([token.text for token in sent if token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV'}])
            poetic_lines.append(poetic_line)
        return poetic_lines

    def identify_entities(self, text):
        """Identify named entities in the text using SpaCy."""
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities

    def identify_main_concept(self, text):
        """Identify the main concept in the text."""
        doc = self.nlp(text)
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        main_concept = max(noun_chunks, key=noun_chunks.count)
        return main_concept

    def identify_main_direction(self, text):
        """Identify the main verb (direction) in the text."""
        doc = self.nlp(text)
        verbs = [token.lemma_ for token in doc if token.pos_ == 'VERB']
        main_direction = max(verbs, key=verbs.count)
        return main_direction

    def calculate_similarity(self, text1, text2):
        """Calculate the similarity between two texts."""
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        similarity = doc1.similarity(doc2)
        return similarity

    def process_knowledge_base(self, knowledge_base):
        """Process a knowledge base for use in the knowledge graph."""
        processed_kb = {}
        for concept, value in knowledge_base.items():
            processed_kb[concept] = self.preprocess_text(value)
        return processed_kb

    def identify_topics(self, summaries):
        """Identify topics from a list of summaries."""
        topics = []
        for summary in summaries:
            doc = self.nlp(summary)
            topic = self.identify_main_concept(doc.text)
            topics.append(topic)
        return topics

    def calculate_sentence_scores(self, sentences):
        """Calculate the scores of sentences based on TF-IDF."""
        preprocessed_sentences = [self.preprocess_text(sentence) for sentence
