import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import networkx as nx
import matplotlib.pyplot as plt

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
        main_concept = max(noun_chunks, key=noun_chunks.count) if noun_chunks else ""
        return main_concept

    def identify_main_direction(self, text):
        """Identify the main verb (direction) in the text."""
        doc = self.nlp(text)
        verbs = [token.lemma_ for token in doc if token.pos_ == 'VERB']
        main_direction = max(verbs, key=verbs.count) if verbs else ""
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
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        feature_names = self.vectorizer.get_feature_names_out()
        sentence_scores = []
        
        for i, sentence in enumerate(sentences):
            score = sum(tfidf_matrix[i, self.vectorizer.vocabulary_[word]] for word in word_tokenize(sentence) if word in feature_names)
            sentence_scores.append(score)
        
        return sentence_scores

    def get_top_sentences(self, sentences, sentence_scores, max_length=100, min_length=30):
        """Get top sentences based on their scores."""
        sorted_sentences = sorted(zip(sentences, sentence_scores), key=lambda x: x[1], reverse=True)
        summary = ""
        for sentence, score in sorted_sentences:
            if len(summary) + len(sentence) <= max_length:
                summary += sentence + " "
            if len(summary) >= min_length:
                break
        return summary.strip()

    def build_knowledge_graph(self, topics):
        """Build a knowledge graph from identified topics."""
        G = nx.Graph()
        for i, topic in enumerate(topics):
            G.add_node(topic)
            for j, other_topic in enumerate(topics):
                if i != j:
                    similarity = self.calculate_similarity(topic, other_topic)
                    if similarity > 0.5:  # You can adjust this threshold
                        G.add_edge(topic, other_topic, weight=similarity)
        return G

    def visualize_knowledge_graph(self, G):
        """Visualize the knowledge graph using NetworkX and Matplotlib."""
        pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=1500, font_size=10, font_weight='bold')
        edge_weights = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights)
        plt.title("Knowledge Graph")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def extract_keywords(self, text, top_n=5):
        """Extract top keywords from the text using TF-IDF."""
        tfidf_matrix = self.vectorizer.fit_transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_scores = zip(feature_names, tfidf_matrix.toarray()[0])
        sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        return [word for word, score in sorted_scores[:top_n]]

    def generate_word_cloud(self, text):
        """Generate a word cloud from the text."""
        from wordcloud import WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud')
        plt.show()

    def sentiment_analysis(self, text):
        """Perform sentiment analysis on the text."""
        from textblob import TextBlob
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        if sentiment > 0:
            return 'Positive'
        elif sentiment < 0:
            return 'Negative'
        else:
            return 'Neutral'

    def generate_text_summary(self, text, ratio=0.2):
        """Generate a summary using TextRank algorithm."""
        from gensim.summarization import summarize
        return summarize(text, ratio=ratio)

    def export_summary(self, summary, filename='summary.txt'):
        """Export the summary to a text file."""
        with open(filename, 'w') as file:
            file.write(summary)
        print(f"Summary exported to {filename}")

    def process_and_analyze(self, text):
        """Process and analyze the input text."""
        preprocessed_text = self.preprocess_text(text)
        summary = self.generate_text_summary(preprocessed_text)
        entities = self.identify_entities(text)
        main_concept = self.identify_main_concept(text)
        main_direction = self.identify_main_direction(text)
        keywords = self.extract_keywords(preprocessed_text)
        sentiment = self.sentiment_analysis(text)

        print("Summary:", summary)
        print("Entities:", entities)
        print("Main Concept:", main_concept)
        print("Main Direction:", main_direction)
        print("Keywords:", keywords)
        print("Sentiment:", sentiment)

        self.generate_word_cloud(text)
        self.export_summary(summary)

    def batch_process(self, texts):
        """Process and analyze a batch of texts."""
        summaries = self.generate_summaries(texts)
        topics = self.identify_topics(summaries)
        G = self.build_knowledge_graph(topics)
        self.visualize_knowledge_graph(G)

        for i, text in enumerate(texts):
            print(f"\nProcessing text {i+1}:")
            self.process_and_analyze(text)

if __name__ == "__main__":
    # Example usage
    summarizer = SUM()
    
    # Load data (assuming you have a JSON file with texts)
    texts = summarizer.load_data('data.json')
    
    # Process the batch of texts
    summarizer.batch_process(texts)
