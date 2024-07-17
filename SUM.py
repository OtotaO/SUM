import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from gensim.summarization import summarize
from textblob import TextBlob
import pandas as pd
from wordcloud import WordCloud
from datetime import datetime
from langdetect import detect
from googletrans import Translator

class AdvancedSUM:
    def __init__(self):
        # Download necessary NLTK resources
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
        # Initialize NLP tools
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.nlp = spacy.load('en_core_web_lg')
        self.translator = Translator()
        
        # Initialize data structures
        self.summaries = []  # Will store tuples of (tag_summary, sentence_summary, paragraph_summary)
        self.current_summary_level = 0  # 0: tags, 1: sentences, 2: paragraph
        self.feedback_scores = []  # Store user feedback
        
        # Customizable parameters
        self.num_tags = 5
        self.num_sentences = 3
        self.paragraph_ratio = 0.2

    def load_data(self, data_source):
        """Load data from a JSON file."""
        with open(data_source, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def preprocess_text(self, text):
        """Preprocess text by tokenizing, removing stop words, and lemmatizing."""
        words = word_tokenize(text.lower())
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        return ' '.join(words)

    def generate_tag_summary(self, text):
        """Generate a summary as a list of tags."""
        words = word_tokenize(text.lower())
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        freq_dist = nltk.FreqDist(words)
        return [word for word, _ in freq_dist.most_common(self.num_tags)]

    def generate_sentence_summary(self, text):
        """Generate a summary of a few sentences."""
        sentences = sent_tokenize(text)
        sentence_scores = self.calculate_sentence_scores(sentences)
        top_sentences = self.get_top_sentences(sentences, sentence_scores, max_length=len(' '.join(sentences[:self.num_sentences])))
        return top_sentences

    def generate_paragraph_summary(self, text):
        """Generate a paragraph summary."""
        return summarize(text, ratio=self.paragraph_ratio)

    def generate_multi_level_summary(self, text):
        """Generate summaries at all three levels."""
        tag_summary = self.generate_tag_summary(text)
        sentence_summary = self.generate_sentence_summary(text)
        paragraph_summary = self.generate_paragraph_summary(text)
        self.summaries.append((tag_summary, sentence_summary, paragraph_summary))
        return tag_summary, sentence_summary, paragraph_summary

    def get_current_summary(self, index=-1):
        """Get the summary at the current level for the most recent text."""
        if not self.summaries:
            return "No summaries available."
        return self.summaries[index][self.current_summary_level]

    def cycle_summary_level(self):
        """Cycle to the next summary level."""
        self.current_summary_level = (self.current_summary_level + 1) % 3
        return self.get_current_summary()

    def analyze_summary_level(self, text, level):
        """Generate insights based on the current summary level."""
        if level == 0:
            return f"Key concepts: {', '.join(self.get_current_summary())}"
        elif level == 1:
            return f"Main ideas: {self.get_current_summary()}"
        else:
            entities = self.identify_entities(self.get_current_summary())
            return f"Detailed summary with key entities: {entities}"

    def identify_entities(self, text):
        """Identify named entities in the text using SpaCy."""
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities

    def identify_main_concept(self, text):
        """Identify the main concept in the text."""
        doc = self.nlp(text)
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        return max(noun_chunks, key=noun_chunks.count) if noun_chunks else ""

    def identify_main_direction(self, text):
        """Identify the main verb (direction) in the text."""
        doc = self.nlp(text)
        verbs = [token.lemma_ for token in doc if token.pos_ == 'VERB']
        return max(verbs, key=verbs.count) if verbs else ""

    def calculate_similarity(self, text1, text2):
        """Calculate the similarity between two texts."""
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        return doc1.similarity(doc2)

    def extract_keywords(self, text, top_n=5):
        """Extract top keywords from the text using TF-IDF."""
        tfidf_matrix = self.vectorizer.fit_transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_scores = zip(feature_names, tfidf_matrix.toarray()[0])
        return [word for word, score in sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:top_n]]

    def generate_word_cloud(self, text):
        """Generate a word cloud from the text."""
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud')
        plt.show()

    def sentiment_analysis(self, text):
        """Perform sentiment analysis on the text."""
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        if sentiment > 0.1:
            return 'Positive'
        elif sentiment < -0.1:
            return 'Negative'
        else:
            return 'Neutral'

    def calculate_sentence_scores(self, sentences):
        """Calculate the scores of sentences based on TF-IDF."""
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        feature_names = self.vectorizer.get_feature_names_out()
        sentence_scores = []
        
        for i, sentence in enumerate(sentences):
            score = sum(tfidf_matrix[i, self.vectorizer.vocabulary_[word]] for word in word_tokenize(sentence) if word in feature_names)
            sentence_scores.append(score)
        
        return sentence_scores

    def get_top_sentences(self, sentences, sentence_scores, max_length=100):
        """Get top sentences based on their scores."""
        sorted_sentences = sorted(zip(sentences, sentence_scores), key=lambda x: x[1], reverse=True)
        summary = ""
        for sentence, score in sorted_sentences:
            if len(summary) + len(sentence) <= max_length:
                summary += sentence + " "
            if len(summary.split()) >= self.num_sentences:
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

    def perform_topic_modeling(self, texts, num_topics=5):
        """Perform topic modeling on a collection of texts."""
        vectorized_texts = self.vectorizer.fit_transform(texts)
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda_output = lda_model.fit_transform(vectorized_texts)
        
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
            topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
        return topics

    def detect_language(self, text):
        """Detect the language of the text."""
        return detect(text)

    def translate_text(self, text, target_lang='en'):
        """Translate the text to the target language."""
        detected_lang = self.detect_language(text)
        if detected_lang != target_lang:
            return self.translator.translate(text, dest=target_lang).text
        return text

    def process_and_analyze(self, text, timestamp=None):
        """Process and analyze the input text with multi-level summarization."""
        if timestamp is None:
            timestamp = datetime.now()

        # Detect language and translate if necessary
        detected_lang = self.detect_language(text)
        if detected_lang != 'en':
            text = self.translate_text(text)

        preprocessed_text = self.preprocess_text(text)
        tag_summary, sentence_summary, paragraph_summary = self.generate_multi_level_summary(preprocessed_text)
        
        analysis_result = {
            'timestamp': timestamp,
            'original_language': detected_lang,
            'tag_summary': tag_summary,
            'sentence_summary': sentence_summary,
            'paragraph_summary': paragraph_summary,
            'entities': self.identify_entities(text),
            'main_concept': self.identify_main_concept(text),
            'main_direction': self.identify_main_direction(text),
            'keywords': self.extract_keywords(preprocessed_text),
            'sentiment': self.sentiment_analysis(text)
        }

        return analysis_result

    def batch_process(self, texts, timestamps=None):
        """Process and analyze a batch of texts with multi-level summarization."""
        if timestamps is None:
            timestamps = [datetime.now() for _ in texts]

        results = []
        for text, timestamp in zip(texts, timestamps):
            result = self.process_and_analyze(text, timestamp)
            results.append(result)
            print(f"\nProcessed text at {timestamp}:")
            for key, value in result.items():
                print(f"{key}: {value}")

        # Perform cross-document analysis
        all_summaries = [result['paragraph_summary'] for result in results]
        topics = self.perform_topic_modeling(all_summaries)
        print("\nTopic Modeling Results:")
        for topic in topics:
            print(topic)

        # Build and visualize knowledge graph
        G = self.build_knowledge_graph(all_summaries)
        self.visualize_knowledge_graph(G)

        return results

    def temporal_analysis(self, results):
        """Perform temporal analysis on processed texts."""
        df = pd.DataFrame(results)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # Analyze how main concepts change over time
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['main_concept'], marker='o')
        plt.title('Evolution of Main Concepts Over Time')
        plt.xlabel('Timestamp')
        plt.ylabel('Main Concept')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Analyze sentiment changes over time
        sentiment_scores = df['sentiment'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], sentiment_scores, marker='o')
        plt.title('Sentiment Changes Over Time')
        plt.xlabel('Timestamp')
        plt.ylabel('Sentiment Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def get_user_feedback(self, summary):
        """Get user feedback on summary quality."""
        print("\nPlease rate the quality of this summary (1-5):")
        print(summary)
        while True:
            try:
                score = int(input("Your rating: "))
                if 1 <= score <= 5:
                    self.feedback_scores.append(score)
                    return score
                else:
                    print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def adjust_parameters(self):
        """Adjust summarization parameters based on user feedback."""
        if len(self.feedback_scores) < 5:
            return  # Not enough feedback to make adjustments

        avg_score = sum(self.feedback_scores) / len(self.feedback_scores)
        if avg_score < 3:
            # If average score is low, increase detail
            self.num_tags = min(10, self.num_tags + 1)
            self.num_sentences = min(5, self.num_sentences + 1)
            self.paragraph_ratio = min(0.3, self.paragraph_ratio + 0.05)
        elif avg_score > 4:
            # If average score is high, we can potentially reduce detail
            self.num_tags = max(3, self.num_tags - 1)
            self.num_sentences = max(2, self.num_sentences - 1)
            self.paragraph_ratio = max(0.1, self.paragraph_ratio - 0.05)

        print(f"Parameters adjusted: Tags={self.num_tags}, Sentences={self.num_sentences}, Paragraph ratio={self.paragraph_ratio:.2f}")

    def export_results(self, results, filename='analysis_results.json'):
        """Export analysis results to a JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4, default=str)
        print(f"Results exported to {filename}")

    def simulate_interactive_analysis(self):
        """Simulate an interactive analysis session."""
        print("\nWelcome to the Advanced Multi-Level Summarization and Analysis System!")
        print("Enter texts to analyze. Type 'quit' to exit.")

        results = []
        while True:
            text = input("\nEnter text to analyze (or 'quit'): ")
            if text.lower() == 'quit':
                break

            result = self.process_and_analyze(text)
            results.append(result)

            print("\nAnalysis Result:")
            for key, value in result.items():
                print(f"{key}: {value}")

            print("\nSummary Levels:")
            for _ in range(3):
                current_summary = self.get_current_summary()
                print(f"Level {self.current_summary_level + 1}: {current_summary}")
                self.cycle_summary_level()

            feedback = self.get_user_feedback(result['paragraph_summary'])
            print(f"Thank you for your feedback! (Score: {feedback})")

            self.adjust_parameters()

        if results:
            self.temporal_analysis(results)
            self.export_results(results)

if __name__ == "__main__":
    # Example usage
    summarizer = AdvancedSUM()
    
    # Option 1: Load data from a JSON file
    # texts = summarizer.load_data('data.json')
    # results = summarizer.batch_process(texts)

    # Option 2: Interactive analysis
    summarizer.simulate_interactive_analysis()

    print("Analysis complete. Thank you for using the SUM system!")
