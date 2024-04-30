import os
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

class SUM:
    def __init__(self):
        self.knowledge_base = {}
        self.data_sources = []
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')

    def load_data(self, data_source):
        with open(data_source, 'r') as f:
            self.knowledge_base.update(json.load(f))
            self.data_sources.append(data_source)

    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ''.join(tokens)

    def calculate_tfidf(self, texts):
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(texts)
        return tfidf, vectorizer

    def calculate_lda(self, tfidf, num_topics):
        lda = LatentDirichletAllocation(num_topics=num_topics, max_iter=5)
        lda_representation = lda.fit_transform(tfidf)
        return lda_representation

    def calculate_cosine_similarity(self, lda_representation):
        similarity = cosine_similarity(lda_representation)
        return similarity

    def visualize_similarity(self, similarity):
        plt.imshow(similarity, interpolation='nearest')
        plt.colorbar()
        plt.show()

    def process_text(self, texts, num_topics):
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        tfidf, vectorizer = self.calculate_tfidf(preprocessed_texts)
        lda_representation = self.calculate_lda(tfidf, num_topics)
        similarity = self.calculate_cosine_similarity(lda_representation)
        self.visualize_similarity(similarity)

    def generate_summaries(self, num_sentences=3):
        summaries = []
        for topic in self.knowledge_base['analyzed_data']:
            summary = '.join([word for word, prob in topic])
            summaries.append(summary)
        return summaries

    def interactive_interface(self):
        print("Welcome to the SUM interactive interface!")
        while True:
            user_input = input("Enter a topic or keyword to explore: ")
            if user_input in self.knowledge_base:
                print("Summary:", self.knowledge_base['summaries'][self.knowledge_base.index(user_input)])
                print("Similar topics:")
                for topic, similarity in zip(self.knowledge_base['analyzed_data'], self.knowledge_base['similarity_matrix'][self.knowledge_base.index(user_input)]):
                    if similarity > 0.5:
                        print(topic)
            else:
                print("Topic not found. Try again!")

            # Allow users to explore related topics
            related_topics = input("Enter a related topic to explore: ")
            if related_topics in self.knowledge_base:
                print("Related topic summary:", self.knowledge_base['summaries'][self.knowledge_base.index(related_topics)])
            else:
                print("Related topic not found. Try again!")

            # Allow users to save their progress
            save_progress = input("Do you want to save your progress? (y/n): ")
            if save_progress.lower() == 'y':
                with open('progress.json', 'w') as f:
                    json.dump(self.knowledge_base, f)
                print("Progress saved!")
            else:
                print("Progress not saved.")

            # Allow users to visualize the topic model
            visualize_topics = input("Do you want to visualize the topic model? (y/n): ")
            if visualize_topics.lower() == 'y':
                import matplotlib.pyplot as plt
                plt.imshow(self.knowledge_base['similarity_matrix'], cmap='hot', interpolation='nearest')
                plt.show()
                print("Topic model visualized!")

    def save_knowledge_base(self):
        with open('knowledge_base.json', 'w') as f:
            json.dump(self.knowledge_base, f)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1)

    def train_model(self, X_train, y_train):
        self.model = MultinomialNB()
        self.model.fit(X_train, y_train)

    def preprocess_data(self, data):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(data)
        y = [0] * len(data)  # dummy labels
        return X, y

    def split_data(self, data):
        X, y = self.preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

# Example usage
sum = SUM()
data = ["This is a sample document.", "This is another document."]
X_train, X_test, y_train, y_test = sum.split_data(data)
sum.train_model(X_train, y_train)
sum.evaluate_model(X_test, y_test)
sum.process_text(data, num_topics=2)
sum.generate_summaries()
sum.interactive_interface()
