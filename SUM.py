# sum.py

import os
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

class SUM:
    def __init__(self):
        self.knowledge_base = {}
        self.data_sources = []

    def load_data(self, data_source):
        self.knowledge_base.update(json.load(open(data_source, 'r')))
        self.data_sources.append(data_source)

    def analyze_data(self):
        tokenized_data = [word_tokenize(text) for text in self.knowledge_base.values()]
        filtered_data = [' '.join([token for token in tokens if token not in set(stopwords.words('english'))]) for tokens in tokenized_data]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(filtered_data)
        lda_model = LatentDirichletAllocation(n_topics=5)
        topic_matrix = lda_model.fit_transform(tfidf_matrix)
        self.knowledge_base['analyzed_data'] = topic_matrix
        self.knowledge_base['similarity_matrix'] = cosine_similarity(topic_matrix)

    def generate_summaries(self):
        summaries = [' '.join([word for word, prob in topic]) for topic in self.knowledge_base['analyzed_data']]
        self.knowledge_base['summaries'] = summaries

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

            related_topics = input("Enter a related topic to explore: ")
            if related_topics in self.knowledge_base:
                print("Related topic summary:", self.knowledge_base['summaries'][self.knowledge_base.index(related_topics)])
            else:
                print("Related topic not found. Try again!")

            save_progress = input("Do you want to save your progress? (y/n): ")
            if save_progress.lower() == 'y':
                with open('progress.json', 'w') as f:
                    json.dump(self.knowledge_base, f)
                print("Progress saved!")
            else:
                print("Progress not saved.")

            visualize_topics = input("Do you want to visualize the topic model? (y/n): ")
            if visualize_topics.lower() == 'y':
                plt.imshow(self.knowledge_base['similarity_matrix'], cmap='hot', interpolation='nearest')
                plt.show()
                print("Topic model visualized!")

    def save_knowledge_base(self):
        with open('knowledge_base.json', 'w') as f:
            json.dump(self.knowledge_base, f)

    def load_knowledge_base(self):
        with open('knowledge_base.json', 'r') as f:
            self.knowledge_base = json.load(f)

if __name__ == "__main__":
    sum = SUM()
    sum.load_data('data.json')
    sum.analyze_data()
    sum.generate_summaries()
    sum.interactive_interface()
    sum.save_knowledge_base()
