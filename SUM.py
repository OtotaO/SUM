import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt

class SUM:
def __init__(self):
self.stop_words = set(stopwords.words('english'))
self.lemmatizer = WordNetLemmatizer()
self.vectorizer = TfidfVectorizer()
self.nlp = spacy.load('en_core_web_lg')
self.running_summary = ""

def load_data(self, data_source):
with open(data_source, 'r') as file:
data = json.load(file)
return data

def preprocess_text(self, text):
words = word_tokenize(text.lower())
words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
preprocessed_text = ' '.join(words)
return preprocessed_text

def generate_summaries(self, texts, max_length=100, min_length=30):
summaries = []
for text in texts:
sentences = sent_tokenize(text)
sentence_scores = self.calculate_sentence_scores(sentences)
summary = self.get_top_sentences(sentences, sentence_scores, max_length, min_length)
summaries.append(summary)
self.update_running_summary(summary)
return summaries

def update_running_summary(self, summary):
self.running_summary += summary + " "

def identify_entities(self, text):
doc = self.nlp(text)
entities = [(ent.text, ent.label_) for ent in doc.ents]
return entities

def identify_main_concept(self, text):
doc = self.nlp(text)
noun_chunks = [chunk.text for chunk in doc.noun_chunks]
main_concept = max(noun_chunks, key=noun_chunks.count)
return main_concept

def identify_main_direction(self, text):
doc = self.nlp(text)
verbs = [token.lemma_ for token in doc if token.pos_ == 'VERB']
main_direction = max(verbs, key=verbs.count)
return main_direction

def calculate_similarity(self, text1, text2):
doc1 = self.nlp(text1)
doc2 = self.nlp(text2)
similarity = doc1.similarity(doc2)
return similarity

def process_knowledge_base(self, knowledge_base):
processed_kb = {}
for concept, value in knowledge_base.items():
processed_kb[concept] = self.preprocess_text(value)
return processed_kb

def identify_topics(self, summaries):
topics = []
for summary in summaries:
doc = self.nlp(summary)
topic = self.identify_main_concept(doc.text)
topics.append(topic)
return topics

def build_knowledge_graph(self, topics):
G = nx.Graph()
for topic in topics:
G.add_node(topic)
for other_topic in topics:
if topic != other_topic:
similarity = self.calculate_similarity(topic, other_topic)
if similarity > 0.5:
G.add_edge(topic, other_topic, weight=similarity)
return G

def visualize_knowledge_graph(self, G):
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=1000, font_size=12, edge_color='gray', node_color='skyblue')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.axis('off')
plt.show()

def calculate_sentence_scores(self, sentences):
vectors = self.vectorizer.fit_transform(sentences)
scores = cosine_similarity(vectors[-1], vectors[:-1])[0]
return scores

def get_top_sentences(self, sentences, scores, max_length, min_length):
ranked_sentences = sorted(zip(scores, sentences), reverse=True)
summary = ranked_sentences[0][1]
for i in range(1, len(ranked_sentences)):
if len(summary) + len(ranked_sentences[i][1]) <= max_length:
summary += " " + ranked_sentences[i][1]
else:
break
if len(summary) < min_length:
for i in range(len(ranked_sentences)):
if len(summary) + len(ranked_sentences[i][1]) <= min_length:
summary += " " + ranked_sentences[i][1]
else:
break
return summary

def run(self, data_source, knowledge_base_source):
data = self.load_data(data_source)
preprocessed_data = [self.preprocess_text(text) for text in data]
summaries = self.generate_summaries(preprocessed_data)

print("Final Running Summary:")
print(self.running_summary)

knowledge_base = self.load_data(knowledge_base_source)
processed_kb = self.process_knowledge_base(knowledge_base)
topics = self.identify_topics(summaries)
G = self.build_knowledge_graph(topics)
self.visualize_knowledge_graph(G)
```
