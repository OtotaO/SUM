import os
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
from nltk.tokenize import sent_tokenize
from heapq import nlargest
from textblob import TextBlob
import pandas as pd
from wordcloud import WordCloud
from datetime import datetime
from langdetect import detect
from googletrans import Translator
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import PyPDF2
from docx import Document
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from heapq import nlargest
import nltk

class MagnumOpusSUM:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Download necessary NLTK resources
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        # Initialize NLP tools
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.nlp = spacy.load('en_core_web_sm')  # Use smaller model for efficiency
        self.translator = Translator()
        
        # Initialize data structures
        self.summaries = []
        self.feedback_scores = []
        
        # Customizable parameters
        self.num_tags = 5
        self.num_sentences = 3
        self.paragraph_ratio = 0.2

    def preprocess_text(self, text):
        words = word_tokenize(text.lower())
        return ' '.join([self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words])

    def generate_tag_summary(self, text):
        words = word_tokenize(text.lower())
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        return [word for word, _ in nltk.FreqDist(words).most_common(self.num_tags)]

    def generate_sentence_summary(self, text):
        sentences = sent_tokenize(text)
        sentence_scores = self.calculate_sentence_scores(sentences)
        return self.get_top_sentences(sentences, sentence_scores)

    def generate_paragraph_summary(self, text):
        sentences = sent_tokenize(text)
        sentence_scores = self.calculate_sentence_scores(sentences)
        select_length = max(int(len(sentences) * self.paragraph_ratio), 1)
        summary = nlargest(select_length, zip(sentences, sentence_scores), key=lambda x: x[1])
        return ' '.join([s[0] for s in summary])

    def process_text(self, text, model_type='tiny', num_topics=5):
        if model_type == 'tiny':
            # Simple extractive summarization for browser-based processing
            sentences = sent_tokenize(text)
            word_freq = {}
            for sentence in sentences:
                words = word_tokenize(sentence.lower())
                for word in words:
                    if word not in self.stop_words:
                        word_freq[word] = word_freq.get(word, 0) + 1
                        
            sentence_scores = {}
            for sentence in sentences:
                score = sum(word_freq.get(word, 0) for word in word_tokenize(sentence.lower()))
                sentence_scores[sentence] = score
                
            summary_sentences = nlargest(3, sentence_scores, key=sentence_scores.get)
            summary = ' '.join(summary_sentences)
            
            return {'minimum': summary, 'full': text}
            
        preprocessed_text = self.preprocess_text(text)
        
        result = {
            'tags': self.generate_tag_summary(preprocessed_text),
            'minimum': self.generate_sentence_summary(preprocessed_text),
            'full': self.generate_paragraph_summary(preprocessed_text),
            'entities': self.identify_entities(text),
            'main_concept': self.identify_main_concept(text),
            'sentiment': self.sentiment_analysis(text),
            'keywords': self.extract_keywords(preprocessed_text),
            'topics': self.perform_topic_modeling([preprocessed_text], num_topics)
        }
        
        self.summaries.append(result)
        return result

    def identify_entities(self, text):
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def identify_main_concept(self, text):
        doc = self.nlp(text)
        noun_chunks = list(doc.noun_chunks)
        return max(noun_chunks, key=lambda chunk: chunk.root.vector_norm).text if noun_chunks else ""

    def sentiment_analysis(self, text):
        sentiment = TextBlob(text).sentiment.polarity
        if sentiment > 0.1:
            return 'Positive'
        elif sentiment < -0.1:
            return 'Negative'
        else:
            return 'Neutral'

    def extract_keywords(self, text, top_n=5):
        tfidf_matrix = self.vectorizer.fit_transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_scores = zip(feature_names, tfidf_matrix.toarray()[0])
        return [word for word, score in sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:top_n]]

    def calculate_sentence_scores(self, sentences):
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        feature_names = self.vectorizer.get_feature_names_out()
        return [sum(tfidf_matrix[i, self.vectorizer.vocabulary_[word]] 
                    for word in word_tokenize(sentence) if word in feature_names)
                for i, sentence in enumerate(sentences)]

    def get_top_sentences(self, sentences, sentence_scores):
        return " ".join([sentence for sentence, _ in 
                         sorted(zip(sentences, sentence_scores), key=lambda x: x[1], reverse=True)[:self.num_sentences]])

    def perform_topic_modeling(self, texts, num_topics=5):
        vectorized_texts = self.vectorizer.fit_transform(texts)
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda_model.fit(vectorized_texts)
        
        feature_names = self.vectorizer.get_feature_names_out()
        return [f"Topic {i+1}: {', '.join([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]])}" 
                for i, topic in enumerate(lda_model.components_)]

    def detect_language(self, text):
        return detect(text)

    def translate_text(self, text, target_lang='en'):
        detected_lang = self.detect_language(text)
        return self.translator.translate(text, dest=target_lang).text if detected_lang != target_lang else text

    def generate_word_cloud(self, text):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud')
        return plt

    def adjust_parameters(self, feedback_score):
        self.feedback_scores.append(feedback_score)
        if len(self.feedback_scores) < 5:
            return None

        avg_score = sum(self.feedback_scores[-5:]) / 5
        if avg_score < 3:
            self.num_tags = min(10, self.num_tags + 1)
            self.num_sentences = min(5, self.num_sentences + 1)
            self.paragraph_ratio = min(0.3, self.paragraph_ratio + 0.05)
        elif avg_score > 4:
            self.num_tags = max(3, self.num_tags - 1)
            self.num_sentences = max(2, self.num_sentences - 1)
            self.paragraph_ratio = max(0.1, self.paragraph_ratio - 0.05)

        return f"Parameters adjusted: Tags={self.num_tags}, Sentences={self.num_sentences}, Paragraph ratio={self.paragraph_ratio:.2f}"

# Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

summarizer = MagnumOpusSUM()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_text', methods=['POST'])
def process_text():
    try:
        text = request.form['text']
        num_topics = int(request.form['num_topics'])
        summary_level = request.form['summary_level']

        detected_lang = summarizer.detect_language(text)
        if detected_lang != 'en':
            text = summarizer.translate_text(text)

        result = summarizer.process_text(text, num_topics)
        
        wordcloud = summarizer.generate_word_cloud(text)
        wordcloud_path = os.path.join('static', 'wordcloud.png')
        wordcloud.savefig(wordcloud_path)
        plt.close()

        response = {
            'tags': result['tags'],
            'minimum_summary': result['minimum'],
            'full_summary': result['full'],
            'entities': result['entities'],
            'main_concept': result['main_concept'],
            'sentiment': result['sentiment'],
            'keywords': result['keywords'],
            'topics': result['topics'],
            'original_language': detected_lang,
            'wordcloud_path': wordcloud_path,
            'current_level': summary_level
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            text = extract_text_from_file(file_path)
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 400
        finally:
            os.remove(file_path)  # Remove the file after processing
        
        return jsonify({'text': text})

def extract_text_from_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension.lower() in ['.doc', '.docx']:
        return extract_text_from_docx(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        return ' '.join(page.extract_text() for page in reader.pages)

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return ' '.join(paragraph.text for paragraph in doc.paragraphs)

@app.route('/export_summary', methods=['POST'])
def export_summary():
    summary = request.json['summary']
    filename = 'summary.txt'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(summary)
    return send_file(filename, as_attachment=True)

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        score = int(request.json['score'])
        message = summarizer.adjust_parameters(score)
        return jsonify({'message': message if message else "Feedback recorded. Not enough data to adjust parameters yet."})
    except ValueError:
        return jsonify({'error': 'Invalid feedback score'}), 400

if __name__ == '__main__':
    app.run(debug=True)
