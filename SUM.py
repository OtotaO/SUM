import os
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from heapq import nlargest
from collections import defaultdict
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import PyPDF2
from docx import Document

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

        if summary_type == 'tags':
            return {'tags': self._get_top_words(word_freq, 5)}
        elif summary_type == 'sum':
            return {'sum': self._get_summary(sentences, sentence_scores, 1)}
        else:  # summary
            return {'summary': self._get_summary(sentences, sentence_scores, 3)}

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

    def _get_top_words(self, word_freq, n):
        return ', '.join(sorted(word_freq, key=word_freq.get, reverse=True)[:n])

    def _get_summary(self, sentences, sentence_scores, n):
        summary_sentences = nlargest(n, sentence_scores, key=sentence_scores.get)
        return ' '.join(summary_sentences)

# Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

summarizer = SimpleSUM()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_text', methods=['POST'])
def process_text():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        text = data.get('text')
        summary_level = data.get('level', 'sum')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        result = summarizer.process_text(text, summary_level)
        
        response = {}
        if summary_level == 'tags':
            response['tags'] = result.get('tags', '')
        elif summary_level == 'sum':
            response['minimum_summary'] = result.get('sum', '')
        else:
            response['full_summary'] = result.get('summary', '')
        
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
    doc = Document(file_path)
    return ' '.join(paragraph.text for paragraph in doc.paragraphs)

@app.route('/export_summary', methods=['POST'])
def export_summary():
    summary = request.json['summary']
    filename = 'summary.txt'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(summary)
    return send_file(filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)