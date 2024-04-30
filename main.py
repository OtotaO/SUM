import os
import json
from flask import Flask, request, jsonify, render_template
from SUM import SUM

app = Flask(__name__)

# Load the knowledge base from JSON file
sum = SUM()
with open(os.path.join('data', 'knowledge_base.json'), 'r') as f:
    sum.knowledge_base = json.load(f)

@app.route('/')
def index():
    """Renders the main HTML page for interactive interface."""
    return render_template('index.html')

@app.route('/process_text', methods=['POST'])
def process_text():
    """Processes user-provided text."""
    text = request.form['text']
    num_topics = int(request.form['num_topics'])

    sum.process_text(text, num_topics)

    response = {
        'summaries': sum.knowledge_base['summaries'],
        'similarity_matrix': sum.knowledge_base['similarity_matrix'],
        'data_sources': sum.data_sources,
    }

    return jsonify(response)

@app.route('/generate_summaries')
def generate_summaries():
    """Generates summaries for existing topics."""
    summaries = sum.generate_summaries()

    return jsonify({'summaries': summaries})

@app.route('/save_progress', methods=['POST'])
def save_progress():
    """Saves user progress to JSON file."""
    progress = request.json['progress']

    with open('progress.json', 'w') as f:
        json.dump(progress, f)

    return jsonify({'message': 'Progress saved successfully!'})

if __name__ == '__main__':
    app.run(debug=True)
