import os
import json
import time
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
from SUM import MagnumOpusSUM
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid GUI issues
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure necessary folders exist
for folder in [app.config['UPLOAD_FOLDER'], 'static', 'data']:
    os.makedirs(folder, exist_ok=True)

# Initialize MagnumOpusSUM
summarizer = MagnumOpusSUM()

# Load the knowledge base from JSON file if it exists
knowledge_base_path = os.path.join('data', 'knowledge_base.json')
if os.path.exists(knowledge_base_path):
    with open(knowledge_base_path, 'r') as f:
        summarizer.knowledge_base = json.load(f)

@app.route('/')
def index():
    """Renders the main HTML page for interactive interface."""
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.json
        text = data['text']
        level = int(data['level'])
        model_type = data['model']
        
        start_time = time.time()
        
        # Calculate summary length based on level
        max_length = len(text.split())
        target_length = int(max_length * (level / 100))
        
        # Get appropriate model
        if model_type == 'custom' and os.path.exists('custom_model.pkl'):
            with open('custom_model.pkl', 'rb') as f:
                model = pickle.load(f)
        else:
            model = summarizer
            
        result = model.process_text(text, target_length=target_length)
        
        processing_time = int((time.time() - start_time) * 1000)
        compression_ratio = int((len(result['minimum'].split()) / max_length) * 100)
        
        return jsonify({
            'summary': result['minimum'],
            'compression_ratio': compression_ratio,
            'processing_time': processing_time
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload_model', methods=['POST'])
def upload_model():
    try:
        if 'model' not in request.files:
            return jsonify({'error': 'No model file'}), 400
            
        model_file = request.files['model']
        if model_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        model_file.save('custom_model.pkl')
        return jsonify({'message': 'Model uploaded successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_text', methods=['POST'])
def process_text():
    """Processes user-provided text."""
    try:
        text = request.form['text']
        num_topics = int(request.form['num_topics'])
        summary_level = request.form['summary_level']

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        detected_lang = summarizer.detect_language(text)
        if detected_lang != 'en':
            text = summarizer.translate_text(text)

        result = summarizer.process_text(text, num_topics)
        
        wordcloud = summarizer.generate_word_cloud(text)
        wordcloud_path = os.path.join('static', f'wordcloud_{hash(text)}.png')
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

        # Update knowledge base
        summarizer.knowledge_base['summaries'] = summarizer.summaries
        summarizer.knowledge_base['similarity_matrix'] = result.get('similarity_matrix', [])

        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Error processing text: {str(e)}")
        return jsonify({'error': 'An error occurred while processing the text'}), 500

@app.route('/upload_file', methods=['POST'])
def upload_file():
    """Uploads and processes a file."""
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
            text = summarizer.extract_text_from_file(file_path)
            return jsonify({'text': text})
        except Exception as e:
            app.logger.error(f"Error processing file: {str(e)}")
            return jsonify({'error': 'An error occurred while processing the file'}), 500
        finally:
            os.remove(file_path)  # Remove the file after processing

@app.route('/generate_summaries')
def generate_summaries():
    """Generates summaries for existing topics."""
    summaries = summarizer.summaries
    return jsonify({'summaries': summaries})

@app.route('/export_summary', methods=['POST'])
def export_summary():
    """Exports a summary to a text file."""
    summary = request.json['summary']
    filename = 'summary.txt'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(summary)
    return send_file(filename, as_attachment=True)

@app.route('/feedback', methods=['POST'])
def feedback():
    """Receives user feedback and adjusts parameters."""
    try:
        score = int(request.json['score'])
        message = summarizer.adjust_parameters(score)
        return jsonify({'message': message if message else "Feedback recorded. Not enough data to adjust parameters yet."})
    except ValueError:
        return jsonify({'error': 'Invalid feedback score'}), 400

@app.route('/save_progress', methods=['POST'])
def save_progress():
    """Saves user progress to JSON file."""
    progress = request.json['progress']

    with open('progress.json', 'w') as f:
        json.dump(progress, f)

    # Also save the knowledge base
    with open(knowledge_base_path, 'w') as f:
        json.dump(summarizer.knowledge_base, f)

    return jsonify({'message': 'Progress and knowledge base saved successfully!'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
