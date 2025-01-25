
from flask import Flask, request, jsonify, render_template
from SUM import MVPSummarizer

app = Flask(__name__)
summarizer = MVPSummarizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_text', methods=['POST'])
def process_text():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        model_config = data.get('config', {}) if data.get('model') == 'tiny' else None
        result = summarizer.summarize(data['text'])
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
