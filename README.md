<h1 align="center">
  <img src="https://github.com/OtotaO/SUM/assets/93845604/5749c582-725d-407c-ac6c-06fb8e90ed94" alt="SUM Logo">

</h1>
<h1 align="center">SUM (Summarizer): The Ultimate Knowledge Distiller</h1>

### "Depending on the author, a million books can be distilled into a single sentence, and a single sentence can conceal a million books, depending on the Author." - Me

## Mission Statement

SUM is a knowledge distillation platform that harnesses the power of AI, NLP, and ML to extract, analyze, and present insights from vast datasets in a structured, concise, and engaging manner. With access to potentially all kinds of knowledge, the goal is to summarize it into a succinct & dense human-readable form allowing one to "download" tomes quickly whilst doing away with the "fluff". 

Here is a proof of concept on the amazing Tldraw Computer platform 
![image](https://github.com/user-attachments/assets/9de631e9-7a71-49b8-8313-6d0c6f8324a7)
https://computer.tldraw.com/t/7aR3GPvat7gK5s2TRKGnNG 

And here is a implementation on the mythical Websim 
![image](https://github.com/user-attachments/assets/344b68c8-cba1-4ffd-ade0-5625f5ff8beb)
https://websim.ai/p/vvz4uk4ik02f43adxduf/1 


## Overview

SUM (Summarizer) is an advanced tool for knowledge distillation, leveraging cutting-edge AI, NLP, and ML techniques to transform vast datasets into concise and insightful summaries. Key features include:

- Multi-level summarization (tags, sentences, paragraphs)
- Interactive analysis with user feedback
- Temporal analysis for tracking concept and sentiment changes
- Topic modeling for cross-document analysis
- Knowledge Graph construction and visualization
- Multi-lingual support with language detection and translation
- Adaptive parameter adjustment based on user feedback
- Comprehensive text analysis (entity recognition, keyword extraction, sentiment analysis)
- Word cloud generation
- Data export functionality

## Installation

To install the required libraries, run:

```bash
pip install json nltk spacy scikit-learn networkx matplotlib pandas wordcloud textblob gensim langdetect googletrans==3.1.0a0
python -m spacy download en_core_web_lg
python -m nltk.downloader punkt stopwords wordnet
```

## Usage

### 1. Initialize the Class

```python
from advanced_summarizer import AdvancedSUM

summarizer = AdvancedSUM()
```

### 2. Interactive Analysis

```python
summarizer.simulate_interactive_analysis()
```

### 3. Batch Processing

```python
texts = summarizer.load_data('data.json')
results = summarizer.batch_process(texts)
```

### 4. Temporal Analysis

```python
summarizer.temporal_analysis(results)
```

### 5. Export Results

```python
summarizer.export_results(results, 'analysis_results.json')
```

## Methods

### `load_data(data_source)`

Loads data from a JSON file.

### `process_and_analyze(text, timestamp=None)`

Processes and analyzes a single text with multi-level summarization.

### `batch_process(texts, timestamps=None)`

Processes and analyzes a batch of texts with multi-level summarization.

### `temporal_analysis(results)`

Performs temporal analysis on processed texts.

### `generate_word_cloud(text)`

Generates a word cloud from the text.

### `perform_topic_modeling(texts, num_topics=5)`

Performs topic modeling on a collection of texts.

### `translate_text(text, target_lang='en')`

Translates the text to the target language.

### `build_knowledge_graph(topics)`

Builds a knowledge graph from identified topics.

### `visualize_knowledge_graph(G)`

Visualizes the knowledge graph using NetworkX and Matplotlib.

### `export_results(results, filename='analysis_results.json')`

Exports analysis results to a JSON file.

## Contribution Guidelines

We welcome contributions from the community. If you have ideas for improvements or new features, please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## Contact

For any questions, concerns, or suggestions, please reach out via:

- **X**: https://x.com/Otota0
- **Issues**: [SUM Issues](https://github.com/OtotaO/SUM/issues)

I look forward to your feedback and contributions!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Thank you for using SUM! I hope it helps you distill knowledge effortlessly.

---

<p align="center">Made with ❤️ by ototao</p>
