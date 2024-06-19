<h1 align="center">
  <img src="https://github.com/OtotaO/SUM/assets/93845604/5749c582-725d-407c-ac6c-06fb8e90ed94" alt="SUM Logo">

</h1>
<h1 align="center">SUM (Summarizer): The Ultimate Knowledge Distiller</h1>

### "Depending on the author, a million books can be distilled into a single sentence, and a single sentence can conceal a million books, depending on the Author." - Me

## Mission Statement

SUM is a knowledge distillation platform that harnesses the power of AI, NLP, and ML to extract, analyze, and present insights from vast datasets in a structured, concise, and engaging manner. With access to potentially all kinds of knowledge, the goal is to summarize it into a succinct & dense human-readable form allowing one to "download" tomes quickly whilst doing away with the "fluff".

## Overview

SUM (Summarizer) is a powerful tool for knowledge distillation, leveraging advanced AI, NLP, and ML techniques to transform vast datasets into concise and insightful summaries. Key features include:
- Text Preprocessing
- Summarization
- Entity Identification
- Main Concept and Direction Identification
- Similarity Calculation
- Knowledge Graph Construction and Visualization

## Installation

To install the required libraries, run:

```bash
pip install json nltk spacy scikit-learn networkx matplotlib
python -m spacy download en_core_web_lg
```

## Usage

### 1. Initialize the Class

```python
from text_summarizer import SUM

summarizer = SUM()
```

### 2. Load and Preprocess Data

```python
data = summarizer.load_data('data.json')
preprocessed_data = [summarizer.preprocess_text(text) for text in data]
```

### 3. Generate Summaries

```python
summaries = summarizer.generate_summaries(preprocessed_data)
print("Summaries:", summaries)
```

### 4. Identify Entities

```python
entities = summarizer.identify_entities("Your text here")
print("Entities:", entities)
```

### 5. Build and Visualize Knowledge Graph

```python
knowledge_base = summarizer.load_data('knowledge_base.json')
processed_kb = summarizer.process_knowledge_base(knowledge_base)
topics = summarizer.identify_topics(summaries)
G = summarizer.build_knowledge_graph(topics)
summarizer.visualize_knowledge_graph(G)
```

## Methods

### `load_data(data_source)`

Loads data from a JSON file.

### `preprocess_text(text)`

Preprocesses text by tokenizing, removing stop words, and lemmatizing.

### `generate_summaries(texts, max_length=100, min_length=30)`

Generates summaries from a list of texts.

### `identify_entities(text)`

Identifies named entities in the text using SpaCy.

### `identify_main_concept(text)`

Identifies the main concept in the text.

### `identify_main_direction(text)`

Identifies the main verb (direction) in the text.

### `calculate_similarity(text1, text2)`

Calculates the similarity between two texts.

### `process_knowledge_base(knowledge_base)`

Processes a knowledge base for use in the knowledge graph.

### `identify_topics(summaries)`

Identifies topics from a list of summaries.

### `build_knowledge_graph(topics)`

Builds a knowledge graph from identified topics.

### `visualize_knowledge_graph(G)`

Visualizes the knowledge graph using NetworkX and Matplotlib.

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
```
