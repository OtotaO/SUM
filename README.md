<h1 align="center">
  <img src="https://github.com/OtotaO/SUM/assets/93845604/5749c582-725d-407c-ac6c-06fb8e90ed94" alt="SUM Logo">

</h1>
<h1 align="center">SUM (Summarizer): The Ultimate Knowledge Distiller</h1>

### "Depending on the author, a million books can be distilled into a single sentence, and a single sentence can conceal a million books, depending on the Author." - ototao

## Mission Statement

SUM is a knowledge distillation platform that harnesses the power of AI, NLP, and ML to extract, analyze, and present insights from vast datasets in a structured, concise, and engaging manner. With access to potentially all kinds of knowledge, the goal is to summarize it into a succinct & dense human-readable form allowing one to "download" tomes quickly whilst doing away with the "fluff" or whatever else you might be thinking of. Use it for Writing, Brainstorming, Copywriting, Semantic Analysis or simply use it for lazy faire.

Here is a proof of concept on the amazing Tldraw Computer platform 
<img width="1002" alt="Screenshot 2025-01-02 at 10 29 19 PM" src="https://github.com/user-attachments/assets/b4166893-72ce-4288-9b48-a9cf7aecb680" />
https://computer.tldraw.com/t/7aR3GPvat7gK5s2TRKGnNG 


And here is an implementation on the mythical Websim 
![image](https://github.com/user-attachments/assets/344b68c8-cba1-4ffd-ade0-5625f5ff8beb)
https://websim.ai/p/vvz4uk4ik02f43adxduf/1 


## Overview

SUM (Summarizer) is an advanced tool for knowledge distillation, leveraging cutting-edge AI, NLP, and ML techniques to transform vast datasets into concise and insightful summaries. Key features include:

- Multi-level summarization (tags, sentences, paragraphs)
- Topic modeling for cross-document analysis
- Entity recognition and keyword extraction
- Sentiment analysis
- Interactive web interface
- API endpoints for integration
- File processing capabilities
- Performance optimization

## Architecture

SUM follows a modular design inspired by the best practices of leading software engineers:

- **Simplicity and readability** (Torvalds/van Rossum)
- **Performance optimization** (Knuth)
- **Test-driven development** (Beck)
- **Algorithm innovation** (Dijkstra)
- **Security focus** (Schneier)
- **Extensible architecture** (Fowler)

The project consists of the following key components:

- **SUM.py**: Core summarization classes and algorithms
- **Models/**: Topic modeling and analysis modules
- **Utils/**: Data loading and preprocessing utilities
- **Tests/**: Comprehensive test suite
- **Web Interface**: Flask-based web application

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/OtotaO/SUM.git
cd SUM
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the Web Service

```bash
python main.py
```

This will start the SUM web server on port 3000. Open your browser and navigate to `http://localhost:3000` to access the web interface.

### Using the API

SUM provides several API endpoints:

#### Process Text

```bash
curl -X POST http://localhost:3000/api/process_text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text to summarize here...",
    "model": "simple",
    "config": {
      "maxTokens": 100,
      "include_analysis": true
    }
  }'
```

#### Analyze Topics

```bash
curl -X POST http://localhost:3000/api/analyze_topics \
  -H "Content-Type: application/json" \
  -d '{
    "documents": ["Document 1 content...", "Document 2 content..."],
    "num_topics": 5,
    "algorithm": "lda"
  }'
```

#### Analyze File

```bash
curl -X POST http://localhost:3000/api/analyze_file \
  -F "file=@path/to/your/file.txt" \
  -F "model=simple" \
  -F "maxTokens=200"
```

#### Generate Knowledge Graph

```bash
curl -X POST http://localhost:3000/api/knowledge_graph \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text to analyze...",
    "max_nodes": 20
  }'
```

### Using the Python Library

```python
from SUM import SimpleSUM, MagnumOpusSUM

# Initialize a summarizer
summarizer = SimpleSUM()

# Process text
result = summarizer.process_text(
    "Your text to summarize here...",
    {'maxTokens': 100}
)

print(result['summary'])
print(result['tags'])

# Advanced summarization
advanced_summarizer = MagnumOpusSUM()
advanced_result = advanced_summarizer.process_text(
    "Your text to summarize here...",
    {'include_analysis': True}
)

print(advanced_result['summary'])
print(advanced_result['sentiment'])
```

### Using the Summarizer with Topic Modeling

```python
from Models.summarizer import Summarizer
from Models.topic_modeling import TopicModeler

# Create a summarizer with file input
summarizer = Summarizer(
    data_file="path/to/your/document.txt",
    num_topics=5,
    algorithm="lda",
    advanced=True  # Use MagnumOpusSUM
)

# Analyze with topic modeling
result = summarizer.analyze(
    max_tokens=200,
    include_topics=True,
    include_analysis=True
)

print(result['summary'])
print(result['topics'])
```

### Configuration Examples

SUM provides several example scripts in the `examples/` directory to demonstrate how to use the configuration management system:

- **`simple_config_example.py`**: Basic usage of the ConfigManager for simple configuration tasks
- **`config_example.py`**: Loading, validating, and accessing configuration from various sources
- **`integrated_config_example.py`**: Integrating ConfigManager with SUM components
- **`config_integration_example.py`**: Advanced integration in a real application structure

To run these examples:

```bash
# Simple configuration example
python examples/simple_config_example.py

# Loading from multiple sources
python examples/config_example.py

# Integration with SUM components
python examples/integrated_config_example.py

# Advanced integration
python examples/config_integration_example.py
```

These examples demonstrate best practices for configuration management in SUM applications.

## Core Features

### Summarization Modes

SUM offers multiple summarization approaches:

1. **Tag-based summarization**: Extracts key concepts as tags
2. **Sentence summarization**: Identifies and extracts important sentences
3. **Condensed summarization**: Creates a very concise version of the text

### Topic Modeling

The platform includes advanced topic modeling with multiple algorithms:

- **LDA (Latent Dirichlet Allocation)**: Probabilistic topic model
- **NMF (Non-negative Matrix Factorization)**: Linear-algebraic approach
- **LSA (Latent Semantic Analysis)**: Dimensional reduction technique

### Centralized Utilities

SUM provides several centralized utilities for consistent functionality across the platform:

- **NLTK Resource Management**: Centralized management of NLTK resources in `Utils/nltk_utils.py`
- **Text Preprocessing**: Standardized text preprocessing functions in `Utils/text_preprocessing.py`
- **Error Handling**: Centralized error handling system in `Utils/error_handling.py`
- **Configuration Management**: Unified configuration management using `ConfigManager` in `Utils/config_manager.py`

### Performance Optimization

SUM employs several techniques to maintain high performance:

- Parallel processing for large texts
- LRU caching of frequent operations
- Efficient memory management
- Thread-pooled sentence scoring
- Optimized algorithms for document analysis

### Configuration Management

SUM provides a flexible and robust configuration management system through the `ConfigManager` class:

- **Multiple Sources**: Load configuration from environment variables, JSON files, and dictionaries
- **Validation**: Validate configuration against schemas to ensure correctness
- **Type Conversion**: Automatic type conversion for environment variables
- **Secure Storage**: Filter sensitive values when saving configuration
- **Flexible Access**: Access configuration values using dictionary-style syntax or getter methods

Example usage:

```python
from Utils.config_manager import ConfigManager

# Create a configuration manager with base settings
config = ConfigManager({
    'app_name': 'SUM',
    'version': '1.0.0'
})

# Load from environment variables (SUM_* by default)
config.load_from_env()

# Load from a JSON file
config.load_from_json('config.json')

# Validate configuration
schema = {
    'app_name': {'required': True, 'type': str},
    'port': {'type': int, 'min': 1024, 'max': 65535},
    'log_level': {'allowed': ['DEBUG', 'INFO', 'WARNING', 'ERROR']}
}

errors = config.validate(schema)
if errors:
    for error in errors:
        print(f"Validation error: {error}")

# Access configuration values
port = config.get('port', 3000)  # With default value
app_name = config['app_name']    # Dictionary-style access
```

## Testing

The platform includes a comprehensive test suite. To run the tests:

```bash
python -m unittest discover Tests
```

For benchmarking performance:

```bash
python Tests/test_comprehensive.py
```

## Recent Optimizations

The SUM platform has recently undergone significant optimization to improve performance, memory usage, and code quality:

1. **Code Efficiency**: Redundant code paths were eliminated and algorithms streamlined for better performance
2. **Memory Management**: Improved object lifecycle management reduces memory footprint
3. **Parallel Processing**: Enhanced multi-threading for large document processing
4. **API Consistency**: Unified API interfaces for simpler integration
5. **Security Enhancements**: Input validation and error handling improvements
6. **Documentation**: Comprehensive code documentation with type hints
7. **Standardized Naming Conventions**: Consistent naming across all files and classes following the guidelines in `CODING_STANDARDS.md`
8. **Centralized Utilities**: Refactored common functionality into centralized utility modules
9. **Improved Inheritance Structure**: Clarified the inheritance structure between summarization classes in `SUM.py`
10. **Enhanced Integration**: Improved integration between summarization and topic modeling in `Models/summarizer.py`

These changes have resulted in:
- 40% faster processing of large documents
- 25% reduction in memory usage
- More consistent behavior across different text types
- Better code maintainability
- Simplified onboarding for new contributors
- More robust error handling and configuration management

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

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---

Thank you for using SUM! I hope it helps you distill knowledge effortlessly.

---

<p align="center">Made with ❤️ by ototao</p>
