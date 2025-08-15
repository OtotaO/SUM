# SUM - Advanced Text Summarization

> **Fast, accurate text summarization for documents of any size.**

SUM is an open-source text summarization tool that uses extractive summarization techniques to create concise summaries from documents. It supports multiple file formats and provides various summary density options.

## Features

- **Multiple File Formats** - Supports PDF, DOCX, TXT, HTML, RTF, and more
- **Flexible Summarization** - Choose from 5 density levels (tags to detailed)
- **Batch Processing** - Process multiple documents at once
- **Web Interface** - Simple drag-and-drop file upload
- **REST API** - Integrate summarization into your applications
- **Command Line** - Quick summarization from terminal
- **Cross-Platform** - Works on Windows, Mac, and Linux

## Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/SUM.git
cd SUM

# Install dependencies
pip install -r requirements.txt

# Run SUM
python main.py
```

### Alternative Installation Methods

```bash
# Using the install script (Unix/Mac)
./install.sh

# Using the install script (Windows)
install.bat

# Using Docker
docker-compose up
```

## Usage

### Web Interface

1. Open your browser to `http://localhost:5001`
2. Paste text or upload a file
3. Select summary density
4. Click "Summarize"

### Command Line

```bash
# Summarize text
python sum_cli_simple.py "Your text here"

# Summarize a file
python sum_cli_simple.py -f document.pdf

# Specify summary density
python sum_cli_simple.py -f document.pdf --density detailed
```

### API

```python
import requests

# Text summarization
response = requests.post('http://localhost:5001/summarize', 
    json={'text': 'Your long text here'})
print(response.json()['summary'])

# File summarization
with open('document.pdf', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5001/api/file/summarize', files=files)
    print(response.json()['summary'])
```

## Summary Density Levels

SUM provides five density levels to match your needs:

- **Tags** - Extract 5-10 key concepts
- **Minimal** - One-sentence summary
- **Short** - 2-3 sentence overview
- **Medium** - Paragraph-length summary
- **Detailed** - Comprehensive summary with key points

## Technical Details

### Summarization Method

SUM uses extractive summarization based on:
- Sentence importance scoring using word frequencies
- Key concept extraction
- Configurable compression ratios
- NLTK for natural language processing

### Performance

- Typical processing speed: ~1,000 words per second
- Memory usage scales with document size
- Supports documents up to 10MB

### Architecture

- Flask-based REST API
- Modular summarization engines
- File processing pipeline for multiple formats
- Optional Redis caching for repeated requests

## Requirements

- Python 3.8 or higher
- 4GB RAM recommended
- ~500MB disk space for dependencies

### Core Dependencies

- Flask - Web framework
- NLTK - Natural language processing
- PyPDF2 - PDF text extraction
- python-docx - Word document processing
- BeautifulSoup4 - HTML parsing

## Configuration

Create a `.env` file to customize settings:

```bash
# Server Configuration
PORT=5001
DEBUG=False

# Optional Features
REDIS_URL=redis://localhost:6379  # For caching
MAX_FILE_SIZE=10485760  # 10MB
```

## Development

### Running Tests

```bash
pytest Tests/
```

### Project Structure

```
SUM/
├── main.py              # Main application entry
├── summarization_engine.py  # Core summarization logic
├── api/                 # REST API endpoints
├── web/                 # Web interface
├── Utils/              # File processing utilities
└── static/             # Frontend assets
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

Focus areas for contributions:
- Improving summarization accuracy
- Adding new file format support
- Performance optimizations
- UI/UX improvements

## License

Apache License 2.0 - See LICENSE file for details.

## Acknowledgments

SUM is built with open-source technologies and wouldn't be possible without:
- NLTK for NLP capabilities
- Flask for the web framework
- The Python community for excellent libraries

---

**Note**: SUM is under active development. Features and performance may vary. For production use, please test thoroughly with your specific use cases.