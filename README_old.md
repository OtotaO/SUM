# SUM: Advanced Text Summarization API

> **Transform any text into summaries at multiple density levels**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)

## What is SUM?

SUM is a high-performance text summarization platform that provides:
- **Flexible summarization** - From single-sentence to comprehensive summaries
- **Universal file support** - Process PDFs, Word documents, HTML, and plain text
- **Scalable architecture** - Handle documents of any length through intelligent chunking
- **Real-time processing** - Stream summaries as documents are processed
- **Production-ready** - Redis caching, rate limiting, and REST API

## Features

### Core Functionality
- **Multi-Level Summarization**
  - Tags: Extract key terms and entities
  - Minimal: Single sentence summary
  - Short: One paragraph summary
  - Medium: 2-3 paragraph summary
  - Detailed: Comprehensive multi-paragraph summary
- **Universal File Processing** - Extract text from ANY file type with intelligent fallbacks
  - Supports 50+ common formats out of the box
  - Binary file text extraction
  - Automatic encoding detection
  - Graceful fallback for unknown types
- **Clipboard Integration** - Instant capture from anywhere with global hotkey (Ctrl+Shift+T)
- **Long Document Support** - Intelligent chunking for documents exceeding model limits
- **Real-time Progress** - Live progress visualization for large file processing
- **Streaming API** - Real-time progress updates via Server-Sent Events
- **Performance Optimization** - Redis caching with configurable TTL
- **Rate Limiting** - Protect API from abuse

### Technical Specifications
- Built on Hugging Face Transformers (BART model)
- RESTful API design
- Horizontal scaling support
- Docker-ready deployment
- Comprehensive error handling

## User Interfaces

SUM provides three ways to interact:

### 1. Web Interface
Navigate to `http://localhost:3000` after starting the server

**Features:**
- 📁 Drag & drop ANY file type - universal text extraction
- 🎯 Multiple density levels (tags → detailed)
- 📋 Copy-to-clipboard for all summaries
- ⌨️ Global hotkey capture (Ctrl+Shift+T from anywhere!)
- 📊 Real-time progress for large files
- 💾 Auto-save drafts
- 🔄 Auto-retry on connection errors
- 📈 Live statistics and word count

### 2. Command Line Interface
```bash
# Use the CLI
./sum cli text "Your text here"
./sum cli file document.pdf --density minimal
./sum cli stream --file book.txt
./sum cli examples
```

### 3. REST API
Direct API access for integration (see API Documentation below)

## Quick Start

### Option 1: One-Line Install (Recommended)
```bash
curl -sSL https://raw.githubusercontent.com/OtotaO/SUM/main/install.sh | bash
```

### Option 2: Docker (Easiest)
```bash
git clone https://github.com/OtotaO/SUM.git && cd SUM
docker-compose -f docker-compose-simple.yml up
# Access at http://localhost:3000
```

### Option 3: Manual Setup
```bash
# Clone repository
git clone https://github.com/OtotaO/SUM.git
cd SUM

# Install dependencies
pip install -r requirements.txt

# Download language models (for entity extraction)
python -m spacy download en_core_web_sm

# Start Redis (optional but recommended)
docker run -d -p 6379:6379 redis:7-alpine

# Run server
python main.py

# Access web UI at http://localhost:3000
```

### Option 4: Quick Knowledge Crystallization Setup
```bash
# Clone and setup
git clone https://github.com/OtotaO/SUM.git && cd SUM
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Initialize knowledge systems
python -c "
from memory.semantic_memory import get_semantic_memory_engine
from memory.knowledge_graph import get_knowledge_graph_engine
print('Initializing knowledge systems...')
memory = get_semantic_memory_engine()
kg = get_knowledge_graph_engine()
print('Knowledge crystallization ready!')
"

# Start server
python main.py
```


## API Documentation

### Summarize Text
```bash
POST /summarize/ultimate
Content-Type: application/json

{
  "text": "Your text here...",
  "density": "all" | "tags" | "minimal" | "short" | "medium" | "detailed"
}
```

### Process Files
```bash
POST /summarize/ultimate
Content-Type: multipart/form-data

file: [binary]
density: "minimal"
```

### Stream Processing
```bash
POST /summarize/stream
Content-Type: application/json

{
  "text": "Long document text..."
}
```

Returns Server-Sent Events with progress updates.

### Response Format
```json
{
  "result": {
    "tags": ["keyword1", "keyword2", "keyword3"],
    "minimal": "Single sentence summary.",
    "short": "One paragraph summary of the content...",
    "medium": "More detailed summary spanning multiple paragraphs...",
    "detailed": "Comprehensive summary with full details...",
    "original_words": 5000,
    "compression_ratio": 25.5
  },
  "cached": false
}
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│              REST API Layer                      │
│  /summarize  /memory  /knowledge  /synthesize   │
├─────────────────────────────────────────────────┤
│         Knowledge Crystallization Layer          │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────┐ │
│  │  Semantic   │ │  Knowledge   │ │ Synthesis│ │
│  │   Memory    │ │    Graph     │ │  Engine  │ │
│  └─────────────┘ └──────────────┘ └──────────┘ │
├─────────────────────────────────────────────────┤
│         Async Processing Pipeline                │
│  - Concurrent document processing                │
│  - Stream-based file handling                    │
│  - Real-time progress tracking                   │
├─────────────────────────────────────────────────┤
│         Core Summarization Engine                │
│  - Multi-density summarization                   │
│  - Universal file processing                     │
│  - Intelligent text extraction                   │
├─────────────────────────────────────────────────┤
│      Storage & Persistence Layer                 │
│  ┌──────────┐ ┌────────────┐ ┌───────────────┐ │
│  │  Vector  │ │   Graph    │ │     Redis     │ │
│  │    DB    │ │     DB     │ │    Cache      │ │
│  └──────────┘ └────────────┘ └───────────────┘ │
└─────────────────────────────────────────────────┘
```

## Performance Metrics

- **Response Time**: <2s for standard documents
- **Throughput**: 100+ requests/minute
- **Max Document Size**: No hard limit (chunked processing)
- **Compression Ratios**: 10:1 to 100:1 depending on density
- **Cache Hit Rate**: >90% for repeated content

## Configuration

Environment variables:
```bash
REDIS_URL=redis://localhost:6379     # Redis connection string
MAX_TEXT_LENGTH=100000               # Maximum text length per request
RATE_LIMIT=60                        # Requests per minute per IP
CACHE_TTL=3600                       # Cache expiration in seconds
MODEL_NAME=facebook/bart-large-cnn   # Transformer model to use
SUM_UNIVERSAL_FILE_SUPPORT=True      # Accept any file extension
SUM_MAX_CONTENT_LENGTH=104857600     # Max upload size (100MB default)
```

## Testing

```bash
# Run unit tests
pytest tests/

# Run integration tests
python test_simple.py

# Load testing
python load_test.py --concurrent 10 --requests 1000
```

## Deployment

### Docker
```bash
docker build -t sum-api .
docker run -p 3000:3000 sum-api
```

### Docker Compose
```bash
docker-compose up -d
```

### Kubernetes
```bash
kubectl apply -f k8s/
```

## Advanced Features

### Custom Models
```python
# Use a different summarization model
SUMMARIZER_MODEL = "google/pegasus-xsum"
```

### Batch Processing
```python
# Process multiple documents
POST /batch/summarize
{
  "documents": [
    {"id": "1", "text": "..."},
    {"id": "2", "text": "..."}
  ]
}
```

## What's New in This Version

### 🧠 Knowledge Crystallization - Transform Information into Intelligence
- **Semantic Memory System**: Vector-based storage with intelligent retrieval
- **Knowledge Graph**: Entity extraction and relationship mapping
- **Cross-Document Synthesis**: Merge insights from multiple sources
- **Contradiction Detection**: Identify and resolve conflicting information
- **Concept Evolution**: Track how ideas develop over time

### Universal File Support 🎉
- **ANY file type** can now be processed - no more file type restrictions!
- Intelligent text extraction with multiple fallback strategies
- Binary file string extraction for compiled code analysis
- Automatic encoding detection for international text files

### ⚡ High-Performance Async Processing
- **Concurrent Operations**: Process multiple documents simultaneously
- **Stream Processing**: Handle gigabyte files without memory issues
- **Real-time Progress**: Live updates during processing
- **Scalable Architecture**: Ready for enterprise workloads

### Enhanced Progress Tracking 📊
- Real-time progress bars for large file processing
- Live word count during summarization
- Streaming updates via Server-Sent Events
- Visual feedback for every stage of processing

### Clipboard Integration ⚡
- Global hotkey (Ctrl+Shift+T) for instant capture
- Auto-paste clipboard content
- Sub-100ms popup response time
- Beautiful capture UI with dark theme

### Improved File Processing
- Increased upload limit to 100MB
- Support for 50+ file formats out of the box
- Graceful fallbacks for unknown types
- Better handling of corrupted files

## Supported File Types

While SUM now accepts ANY file type, it has optimized support for:

**Documents**: PDF, DOCX, DOC, ODT, RTF, TXT, MD
**Code**: Python, JavaScript, Java, C/C++, Go, Rust, and 40+ more
**Data**: JSON, XML, CSV, Excel (XLS/XLSX)
**Web**: HTML, CSS, JavaScript, TypeScript
**Config**: YAML, TOML, INI, ENV
**And literally anything else** - if it has text, we'll extract it!

## Roadmap

- [x] Universal file type support
- [x] Real-time progress visualization  
- [x] Global clipboard integration
- [ ] Support for additional languages
- [ ] Custom fine-tuned models
- [ ] API key authentication
- [ ] Webhook notifications
- [ ] OCR for scanned documents

## Contributing

Contributions are welcome. Please ensure:
1. Code follows PEP 8 style guidelines
2. All tests pass
3. New features include tests
4. Documentation is updated

## License

Apache License 2.0 - See [LICENSE](LICENSE) file

## Support

- Issues: [GitHub Issues](https://github.com/OtotaO/SUM/issues)
- Discussions: [GitHub Discussions](https://github.com/OtotaO/SUM/discussions)

---

**SUM**: Enterprise-grade text summarization made simple.