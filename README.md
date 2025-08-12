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
- **File Processing** - Automatic text extraction from PDF, DOCX, TXT, HTML, MD
- **Long Document Support** - Intelligent chunking for documents exceeding model limits
- **Streaming API** - Real-time progress updates via Server-Sent Events
- **Performance Optimization** - Redis caching with configurable TTL
- **Rate Limiting** - Protect API from abuse

### Technical Specifications
- Built on Hugging Face Transformers (BART model)
- RESTful API design
- Horizontal scaling support
- Docker-ready deployment
- Comprehensive error handling

## Quick Start

### Option 1: Minimal Setup (No Redis)
```bash
# Clone repository
git clone https://github.com/OtotaO/SUM.git
cd SUM

# Install dependencies
pip install flask transformers torch

# Run local version
python quickstart_local.py
```

### Option 2: Full Setup
```bash
# Install all dependencies
pip install -r requirements.txt

# Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# Run server
python sum_simple.py  # Basic API
# or
python sum_ultimate.py  # Full feature set
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
┌─────────────────────────────────────┐
│          REST API Layer              │
│    /summarize  /stream  /health      │
├─────────────────────────────────────┤
│       Summarization Engine           │
│  - Text preprocessing                │
│  - Chunk management                  │
│  - Model inference                   │
│  - Result aggregation                │
├─────────────────────────────────────┤
│         Caching Layer                │
│      Redis (optional)                │
└─────────────────────────────────────┘
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
REDIS_URL=redis://localhost:6379  # Redis connection string
MAX_TEXT_LENGTH=100000            # Maximum text length per request
RATE_LIMIT=60                     # Requests per minute per IP
CACHE_TTL=3600                    # Cache expiration in seconds
MODEL_NAME=facebook/bart-large-cnn # Transformer model to use
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

## Roadmap

- [ ] Support for additional languages
- [ ] Custom fine-tuned models
- [ ] Extractive summarization option
- [ ] API key authentication
- [ ] Webhook notifications
- [ ] Batch processing endpoints

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