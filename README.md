# SUM: Simple Unified Summarizer

> **Transform information into understanding. Instantly.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)

## What is SUM?

SUM is an AI-powered text summarization platform that transforms long content into clear, actionable insights. It started as a simple summarizer and evolved into a comprehensive intelligence platform.

## Current Status

🚧 **Active Development** - We're currently simplifying the architecture from 50,000+ lines to under 1,000 lines while maintaining core functionality.

## Features

### Available Now
- ✅ **Text Summarization** - State-of-the-art transformer models
- ✅ **Multiple Algorithms** - LDA, NMF, LSA topic modeling
- ✅ **REST API** - Simple HTTP endpoints
- ✅ **Caching** - Redis-powered performance
- ✅ **Web Interface** - Clean, responsive UI

### In Development
- 🔄 **Simplified Architecture** - Reducing complexity by 98%
- 🔄 **Pattern Recognition** - Smart content analysis
- 🔄 **User Memory** - Historical summary tracking
- 🔄 **Performance Optimization** - 10x speed improvements

## Quick Start

### Option 1: Current Version
```bash
# Clone the repository
git clone https://github.com/OtotaO/SUM.git
cd SUM

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py

# Access at http://localhost:3000
```

### Option 2: Simplified Version (Beta)
```bash
# Install minimal dependencies
pip install flask redis transformers torch

# Run the simplified version
python sum_simple.py

# Test it
curl -X POST localhost:3000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here..."}'
```

## Project Structure

```
SUM/
├── main.py              # Current entry point
├── sum_simple.py        # Simplified version (in development)
├── Models/              # ML models and algorithms
├── Utils/               # Utility functions
├── api/                 # API endpoints
├── static/              # Web interface files
└── Tests/               # Test suite
```

## API Usage

### Basic Summarization
```bash
POST /summarize
Content-Type: application/json

{
  "text": "Your long text here...",
  "max_length": 150
}
```

### Response
```json
{
  "summary": "Concise summary of your text",
  "keywords": ["key", "words"],
  "topics": ["main", "topics"]
}
```

## Performance

Current version:
- Handles texts up to 100,000 characters
- Average response time: 1-2 seconds
- Supports concurrent requests

Simplified version (in testing):
- 10x faster response times
- 60% less memory usage
- Cleaner, more maintainable code

## Development Roadmap

### Phase 1: Simplification (Current)
- Reduce codebase from 50k to 1k lines
- Maintain backward compatibility
- Improve performance

### Phase 2: Enhancement
- Add smart pattern recognition
- Implement user memory features
- Create better documentation

### Phase 3: Polish
- Production-ready deployment
- Comprehensive testing
- Performance optimization

## Contributing

We welcome contributions! The project is undergoing major refactoring to improve code quality and performance.

### Guidelines
1. Keep it simple
2. Measure performance impact
3. Write tests
4. Document your changes

## Requirements

- Python 3.8+
- Redis (for caching)
- 4GB RAM minimum
- Optional: GPU for faster processing

## Installation

### Basic Setup
```bash
pip install -r requirements.txt
```

### With Docker
```bash
docker-compose up
```

## Testing

```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_sum.py
```

## Architecture

The project is transitioning from a complex multi-layer architecture to a simple, efficient design:

**Current**: Web → API → Application → Domain → Models → Utils (6 layers)

**Target**: API → Core → Storage (3 layers)

## License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Flask, Transformers, and Redis
- Inspired by the principle: "Simplicity is the ultimate sophistication"

## Support

- 📖 [Documentation](docs/) (Being updated)
- 🐛 [Issue Tracker](https://github.com/OtotaO/SUM/issues)
- 💬 [Discussions](https://github.com/OtotaO/SUM/discussions)

---

<p align="center">
<strong>SUM: Making information accessible</strong><br>
<em>Currently being simplified for better performance and maintainability</em>
</p>