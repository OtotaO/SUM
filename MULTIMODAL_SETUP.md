# 🚀 Multi-Modal SUM Setup Guide

This guide will help you set up the enhanced SUM platform with multi-modal processing capabilities, local AI integration, and advanced document handling.

## 🎯 New Capabilities

The enhanced SUM platform now includes:

- **📄 Document Processing**: PDF, DOCX, HTML, Markdown support
- **🖼️ Image Analysis**: OCR text extraction + vision-language models
- **🤖 Local AI**: Privacy-focused processing with Ollama
- **⚡ Vision Models**: Analyze images, diagrams, and document layouts
- **🔄 Batch Processing**: Handle multiple files simultaneously
- **📊 Performance Monitoring**: Track processing statistics and model performance

## 🛠️ Installation Steps

### 1. Install System Dependencies

#### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Tesseract for OCR
brew install tesseract tesseract-lang

# Install Ollama for local AI
brew install ollama
```

#### Ubuntu/Debian
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Tesseract for OCR
sudo apt install tesseract-ocr tesseract-ocr-eng -y

# Install additional dependencies
sudo apt install python3-dev python3-pip build-essential -y

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Windows
```powershell
# Install Tesseract (download from GitHub releases)
# https://github.com/UB-Mannheim/tesseract/wiki

# Install Ollama (download from official site)
# https://ollama.ai/download

# Or use Chocolatey
choco install tesseract ollama
```

### 2. Install Python Dependencies

```bash
# Navigate to SUM directory
cd /path/to/SUM

# Install enhanced requirements
pip install -r requirements.txt

# For development/testing
pip install pytest pytest-cov black isort flake8
```

### 3. Setup Ollama and Local Models

```bash
# Start Ollama service
ollama serve

# In another terminal, pull recommended models
ollama pull llama3.2:1b        # Ultra-fast, 1B parameters
ollama pull llama3.2:3b        # Balanced performance, 3B parameters
ollama pull llava:7b           # Vision + text analysis
ollama pull codellama:7b       # Code analysis (optional)

# Verify installation
ollama list
```

### 4. Configure Environment

Create a `.env` file in your SUM directory:

```bash
# Basic Configuration
FLASK_PORT=5000
DEBUG=True
LOG_LEVEL=INFO

# File Upload Settings
MAX_CONTENT_LENGTH=50000000  # 50MB
UPLOADS_DIR=uploads

# Multi-Modal Settings
ENABLE_MULTIMODAL=True
ENABLE_OCR=True
ENABLE_VISION_MODELS=True

# Ollama Settings
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
DEFAULT_MODEL=llama3.2:3b

# Performance Settings
MAX_CONCURRENT_REQUESTS=5
CACHE_ENABLED=True
```

## 🚀 Quick Start

### 1. Test Installation

```bash
# Run comprehensive tests
python test_multimodal_system.py

# Test individual components
python multimodal_processor.py
python ollama_manager.py
```

### 2. Start the Enhanced Server

```bash
# Start with enhanced multi-modal support
python main_enhanced.py

# Or use the original main.py (basic functionality)
python main.py
```

### 3. Access the Interface

Open your browser and navigate to:
- **Main Interface**: http://localhost:5000
- **API Documentation**: http://localhost:5000/api/system/status
- **Processing Stats**: http://localhost:5000/api/stats

## 📝 Usage Examples

### Text Processing with Local AI

```python
import requests

# Process text with local AI enhancement
response = requests.post('http://localhost:5000/api/process/text', 
    json={
        'text': 'Your text here...',
        'use_local_ai': True,
        'model': 'llama3.2:3b',
        'config': {
            'max_concepts': 7,
            'max_summary_tokens': 100,
            'task_type': 'summarization'
        }
    }
)

result = response.json()
print("Hierarchical Summary:", result['hierarchical_summary'])
print("Local AI Analysis:", result.get('local_ai_analysis'))
```

### Multi-Modal File Processing

```python
# Process PDF with vision enhancement
files = {'file': open('document.pdf', 'rb')}
data = {
    'use_local_ai': 'true',
    'use_vision': 'true',
    'hierarchical_config': '{"max_concepts": 10}'
}

response = requests.post(
    'http://localhost:5000/api/process/file',
    files=files,
    data=data
)

result = response.json()
print("Content Type:", result['content_type'])
print("Extracted Text:", result['extracted_text'][:500])
print("Confidence:", result['confidence_score'])
```

### Batch Processing

```python
# Process multiple files at once
files = [
    ('files', open('doc1.pdf', 'rb')),
    ('files', open('doc2.docx', 'rb')),
    ('files', open('image.png', 'rb'))
]

response = requests.post(
    'http://localhost:5000/api/process/batch',
    files=files,
    data={'use_local_ai': 'true'}
)

results = response.json()
for file_result in results['results']:
    print(f"{file_result['filename']}: {file_result['content_type']}")
```

## 🔧 Configuration Options

### Multi-Modal Processor Settings

```python
config = {
    'max_concepts': 10,           # Number of key concepts to extract
    'max_summary_tokens': 200,    # Maximum summary length
    'complexity_threshold': 0.7,  # Threshold for expansion
    'max_insights': 5,            # Maximum insights to extract
    'use_vision': True,           # Enable vision model analysis
    'ocr_confidence': 0.6         # Minimum OCR confidence threshold
}
```

### Ollama Model Settings

```python
ollama_config = {
    'default_model': 'llama3.2:3b',
    'prefer_speed': False,        # Prefer quality over speed
    'max_tokens': 500,
    'temperature': 0.3,
    'timeout': 30                 # seconds
}
```

## 📊 Monitoring and Performance

### System Status

```bash
# Check system capabilities
curl http://localhost:5000/api/system/status

# View processing statistics
curl http://localhost:5000/api/stats

# List available local models
curl http://localhost:5000/api/models/local

# Benchmark model performance
curl -X POST http://localhost:5000/api/models/benchmark \
  -H "Content-Type: application/json" \
  -d '{"sample_text": "Test text for benchmarking"}'
```

### Performance Optimization

1. **Model Selection**:
   - Use `llama3.2:1b` for fast processing
   - Use `llama3.2:3b` for balanced performance
   - Use `llama3.1:8b` for highest quality

2. **Memory Management**:
   - Monitor RAM usage with large documents
   - Use batch processing for multiple files
   - Clear cache periodically: `POST /api/cache/clear`

3. **Processing Optimization**:
   - Enable caching for repeated content
   - Use appropriate confidence thresholds
   - Batch similar document types together

## 🔍 Troubleshooting

### Common Issues

#### Ollama Connection Failed
```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama service
ollama serve

# Test connection
ollama list
```

#### OCR Not Working
```bash
# Verify Tesseract installation
tesseract --version

# Check language packs
tesseract --list-langs

# Install additional languages
sudo apt install tesseract-ocr-fra  # French
sudo apt install tesseract-ocr-deu  # German
```

#### Memory Issues
```bash
# Monitor memory usage
python -c "
import psutil
print(f'RAM: {psutil.virtual_memory().percent}%')
print(f'Available: {psutil.virtual_memory().available / 1024**3:.1f}GB')
"

# Clear Python cache
python -c "import gc; gc.collect()"
```

### Error Diagnosis

1. **Check Logs**: Look in `sum_service.log` for detailed error messages
2. **Test Components**: Run individual test files to isolate issues
3. **Verify Dependencies**: Ensure all required packages are installed
4. **Check Permissions**: Verify file system permissions for uploads
5. **Monitor Resources**: Watch CPU, memory, and disk usage

## 🎯 Advanced Features

### Custom Model Integration

```python
# Add custom Ollama models
from ollama_manager import OllamaManager

manager = OllamaManager()

# Install custom models
manager.install_recommended_models(['your-custom-model:latest'])

# Benchmark performance
results = manager.benchmark_models()
print(results)
```

### Vision Model Enhancement

```python
# Custom vision processing
from multimodal_processor import MultiModalProcessor

processor = MultiModalProcessor()

# Process image with custom settings
result = processor.process_file(
    'complex_diagram.png',
    use_vision=True,
    hierarchical_config={'max_insights': 10}
)

print("Vision Analysis:", result.metadata.get('vision_analysis'))
```

## 🚦 Production Deployment

### Docker Setup (Optional)

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    build-essential \
    curl

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Setup application
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Expose port
EXPOSE 5000

# Start services
CMD ["python", "main_enhanced.py"]
```

### Security Considerations

1. **File Upload Limits**: Configure appropriate size limits
2. **Input Validation**: Sanitize all user inputs
3. **API Rate Limiting**: Implement request throttling
4. **Secure Headers**: Add security headers to responses
5. **Local Processing**: Keep sensitive data local with Ollama

## 📚 Next Steps

1. **Explore Examples**: Check the `examples/` directory for usage patterns
2. **Customize Models**: Train domain-specific models with your data
3. **Scale Horizontally**: Deploy multiple instances with load balancing
4. **Monitor Performance**: Set up comprehensive logging and metrics
5. **Contribute**: Help improve the platform with new features

## 🆘 Support

- **Documentation**: Check existing documentation files
- **Issues**: Report bugs via GitHub issues
- **Community**: Join discussions and share improvements
- **Performance**: Monitor and optimize for your specific use case

---

**🎉 Congratulations!** You now have a powerful multi-modal AI processing platform with local model support. The system can handle text, documents, images, and more while keeping your data private and secure.

Start processing your documents and explore the advanced capabilities! 🚀