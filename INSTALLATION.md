# 🚀 SUM Installation Guide

Complete installation guide for the SUM Hierarchical Knowledge Densification System with Real-Time Progressive Summarization.

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [System Requirements](#-system-requirements)  
- [Installation Methods](#-installation-methods)
- [Verification](#-verification)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Advanced Setup](#-advanced-setup)

---

## ⚡ Quick Start

### Option 1: Docker (Recommended)
```bash
# One-command installation and startup
git clone https://github.com/OtotaO/SUM.git
cd SUM
docker-compose up -d

# Access the services
# 🌐 Main API: http://localhost:3000
# ⚡ Progressive Summarization: ws://localhost:8765
```

### Option 2: Python Installation
```bash
# Clone repository
git clone https://github.com/OtotaO/SUM.git
cd SUM

# Install dependencies
pip install -r requirements.txt

# Download NLTK resources
python examples/download_nltk_resources.py

# Start services
python main.py
```

---

## 💻 System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher (3.10+ recommended)
- **RAM**: 2GB free memory
- **Storage**: 1GB free space
- **Network**: Internet connection for NLTK downloads

### Recommended Specifications
- **Python**: 3.10 or 3.11
- **RAM**: 4GB+ free memory
- **Storage**: 2GB+ free space
- **CPU**: Multi-core processor for parallel processing

### Supported Platforms
- ✅ **Linux** (Ubuntu 18.04+, CentOS 7+, Debian 9+)
- ✅ **macOS** (10.15+, both Intel and Apple Silicon)
- ✅ **Windows** (10/11, WSL2 recommended)
- ✅ **Docker** (All platforms with Docker support)

---

## 🛠️ Installation Methods

### Method 1: Docker Installation (Recommended)

Docker provides the most reliable installation with all dependencies pre-configured.

#### Prerequisites
```bash
# Install Docker and Docker Compose
# Ubuntu/Debian:
sudo apt update
sudo apt install docker.io docker-compose

# macOS (using Homebrew):
brew install docker docker-compose

# Windows: Download Docker Desktop from docker.com
```

#### Installation Steps
```bash
# 1. Clone repository
git clone https://github.com/OtotaO/SUM.git
cd SUM

# 2. Start services with Docker Compose
docker-compose up -d

# 3. Verify installation
curl http://localhost:3000/api/health

# 4. Test progressive summarization
curl http://localhost:3000/api/progressive_summarization
```

#### What You Get
- ✅ Flask API server on port 3000
- ✅ WebSocket server on port 8765  
- ✅ All NLTK resources pre-downloaded
- ✅ All dependencies installed
- ✅ Health monitoring enabled
- ✅ Auto-restart on failure

### Method 2: Python Virtual Environment

#### Step 1: Setup Python Environment
```bash
# Create virtual environment
python3 -m venv sum_env

# Activate environment
# Linux/macOS:
source sum_env/bin/activate
# Windows:
sum_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

#### Step 2: Clone and Install
```bash
# Clone repository
git clone https://github.com/OtotaO/SUM.git
cd SUM

# Install Python dependencies
pip install -r requirements.txt

# Install additional dependencies for WebSocket support
pip install websockets psutil

# Install Rich for beautiful CLI output
pip install rich
```

#### Step 3: Download NLTK Resources
```bash
# Method 1: Use provided script (recommended)
python examples/download_nltk_resources.py

# Method 2: Manual download
python -c "
import nltk
nltk.download('punkt')
nltk.download('punkt_tab') 
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
print('✅ NLTK resources downloaded')
"
```

#### Step 4: Start Services
```bash
# Start Flask API server
python main.py &

# Start Progressive Summarization WebSocket server
python progressive_summarization.py &

# Or use the enhanced CLI
python cli_enhanced.py
```

### Method 3: Conda Installation

```bash
# Create conda environment
conda create -n sum_env python=3.10
conda activate sum_env

# Clone and install
git clone https://github.com/OtotaO/SUM.git
cd SUM

# Install dependencies
pip install -r requirements.txt
python examples/download_nltk_resources.py

# Start services
python main.py
```

### Method 4: System-Wide Installation (Not Recommended)

```bash
# Install globally (may conflict with other packages)
git clone https://github.com/OtotaO/SUM.git
cd SUM
sudo pip install -r requirements.txt
python examples/download_nltk_resources.py
```

---

## ✅ Verification

### Test Core Functionality
```bash
# Test basic import
python -c "from SUM import HierarchicalDensificationEngine; print('✅ Import successful')"

# Test hierarchical processing
python -c "
from SUM import HierarchicalDensificationEngine
engine = HierarchicalDensificationEngine()
result = engine.process_text('Test text for verification.')
print('✅ Hierarchical engine working')
print('Concepts:', result['hierarchical_summary']['level_1_concepts'])
"

# Test streaming engine
python -c "
from StreamingEngine import StreamingHierarchicalEngine
engine = StreamingHierarchicalEngine()
result = engine.process_streaming_text('Test streaming functionality.')
print('✅ Streaming engine working')
print('Chunks processed:', result['streaming_metadata']['chunks_processed'])
"
```

### Test Web Services
```bash
# Test Flask API
curl -X GET http://localhost:3000/api/health
curl -X GET http://localhost:3000/api/config

# Test text processing
curl -X POST http://localhost:3000/api/process_text \
  -H "Content-Type: application/json" \
  -d '{"text": "Test text processing", "model": "hierarchical"}'

# Test progressive summarization info
curl -X GET http://localhost:3000/api/progressive_summarization
```

### Test WebSocket Connection
```bash
# Install wscat for WebSocket testing
npm install -g wscat

# Test WebSocket connection
wscat -c ws://localhost:8765

# Send test message (after connection)
{"type": "ping"}
```

### Run Test Suite
```bash
# Run comprehensive tests
python Tests/test_sum.py

# Run benchmarks
python benchmark.py

# Test enhanced CLI
python cli_enhanced.py --help
```

---

## ⚙️ Configuration

### Environment Variables
Create a `.env` file in the project root:

```bash
# Flask Configuration
FLASK_ENV=production
FLASK_HOST=0.0.0.0
FLASK_PORT=3000
FLASK_SECRET_KEY=your-secret-key-here

# SUM Configuration
SUM_NUM_TOPICS=5
SUM_MAX_SUMMARY_LENGTH=200
SUM_DEFAULT_THRESHOLD=0.3
SUM_MAX_WORKERS=4

# Logging
LOG_LEVEL=INFO
LOG_FILE=sum_service.log

# WebSocket Configuration
WEBSOCKET_HOST=localhost
WEBSOCKET_PORT=8765
```

### Custom Configuration File
Create `config.json`:

```json
{
  "num_topics": 5,
  "max_summary_length": 200,
  "min_summary_length": 50,
  "default_threshold": 0.3,
  "batch_size": 100,
  "max_workers": 4,
  "allowed_extensions": ["txt", "json", "csv", "md"]
}
```

### NLTK Data Directory
Set custom NLTK data path:

```bash
export NLTK_DATA=/path/to/your/nltk_data
# or
python -c "import nltk; nltk.data.path.append('/path/to/your/nltk_data')"
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'SUM'
# Solution: Ensure you're in the correct directory and Python path is set
export PYTHONPATH="${PYTHONPATH}:/path/to/SUM"
```

#### 2. NLTK Resource Errors
```bash
# Error: Resource punkt not found
# Solution: Download NLTK resources
python examples/download_nltk_resources.py

# Or manually:
python -c "import nltk; nltk.download('punkt', quiet=True)"
```

#### 3. Memory Issues
```bash
# Error: Memory error during processing
# Solution: Reduce chunk size or use streaming engine
python -c "
from StreamingEngine import StreamingHierarchicalEngine, StreamingConfig
config = StreamingConfig(chunk_size_words=500, max_memory_mb=256)
engine = StreamingHierarchicalEngine(config)
"
```

#### 4. Port Conflicts
```bash
# Error: Port 3000 already in use
# Solution: Use different port
FLASK_PORT=5000 python main.py

# Or kill existing process
sudo lsof -ti:3000 | xargs kill -9
```

#### 5. WebSocket Connection Issues
```bash
# Error: WebSocket connection failed
# Solution: Check if server is running and port is open
python progressive_summarization.py &
netstat -an | grep 8765
```

#### 6. Docker Issues
```bash
# Error: Docker build fails
# Solution: Clean Docker cache and rebuild
docker system prune -a
docker-compose build --no-cache
docker-compose up -d
```

#### 7. Permission Errors
```bash
# Error: Permission denied
# Solution: Fix file permissions
chmod +x cli_enhanced.py
chmod +x benchmark.py

# Or run with appropriate permissions
sudo python main.py
```

### Platform-Specific Issues

#### Windows
```powershell
# PowerShell execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Use Windows Subsystem for Linux (WSL2) for best compatibility
wsl --install
```

#### macOS
```bash
# Install Xcode command line tools if needed
xcode-select --install

# Use Homebrew for Python installation
brew install python@3.10
```

#### Linux
```bash
# Install build essentials
sudo apt update
sudo apt install build-essential python3-dev

# For CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel
```

### Debug Mode
Enable debug mode for detailed error messages:

```bash
# Environment variable
export FLASK_ENV=development
export LOG_LEVEL=DEBUG

# Or in code
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization
```bash
# Run performance benchmarks
python benchmark.py

# Monitor resource usage
python -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}%')
"
```

---

## 🚀 Advanced Setup

### Production Deployment

#### 1. Use Production WSGI Server
```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:3000 main:app
```

#### 2. Nginx Reverse Proxy
Create `/etc/nginx/sites-available/sum`:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /ws {
        proxy_pass http://127.0.0.1:8765;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

#### 3. SSL/HTTPS Setup
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com
```

#### 4. Process Management with Systemd
Create `/etc/systemd/system/sum.service`:

```ini
[Unit]
Description=SUM Hierarchical Knowledge Densification System
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/SUM
Environment=FLASK_ENV=production
ExecStart=/path/to/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable sum.service
sudo systemctl start sum.service
```

### Monitoring and Logging

#### 1. Setup Log Rotation
Create `/etc/logrotate.d/sum`:

```
/path/to/SUM/*.log {
    daily
    missingok
    rotate 30
    compress
    notifempty
    create 644 www-data www-data
}
```

#### 2. Health Monitoring
```bash
# Create health check script
cat > health_check.sh << 'EOF'
#!/bin/bash
curl -f http://localhost:3000/api/health || exit 1
EOF

chmod +x health_check.sh

# Add to crontab
echo "*/5 * * * * /path/to/health_check.sh" | crontab -
```

### Scaling and Load Balancing

#### 1. Multiple Instances
```bash
# Start multiple Flask instances
FLASK_PORT=3001 python main.py &
FLASK_PORT=3002 python main.py &
FLASK_PORT=3003 python main.py &
```

#### 2. Load Balancer Configuration
```nginx
upstream sum_backend {
    server 127.0.0.1:3001;
    server 127.0.0.1:3002;
    server 127.0.0.1:3003;
}

server {
    location / {
        proxy_pass http://sum_backend;
    }
}
```

### Database Integration
```python
# Optional: Add database for result storage
pip install SQLAlchemy psycopg2-binary

# Example configuration
SQLALCHEMY_DATABASE_URI = 'postgresql://user:pass@localhost/sum_db'
```

---

## 🎉 Success!

Your SUM Hierarchical Knowledge Densification System is now installed and ready to revolutionize your text processing workflow!

### Next Steps:
1. **🌐 Web Interface**: Visit `http://localhost:3000`
2. **⚡ Progressive Summarization**: Open `progressive_client.html` in your browser
3. **💻 CLI Interface**: Run `python cli_enhanced.py` for interactive processing
4. **📊 Benchmarks**: Run `python benchmark.py` to test performance
5. **📖 Documentation**: Check the README.md for detailed usage examples

### Get Support:
- 📚 **Documentation**: [GitHub Repository](https://github.com/OtotaO/SUM)
- 🐛 **Issues**: Report bugs and request features on GitHub
- 💡 **Discussions**: Join the community discussions

Welcome to the future of hierarchical knowledge processing! 🚀✨