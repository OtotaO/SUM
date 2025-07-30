# 🐳 Docker Deployment Guide for SUM

This guide provides one-command deployment for the SUM Hierarchical Knowledge Densification System using Docker.

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)
```bash
# Clone and start with one command
git clone https://github.com/OtotaO/SUM.git
cd SUM
docker-compose up -d
```

### Option 2: Docker Build
```bash
# Build and run manually
docker build -t sum-engine .
docker run -d -p 3000:3000 -p 8765:8765 --name sum-container sum-engine
```

## 🌟 What You Get

After running `docker-compose up -d`, you'll have:

- **🌐 Flask API Server**: `http://localhost:3000`
- **⚡ WebSocket Server**: `ws://localhost:8765`
- **📊 Health Check**: `http://localhost:3000/api/health`
- **📖 API Documentation**: `http://localhost:3000/api/progressive_summarization`
- **🎨 Progressive Interface**: Auto-generated `progressive_client.html`

## 🛠️ Available Services

| Service | Port | Description |
|---------|------|-------------|
| **Main API** | 3000 | REST API with all SUM endpoints |
| **Progressive WebSocket** | 8765 | Real-time summarization with live progress |
| **Health Check** | 3000/api/health | Service status monitoring |

## 📋 Container Features

### ✅ **Pre-installed Components**
- Python 3.10 with all dependencies
- NLTK resources (punkt, stopwords, vader, etc.)
- WebSocket support (websockets, psutil)
- All SUM engines (Simple, Advanced, Hierarchical, Streaming)
- Progressive Summarization system

### 🔧 **Auto-Configuration**
- Optimized for production deployment
- Health checks for service monitoring
- Proper logging and error handling
- Memory-efficient processing
- Clean shutdown handling

## 🎯 Usage Examples

### Test the API
```bash
# Check if services are running
curl http://localhost:3000/api/health

# Process text with hierarchical engine
curl -X POST http://localhost:3000/api/process_text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text here...",
    "model": "hierarchical",
    "config": {
      "max_concepts": 7,
      "max_summary_tokens": 50,
      "max_insights": 3
    }
  }'

# Get progressive summarization info
curl http://localhost:3000/api/progressive_summarization
```

### WebSocket Connection
```javascript
// Connect to progressive summarization
const ws = new WebSocket('ws://localhost:8765');

ws.onopen = () => {
    ws.send(JSON.stringify({
        type: 'start_processing',
        text: 'Your long text for progressive processing...',
        config: {
            chunk_size_words: 1000,
            overlap_ratio: 0.15
        }
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Progress update:', data);
};
```

## 🔧 Configuration

### Environment Variables
```bash
# Override default settings
docker run -d \
  -e FLASK_PORT=5000 \
  -e LOG_LEVEL=DEBUG \
  -e SUM_MAX_SUMMARY_LENGTH=300 \
  -p 5000:5000 -p 8765:8765 \
  sum-engine
```

### Volume Mounts
```bash
# Persist data and outputs
docker run -d \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/outputs:/app/Output \
  -v $(pwd)/data:/app/Data \
  -p 3000:3000 -p 8765:8765 \
  sum-engine
```

## 🚀 Production Deployment

### With Nginx Reverse Proxy
```bash
# Start with production profile
docker-compose --profile production up -d
```

This includes:
- Nginx reverse proxy for load balancing
- SSL certificate support
- Production-optimized settings
- Health monitoring

### Docker Swarm
```yaml
# docker-stack.yml
version: '3.8'
services:
  sum-api:
    image: sum-engine:latest
    ports:
      - "3000:3000"
      - "8765:8765"
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
```

Deploy with:
```bash
docker stack deploy -c docker-stack.yml sum-stack
```

## 📊 Monitoring

### Health Checks
```bash
# Check container health
docker ps
docker logs sum-container

# Health endpoint
curl http://localhost:3000/api/health
```

### Resource Usage
```bash
# Monitor resource usage
docker stats sum-container

# Container logs
docker logs -f sum-container
```

## 🛠️ Development

### Development Mode
```bash
# Run with live code reload
docker run -d \
  -v $(pwd):/app \
  -e FLASK_ENV=development \
  -p 3000:3000 -p 8765:8765 \
  sum-engine
```

### Debug Mode
```bash
# Access container shell
docker exec -it sum-container bash

# View logs
docker logs sum-container

# Restart services
docker restart sum-container
```

## 🔧 Troubleshooting

### Common Issues

**Port conflicts:**
```bash
# Use different ports
docker run -d -p 4000:3000 -p 9765:8765 sum-engine
```

**Memory issues:**
```bash
# Increase memory limit
docker run -d --memory=2g sum-engine
```

**NLTK resources:**
```bash
# Check NLTK installation
docker exec sum-container python -c "import nltk; print(nltk.data.path)"
```

### Logs
```bash
# Container logs
docker logs sum-container

# Service-specific logs
docker exec sum-container tail -f /app/sum_service.log
```

## 🎉 Success!

Your SUM Hierarchical Knowledge Densification System is now running in Docker with:

- ✅ One-command deployment
- ✅ Full feature set (Hierarchical, Progressive, Streaming)
- ✅ Production-ready configuration
- ✅ Health monitoring
- ✅ Auto-restart capabilities
- ✅ Beautiful WebSocket interface

Visit `http://localhost:3000` to start processing text with the world's most advanced hierarchical summarization system!