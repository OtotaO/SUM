# SUM Deployment Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Development Setup](#development-setup)
3. [Production Deployment](#production-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Configuration](#configuration)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### Local Development
```bash
# Clone repository
git clone https://github.com/OtotaO/SUM.git
cd SUM

# Install dependencies
pip install -r requirements.txt

# Download NLTK resources
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Run the application
python main.py

# Access at http://localhost:3000
```

## Development Setup

### Prerequisites
- Python 3.10 or higher
- Node.js 16+ (for web interface development)
- Git
- 4GB RAM minimum
- 2GB free disk space

### Virtual Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_optimized.txt  # Additional optimizations
```

### Environment Variables
Create a `.env` file in the project root:
```env
# Core Settings
SUM_ENV=development
SUM_HOST=0.0.0.0
SUM_PORT=3000

# AI Configuration
OPENAI_API_KEY=your_api_key_here
SUM_AI_MODEL=gpt-3.5-turbo

# Logging
SUM_LOG_LEVEL=INFO

# Performance
SUM_CACHE_ENABLED=true
SUM_MAX_WORKERS=4
```

## Production Deployment

### System Requirements
- Ubuntu 20.04+ or similar Linux distribution
- Python 3.10+
- Nginx or Apache for reverse proxy
- PostgreSQL 13+ (optional, for notes storage)
- Redis (optional, for caching)
- 8GB RAM minimum
- 10GB free disk space

### Production Setup

1. **Create deployment user:**
```bash
sudo useradd -m -s /bin/bash sumuser
sudo usermod -aG sudo sumuser
```

2. **Install system dependencies:**
```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip nginx postgresql redis-server
sudo apt install -y build-essential libssl-dev libffi-dev python3-dev
```

3. **Clone and setup application:**
```bash
sudo su - sumuser
git clone https://github.com/OtotaO/SUM.git /opt/sum
cd /opt/sum

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn psycopg2-binary redis
```

4. **Create systemd service:**
```bash
sudo nano /etc/systemd/system/sum.service
```

```ini
[Unit]
Description=SUM Intelligence Amplification System
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=sumuser
Group=sumuser
WorkingDirectory=/opt/sum
Environment="PATH=/opt/sum/venv/bin"
Environment="SUM_ENV=production"
ExecStart=/opt/sum/venv/bin/gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:3000 main:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

5. **Configure Nginx:**
```bash
sudo nano /etc/nginx/sites-available/sum
```

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    client_max_body_size 50M;
    
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /ws {
        proxy_pass http://localhost:8765;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

6. **Enable and start services:**
```bash
sudo ln -s /etc/nginx/sites-available/sum /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable sum
sudo systemctl start sum
```

### SSL/TLS Setup (Let's Encrypt)
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## Docker Deployment

### Single Container
```bash
# Build image
docker build -t sum-app .

# Run container
docker run -d \
  --name sum \
  -p 3000:3000 \
  -p 8765:8765 \
  -e OPENAI_API_KEY=your_key \
  -e SUM_ENV=production \
  -v $(pwd)/data:/app/data \
  sum-app
```

### Docker Compose (Recommended)
```yaml
# docker-compose.yml
version: '3.8'

services:
  sum:
    build: .
    ports:
      - "3000:3000"
      - "8765:8765"
    environment:
      - SUM_ENV=production
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - POSTGRES_HOST=db
      - REDIS_HOST=redis
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - db
      - redis
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=sum_db
      - POSTGRES_USER=sum_user
      - POSTGRES_PASSWORD=secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - sum
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

Run with:
```bash
docker-compose up -d
```

## Cloud Deployment

### AWS EC2

1. **Launch EC2 instance:**
   - AMI: Ubuntu Server 20.04 LTS
   - Instance type: t3.large (minimum)
   - Storage: 20GB GP3
   - Security group: Allow ports 22, 80, 443

2. **Connect and deploy:**
```bash
ssh -i your-key.pem ubuntu@your-instance-ip
# Follow production deployment steps above
```

### Google Cloud Platform

1. **Create VM instance:**
```bash
gcloud compute instances create sum-server \
  --machine-type=e2-standard-2 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=20GB \
  --tags=http-server,https-server
```

2. **Configure firewall:**
```bash
gcloud compute firewall-rules create allow-sum \
  --allow tcp:80,tcp:443,tcp:3000,tcp:8765 \
  --source-ranges 0.0.0.0/0 \
  --target-tags http-server,https-server
```

### Kubernetes Deployment

```yaml
# sum-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sum-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sum
  template:
    metadata:
      labels:
        app: sum
    spec:
      containers:
      - name: sum
        image: yourdockerhub/sum:latest
        ports:
        - containerPort: 3000
        - containerPort: 8765
        env:
        - name: SUM_ENV
          value: "production"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: sum-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: sum-service
spec:
  selector:
    app: sum
  ports:
  - name: http
    port: 80
    targetPort: 3000
  - name: websocket
    port: 8765
    targetPort: 8765
  type: LoadBalancer
```

## Configuration

### Production Configuration File
Create `config.production.yaml`:
```yaml
ai:
  default_model: "gpt-3.5-turbo"
  max_tokens: 4000
  temperature: 0.7
  cache_enabled: true

server:
  host: "0.0.0.0"
  port: 3000
  websocket_port: 8765
  debug: false
  cors_enabled: true
  cors_origins: ["https://your-domain.com"]

processing:
  max_text_length: 500000
  chunk_size: 2000
  parallel_workers: 4
  enable_streaming: true

notes:
  storage_backend: "postgres"
  backup_enabled: true
  backup_interval: 86400

security:
  auth_enabled: true
  encryption_enabled: true
  api_key_required: true

logging:
  level: "WARNING"
  file_enabled: true
  file_path: "/var/log/sum/app.log"
  structured_logging: true

performance:
  cache_backend: "redis"
  cache_size: 10000
  optimize_memory: true
```

## Monitoring

### Health Check Endpoint
```bash
curl http://localhost:3000/api/health
```

### Prometheus Metrics
Add to `prometheus.yml`:
```yaml
scrape_configs:
  - job_name: 'sum'
    static_configs:
      - targets: ['localhost:3000']
    metrics_path: '/metrics'
```

### Logging
```bash
# View logs
tail -f /var/log/sum/app.log

# With systemd
journalctl -u sum -f

# Docker logs
docker logs -f sum
```

### Performance Monitoring
```python
# Add to your monitoring script
import requests

response = requests.get('http://localhost:3000/api/stats')
stats = response.json()

print(f"Active sessions: {stats['active_sessions']}")
print(f"Total processed: {stats['total_processed']}")
print(f"Average response time: {stats['avg_response_time_ms']}ms")
```

## Troubleshooting

### Common Issues

1. **Port already in use:**
```bash
# Find process using port
sudo lsof -i :3000
# Kill process
sudo kill -9 <PID>
```

2. **NLTK data not found:**
```python
python -c "import nltk; nltk.download('all')"
```

3. **Memory issues:**
```bash
# Increase swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

4. **WebSocket connection failed:**
- Check firewall rules
- Ensure Nginx is configured for WebSocket
- Verify WebSocket port (8765) is accessible

5. **Slow performance:**
- Enable Redis caching
- Increase worker processes
- Use production WSGI server (Gunicorn)
- Enable CDN for static assets

### Debug Mode
```bash
# Enable debug logging
export SUM_LOG_LEVEL=DEBUG
export SUM_ENV=development

# Run with verbose output
python main.py --debug
```

### Database Issues
```sql
-- Check PostgreSQL connections
SELECT count(*) FROM pg_stat_activity;

-- Clear old connections
SELECT pg_terminate_backend(pid) 
FROM pg_stat_activity 
WHERE datname = 'sum_db' AND pid <> pg_backend_pid();
```

## Backup and Recovery

### Automated Backups
```bash
# Create backup script
nano /opt/sum/backup.sh
```

```bash
#!/bin/bash
BACKUP_DIR="/backup/sum"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup database
pg_dump -U sum_user sum_db > $BACKUP_DIR/db_$DATE.sql

# Backup data files
tar -czf $BACKUP_DIR/data_$DATE.tar.gz /opt/sum/data

# Backup configuration
cp -r /opt/sum/config* $BACKUP_DIR/

# Keep only last 7 days
find $BACKUP_DIR -name "*.sql" -mtime +7 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
```

### Restore Process
```bash
# Restore database
psql -U sum_user sum_db < backup.sql

# Restore data files
tar -xzf data_backup.tar.gz -C /opt/sum/
```

## Security Best Practices

1. **Use environment variables for secrets**
2. **Enable HTTPS/TLS**
3. **Implement rate limiting**
4. **Regular security updates:**
```bash
sudo apt update && sudo apt upgrade
pip list --outdated
```

5. **Monitor logs for suspicious activity**
6. **Use firewall rules:**
```bash
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

## Support

For deployment assistance:
- GitHub Issues: https://github.com/OtotaO/SUM/issues
- Documentation: https://sum-ai.com/docs
- Community: https://discord.gg/sum-community