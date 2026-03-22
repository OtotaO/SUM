# SUM Platform Production Deployment Guide

## 🚀 Overview

This guide covers deploying the SUM platform in a production environment with high availability, scalability, and robustness.

## 📋 Prerequisites

- Linux server (Ubuntu 20.04+ or CentOS 8+)
- Docker and Docker Compose
- Domain name with SSL certificate
- At least 4GB RAM and 2 CPU cores
- 50GB+ storage for data and logs

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Nginx     │────▶│  Gunicorn   │────▶│     SUM     │
│  (Reverse   │     │  (WSGI      │     │ Application │
│   Proxy)    │     │  Server)    │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
                            │                    │
                            ▼                    ▼
                    ┌─────────────┐     ┌─────────────┐
                    │    Redis    │     │ PostgreSQL  │
                    │  (Cache &   │     │ (Database)  │
                    │  Sessions)  │     │             │
                    └─────────────┘     └─────────────┘
```

## 🔧 Robustness Features Implemented

### 1. **File Handling**
- ✅ Streaming uploads/downloads (no memory exhaustion)
- ✅ File type validation with magic numbers
- ✅ Size limits per file type
- ✅ Virus scanning hooks
- ✅ Duplicate detection via SHA-256

### 2. **Request Management**
- ✅ Request queue with priority handling
- ✅ Concurrent request limits
- ✅ Resource monitoring (CPU/memory)
- ✅ Graceful degradation
- ✅ Circuit breakers for external services

### 3. **Error Recovery**
- ✅ Automatic retry with exponential backoff
- ✅ Error tracking and statistics
- ✅ Recovery strategies per error type
- ✅ Graceful fallbacks
- ✅ Detailed error context logging

### 4. **Database**
- ✅ Connection pooling
- ✅ Query optimization
- ✅ Transaction management
- ✅ Automatic reconnection
- ✅ Migration ready (SQLite → PostgreSQL)

### 5. **Monitoring**
- ✅ Health check endpoints
- ✅ Prometheus metrics
- ✅ Structured JSON logging
- ✅ Request tracing
- ✅ Performance metrics

## 📦 Quick Start with Docker

### 1. Clone and Configure

```bash
git clone https://github.com/yourusername/SUM.git
cd SUM

# Create environment file
cat > .env << EOF
DB_PASSWORD=your-secure-password
SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
GRAFANA_PASSWORD=your-grafana-password
WORKERS=4
EOF
```

### 2. Build and Deploy

```bash
cd deployment
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f sum-app
```

### 3. Initialize Database

```bash
# Run migrations
docker-compose exec sum-app python -c "from app import db; db.create_all()"

# Create indexes
docker-compose exec postgres psql -U sum_user -d sum_platform < init.sql
```

## 🛠️ Manual Deployment

### 1. System Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3.11 python3.11-venv python3-pip \
    postgresql postgresql-contrib redis-server nginx \
    libmagic1 libpq-dev build-essential

# Create application user
sudo useradd -m -s /bin/bash sum
sudo usermod -aG www-data sum
```

### 2. Application Setup

```bash
# Switch to sum user
sudo su - sum

# Clone repository
git clone https://github.com/yourusername/SUM.git
cd SUM

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-production.txt

# Set permissions
chmod +x production.py
mkdir -p uploads logs data/temp
```

### 3. Database Configuration

```bash
# PostgreSQL setup
sudo -u postgres createuser sum_user
sudo -u postgres createdb sum_platform -O sum_user
sudo -u postgres psql -c "ALTER USER sum_user PASSWORD 'your-password';"

# Run migrations
python manage.py db upgrade
```

### 4. Gunicorn Service

Create `/etc/systemd/system/sum.service`:

```ini
[Unit]
Description=SUM Knowledge Platform
After=network.target postgresql.service redis.service

[Service]
Type=notify
User=sum
Group=www-data
WorkingDirectory=/home/sum/SUM
Environment="PATH=/home/sum/SUM/venv/bin"
Environment="FLASK_ENV=production"
ExecStart=/home/sum/SUM/venv/bin/gunicorn \
    --config deployment/gunicorn_config.py \
    --worker-tmp-dir /dev/shm \
    production:application

Restart=always
RestartSec=10

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/home/sum/SUM/uploads /home/sum/SUM/logs /home/sum/SUM/data

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable sum
sudo systemctl start sum
sudo systemctl status sum
```

### 5. Nginx Configuration

```bash
# Copy nginx config
sudo cp deployment/nginx.conf /etc/nginx/sites-available/sum
sudo ln -s /etc/nginx/sites-available/sum /etc/nginx/sites-enabled/

# Test and reload
sudo nginx -t
sudo systemctl reload nginx
```

## 🔐 Security Hardening

### 1. Firewall Rules

```bash
# UFW setup
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### 2. SSL/TLS Setup

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d sum.example.com
```

### 3. Application Security

```bash
# Set secure permissions
chmod 600 .env
chmod 700 uploads/
chmod 700 data/

# Enable SELinux/AppArmor
sudo apt install apparmor-utils
sudo aa-enforce /etc/apparmor.d/usr.bin.python3.11
```

## 📊 Monitoring Setup

### 1. Prometheus Configuration

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'sum-platform'
    static_configs:
      - targets: ['sum-app:5001']
    metrics_path: '/metrics'
```

### 2. Grafana Dashboard

Import dashboard from `deployment/grafana/dashboards/sum-dashboard.json`

### 3. Alerts

Configure alerts for:
- High memory usage (>80%)
- High CPU usage (>85%)
- Request queue backup (>50 pending)
- Error rate (>5% of requests)
- Database connection pool exhaustion

## 🔄 Scaling Strategies

### Horizontal Scaling

```yaml
# docker-compose.yml modification
sum-app:
  deploy:
    replicas: 3
```

### Load Balancing

Update nginx upstream:

```nginx
upstream sum_backend {
    least_conn;
    server sum-app-1:5001 max_fails=3 fail_timeout=30s;
    server sum-app-2:5001 max_fails=3 fail_timeout=30s;
    server sum-app-3:5001 max_fails=3 fail_timeout=30s;
}
```

## 🚨 Troubleshooting

### Common Issues

1. **Memory Issues**
   ```bash
   # Check memory usage
   docker stats
   
   # Increase container limits
   docker-compose down
   # Edit docker-compose.yml memory limits
   docker-compose up -d
   ```

2. **Database Connection Errors**
   ```bash
   # Check connection pool
   curl http://localhost:5001/health/ready
   
   # Restart with fresh connections
   docker-compose restart sum-app
   ```

3. **File Upload Failures**
   ```bash
   # Check disk space
   df -h
   
   # Check permissions
   ls -la uploads/
   
   # Check nginx client_max_body_size
   ```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
docker-compose restart sum-app

# View detailed logs
docker-compose logs -f --tail=100 sum-app
```

## 📈 Performance Tuning

### 1. Database Optimization

```sql
-- Add indexes
CREATE INDEX idx_files_hash ON processed_files(file_hash);
CREATE INDEX idx_memories_timestamp ON memories(timestamp DESC);
CREATE INDEX idx_jobs_status ON queue_jobs(status, created_at);

-- Analyze tables
ANALYZE;
```

### 2. Redis Caching

```python
# Configure Redis in production.py
REDIS_CONFIG = {
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    'CACHE_DEFAULT_TIMEOUT': 300
}
```

### 3. Gunicorn Workers

```python
# Optimal workers calculation
workers = multiprocessing.cpu_count() * 2 + 1
threads = 4  # For I/O bound operations
```

## 🔄 Backup and Recovery

### Automated Backups

```bash
# Create backup script
cat > /home/sum/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backup/sum/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Database backup
pg_dump -U sum_user sum_platform | gzip > "$BACKUP_DIR/database.sql.gz"

# Files backup
tar -czf "$BACKUP_DIR/uploads.tar.gz" /home/sum/SUM/uploads/
tar -czf "$BACKUP_DIR/data.tar.gz" /home/sum/SUM/data/

# Keep only last 7 days
find /backup/sum -type d -mtime +7 -exec rm -rf {} +
EOF

chmod +x /home/sum/backup.sh

# Add to crontab
crontab -e
# 0 2 * * * /home/sum/backup.sh
```

## 🚀 Production Checklist

- [ ] SSL certificates configured
- [ ] Firewall rules applied
- [ ] Database backups scheduled
- [ ] Monitoring alerts configured
- [ ] Log rotation enabled
- [ ] Security headers verified
- [ ] Rate limiting tested
- [ ] Health checks passing
- [ ] Error tracking enabled
- [ ] Performance baseline established

## 📞 Support

For production support:
- Check logs: `docker-compose logs -f`
- Health status: `curl http://localhost:5001/health/ready`
- Metrics: `curl http://localhost:5001/metrics`
- Error tracking: Check Sentry dashboard

Remember: Always test configuration changes in a staging environment first!