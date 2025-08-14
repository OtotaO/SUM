# Multi-stage Dockerfile for SUM Platform

# Stage 1: Builder
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    libpq-dev \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements
COPY requirements.txt .
COPY requirements-prod.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir wheel && \
    pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements.txt -r requirements-prod.txt

# Stage 2: Runtime
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash sum

# Set working directory
WORKDIR /app

# Copy wheels from builder
COPY --from=builder /wheels /wheels

# Install Python packages from wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* && \
    rm -rf /wheels

# Copy application code
COPY --chown=sum:sum . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/logs /app/data/temp && \
    chown -R sum:sum /app

# Switch to non-root user
USER sum

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5001/health/live || exit 1

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_APP=production.py \
    PORT=5001

# Run with Gunicorn
CMD ["gunicorn", \
     "--config", "deployment/gunicorn_config.py", \
     "--worker-tmp-dir", "/dev/shm", \
     "production:application"]