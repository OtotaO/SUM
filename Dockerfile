# Multi-stage Docker build for SUM - Intelligence Amplification System
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FLASK_ENV=production \
    FLASK_HOST=0.0.0.0 \
    FLASK_PORT=3000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN useradd --create-home --shell /bin/bash app

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-prod.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-prod.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Download NLTK resources
RUN python -c "import nltk; \
    nltk.download('punkt', quiet=True); \
    nltk.download('punkt_tab', quiet=True); \
    nltk.download('stopwords', quiet=True); \
    nltk.download('vader_lexicon', quiet=True); \
    nltk.download('averaged_perceptron_tagger_eng', quiet=True); \
    nltk.download('maxent_ne_chunker_tab', quiet=True); \
    nltk.download('words', quiet=True); \
    print('NLTK resources downloaded successfully')"

# Production stage
FROM base as production

# Copy application code
COPY --chown=app:app . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/temp /app/Output /app/Data && \
    chown -R app:app /app

# Switch to app user
USER app

# Expose ports
EXPOSE 3000 8765

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Default command
CMD ["python", "main.py"]

# Development stage
FROM base as development

# Install development dependencies
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy application code
COPY --chown=app:app . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/temp /app/Output /app/Data && \
    chown -R app:app /app

# Switch to app user
USER app

# Expose port and debugger port
EXPOSE 3000 8765 5678

# Development command with hot reloading
CMD ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "main.py"]

# Metadata
LABEL maintainer="ototao <https://x.com/Otota0>"
LABEL description="SUM - Intelligence Amplification System"
LABEL version="3.0.0"
LABEL features="superhuman-memory,pattern-recognition,community-intelligence,collaborative-ai"