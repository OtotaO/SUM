# SUM - Simple, Perfect Summarization
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Toggle optional legendary dependency install at build time
ARG INSTALL_LEGENDARY=false

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first for better layer caching
COPY requirements.txt requirements-legendary.txt ./

# Install Python dependencies (basic + optional legendary)
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$INSTALL_LEGENDARY" = "true" ]; then pip install --no-cache-dir -r requirements-legendary.txt; fi

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p Data Output uploads temp

# Expose port
EXPOSE 5001

# Set environment variables
ENV FLASK_APP=main.py
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "main.py"]
