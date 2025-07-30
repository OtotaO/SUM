# Dockerfile for SUM - Hierarchical Knowledge Densification System
# Provides one-command deployment with all dependencies

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=3000

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional WebSocket dependencies
RUN pip install --no-cache-dir websockets psutil

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/temp /app/Output /app/Data

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

# Expose ports
EXPOSE 3000 8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000/api/health || exit 1

# Create startup script
RUN echo '#!/bin/bash\n\
echo "🚀 Starting SUM - Hierarchical Knowledge Densification System"\n\
echo "📊 Main API Server: http://localhost:3000"\n\
echo "⚡ Progressive Summarization: http://localhost:3000/api/progressive_summarization"\n\
echo "🌐 WebSocket Server: ws://localhost:8765"\n\
echo ""\n\
# Start Flask API server in background\n\
python main.py &\n\
FLASK_PID=$!\n\
echo "✅ Flask API server started (PID: $FLASK_PID)"\n\
\n\
# Wait a moment for Flask to start\n\
sleep 5\n\
\n\
# Start Progressive Summarization WebSocket server\n\
echo "🚀 Starting Progressive Summarization WebSocket server..."\n\
python progressive_summarization.py &\n\
WEBSOCKET_PID=$!\n\
echo "✅ WebSocket server started (PID: $WEBSOCKET_PID)"\n\
\n\
# Function to handle shutdown\n\
shutdown() {\n\
    echo "🛑 Shutting down SUM services..."\n\
    kill $FLASK_PID $WEBSOCKET_PID 2>/dev/null\n\
    exit 0\n\
}\n\
\n\
# Trap signals\n\
trap shutdown SIGTERM SIGINT\n\
\n\
echo "🌟 SUM is ready! All services running."\n\
echo "📖 Visit the API documentation at http://localhost:3000/api/docs"\n\
echo "⚡ Try progressive summarization by opening progressive_client.html"\n\
\n\
# Wait for processes\n\
wait\n\
' > /app/start.sh && chmod +x /app/start.sh

# Default command
CMD ["/app/start.sh"]

# Metadata
LABEL maintainer="ototao <https://x.com/Otota0>"
LABEL description="SUM - Revolutionary Hierarchical Knowledge Densification System with Real-Time Progressive Summarization"
LABEL version="2.0.0"
LABEL features="hierarchical-processing,real-time-progress,websocket-api,unlimited-text-processing"