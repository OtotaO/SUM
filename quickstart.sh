#!/bin/bash
# quickstart.sh - Get SUM running in 30 seconds

echo "🚀 SUM Quick Start - One Click to Summarization!"
echo "================================================"
echo

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

# Quick install if requirements not met
if ! python3 -c "import flask" 2>/dev/null; then
    echo -e "${YELLOW}📦 First time setup - installing dependencies...${NC}"
    python3 -m pip install -r requirements.txt --quiet
    echo -e "${GREEN}✓ Dependencies installed!${NC}"
fi

# Create directories if needed
mkdir -p Data Output uploads temp 2>/dev/null

# Download NLTK data if needed
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)" 2>/dev/null

# Start SUM
echo -e "${GREEN}✨ Starting SUM...${NC}"
echo
echo "================================================"
echo "📊 SUM is running at: http://localhost:5001"
echo "📝 Press Ctrl+C to stop"
echo "================================================"
echo

# Run the app
python3 main.py