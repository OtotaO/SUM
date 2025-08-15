#\!/bin/bash
# SUM One-Click Installation Script for Unix/Linux/Mac

echo "=================================================="
echo "    SUM - One-Click Installation"
echo "    The Text Summarization Standard"
echo "=================================================="
echo

# Check if Python 3.8+ is installed
if \! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Get Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

# Compare versions
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" \!= "$REQUIRED_VERSION" ]; then
    echo "❌ Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)"
    exit 1
fi

echo "✓ Python $PYTHON_VERSION detected"

# Create virtual environment
echo "✓ Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "✓ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "✓ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "✓ Installing dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo "✓ Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"

# Create necessary directories
echo "✓ Creating directories..."
mkdir -p Data Output uploads temp

# Create .env file if it doesn't exist
if [ \! -f .env ]; then
    echo "✓ Creating .env file..."
    cat > .env << EOL
# SUM Configuration
FLASK_APP=main.py
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
PORT=5001

# Optional: Redis for caching
# REDIS_URL=redis://localhost:6379/0

# Optional: API Keys for advanced features
# OPENAI_API_KEY=your-key-here
# ANTHROPIC_API_KEY=your-key-here
EOL
fi

echo
echo "=================================================="
echo "✅ Installation Complete\!"
echo "=================================================="
echo
echo "📚 Quick Start:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo
echo "  2. Run SUM:"
echo "     python main.py"
echo
echo "  3. Open your browser:"
echo "     http://localhost:5001"
echo
echo "💡 Pro Tips:"
echo "  - Use 'python sum_cli_simple.py' for command-line summarization"
echo "  - Check the README.md for API documentation"
echo
echo "🚀 SUM - Making summarization synonymous with SUM"
echo "=================================================="
EOF < /dev/null