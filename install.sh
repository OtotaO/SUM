#!/bin/bash
# SUM - One-line installer
# Usage: curl -sSL https://raw.githubusercontent.com/OtotaO/SUM/main/install.sh | bash

set -e

echo "🚀 Installing SUM - Simple Unified Summarizer"
echo "==========================================="

# Check Python version
if ! python3 --version | grep -E "3\.(8|9|10|11)" > /dev/null; then
    echo "❌ Error: Python 3.8+ required"
    exit 1
fi

# Check if Git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Error: Git is required"
    exit 1
fi

# Clone repository
echo "📦 Downloading SUM..."
git clone https://github.com/OtotaO/SUM.git
cd SUM

# Create virtual environment
echo "🔧 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install flask transformers torch redis

# Check if Redis is installed
if command -v redis-server &> /dev/null; then
    echo "✅ Redis found"
else
    echo "📦 Installing Redis..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install redis
            brew services start redis
        else
            echo "⚠️  Please install Redis manually or use Docker"
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y redis-server
            sudo systemctl start redis-server
        elif command -v yum &> /dev/null; then
            sudo yum install -y redis
            sudo systemctl start redis
        else
            echo "⚠️  Please install Redis manually or use Docker"
        fi
    fi
fi

# Create launcher script
cat > sum <<'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate

if [ "$1" = "simple" ]; then
    echo "Starting SUM (simple mode)..."
    python sum_simple.py
elif [ "$1" = "ultimate" ]; then
    echo "Starting SUM (ultimate mode)..."
    python sum_ultimate.py
elif [ "$1" = "cli" ]; then
    shift
    python sum_cli_simple.py "$@"
elif [ "$1" = "docker" ]; then
    echo "Starting SUM with Docker..."
    docker-compose -f docker-compose-simple.yml up
else
    echo "Starting SUM (default: ultimate mode)..."
    python sum_ultimate.py
fi
EOF

chmod +x sum

# Create desktop shortcut (optional)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    cat > ~/Desktop/SUM.command <<EOF
#!/bin/bash
cd "$PWD"
./sum
EOF
    chmod +x ~/Desktop/SUM.command
elif [[ "$OSTYPE" == "linux-gnu"* ]] && [ -d ~/Desktop ]; then
    # Linux with desktop
    cat > ~/Desktop/SUM.desktop <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=SUM
Comment=Simple Unified Summarizer
Exec=$PWD/sum
Icon=$PWD/static/icon.png
Terminal=true
Categories=Utility;
EOF
    chmod +x ~/Desktop/SUM.desktop
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "🎉 SUM is ready to use!"
echo ""
echo "Usage:"
echo "  ./sum              # Start web interface (default)"
echo "  ./sum simple       # Start simple version"
echo "  ./sum ultimate     # Start ultimate version"
echo "  ./sum cli text \"Your text\"  # CLI usage"
echo "  ./sum docker       # Run with Docker"
echo ""
echo "Web interface will be available at: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Ask if user wants to start now
read -p "Start SUM now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ./sum
fi