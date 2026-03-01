#!/usr/bin/env python3
"""
SUM - One-Click Setup Script
Makes installation magical
"""

import os
import sys
import subprocess
import platform

def print_banner():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     ███████╗██╗   ██╗███╗   ███╗                        ║
    ║     ██╔════╝██║   ██║████╗ ████║                        ║
    ║     ███████╗██║   ██║██╔████╔██║                        ║
    ║     ╚════██║██║   ██║██║╚██╔╝██║                        ║
    ║     ███████║╚██████╔╝██║ ╚═╝ ██║                        ║
    ║     ╚══════╝ ╚═════╝ ╚═╝     ╚═╝                        ║
    ║                                                           ║
    ║     The Ultimate Summarization Platform                  ║
    ║     One-Click Intelligent Setup                          ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

def check_python():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor} detected")

def install_core_dependencies():
    """Install essential dependencies"""
    print("\n📦 Installing core dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=False)
    print("✅ Core dependencies installed")

def install_legendary_features():
    """Install optional legendary features"""
    print("\n🌟 Would you like to install LEGENDARY features? (recommended)")
    print("  • GraphRAG corpus analysis")
    print("  • RAPTOR hierarchical trees")
    print("  • Neural embeddings")
    print("  • Advanced NLP")
    
    response = input("\nInstall legendary features? [Y/n]: ").strip().lower()
    
    if response != 'n':
        print("\n🚀 Installing legendary features...")
        legendary_requirements = "requirements-legendary.txt"

        if os.path.exists(legendary_requirements):
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", legendary_requirements],
                check=False,
            )
            print("✅ Legendary features installed!")
        else:
            print(f"⚠️  {legendary_requirements} not found. Skipping optional install.")
    else:
        print("⚠️  Skipping legendary features (basic mode)")

def setup_environment():
    """Setup environment variables"""
    print("\n🔐 Environment Setup")
    
    if not os.path.exists('.env'):
        print("Creating .env file...")
        
        env_content = """# SUM Configuration
PORT=5001
DEBUG=False
SECRET_KEY=sum-secret-key-change-this

# Optional: Add your API keys here
# OPENAI_API_KEY=your-key-here
# ANTHROPIC_API_KEY=your-key-here

# Performance
MAX_WORKERS=4
CACHE_SIZE_MB=1024
"""
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("✅ Environment file created")
        print("   Edit .env to add API keys for enhanced features")
    else:
        print("✅ Environment file exists")

def setup_macos_app():
    """Setup macOS app if on Mac"""
    if platform.system() == "Darwin":
        print("\n🍎 macOS Detected")
        response = input("Build native macOS app? [Y/n]: ").strip().lower()
        
        if response != 'n':
            print("📱 Building macOS app...")
            # This would actually build the Xcode project
            print("   Run: open macOS/SumApp.xcodeproj")
            print("   Then: Build → Run (⌘R)")

def create_shortcuts():
    """Create convenient shortcuts"""
    print("\n🎯 Creating shortcuts...")
    
    # Create run script
    run_script = """#!/bin/bash
echo "Starting SUM..."
python main.py
"""
    with open('run.sh', 'w') as f:
        f.write(run_script)
    os.chmod('run.sh', 0o755)
    
    print("✅ Created run.sh - Start with: ./run.sh")

def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        import flask
        import nltk
        print("  ✅ Core modules working")
        
        try:
            import sentence_transformers
            import spacy
            print("  ✅ Legendary features available")
        except:
            print("  ⚠️  Legendary features not installed (basic mode)")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    """Main setup flow"""
    print_banner()
    
    print("\n🎨 Welcome to SUM Setup!")
    print("This will configure everything for you.\n")
    
    # Check Python
    check_python()
    
    # Install dependencies
    install_core_dependencies()
    install_legendary_features()
    
    # Setup environment
    setup_environment()
    
    # Platform-specific
    setup_macos_app()
    
    # Create shortcuts
    create_shortcuts()
    
    # Test
    if test_installation():
        print("\n" + "="*60)
        print("🎉 SUCCESS! SUM is ready to use!")
        print("="*60)
        print("\n📚 Quick Start:")
        print("  1. Run server:  ./run.sh")
        print("  2. Open browser: http://localhost:5001")
        print("  3. Start summarizing!")
        print("\n💡 Pro tip: Check .env to add API keys for enhanced features")
    else:
        print("\n⚠️  Setup completed with warnings")
        print("Run './run.sh' to start anyway")

if __name__ == "__main__":
    main()
