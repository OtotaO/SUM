#!/usr/bin/env python3
"""
SUM One-Click Installation Script

Installs all dependencies and sets up SUM in seconds.
Works on Windows, Mac, and Linux.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


class SUMInstaller:
    """Simple, robust installer for SUM."""
    
    def __init__(self):
        self.python_cmd = sys.executable
        self.pip_cmd = [self.python_cmd, '-m', 'pip']
        self.platform = platform.system()
        self.errors = []
        
    def print_banner(self):
        """Display installation banner."""
        print("\n" + "="*50)
        print("    SUM - One-Click Installation")
        print("    The Text Summarization Standard")
        print("="*50 + "\n")
        
    def check_python_version(self):
        """Ensure Python 3.8+ is installed."""
        print("✓ Checking Python version...", end=" ")
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print(f"❌ Python 3.8+ required (found {version.major}.{version.minor})")
            return False
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
        
    def upgrade_pip(self):
        """Upgrade pip to latest version."""
        print("✓ Upgrading pip...", end=" ")
        try:
            subprocess.run(
                self.pip_cmd + ['install', '--upgrade', 'pip'],
                check=True,
                capture_output=True,
                text=True
            )
            print("✓")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌\n  Error: {e.stderr}")
            self.errors.append("Failed to upgrade pip")
            return False
            
    def install_requirements(self):
        """Install all requirements."""
        print("✓ Installing core dependencies...")
        
        requirements_file = Path(__file__).parent / 'requirements.txt'
        if not requirements_file.exists():
            print("❌ requirements.txt not found!")
            return False
            
        try:
            # Install with progress bar
            result = subprocess.run(
                self.pip_cmd + ['install', '-r', str(requirements_file)],
                check=True,
                text=True
            )
            print("✓ All dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Installation failed")
            self.errors.append("Failed to install requirements")
            return False
            
    def download_nltk_data(self):
        """Download required NLTK data."""
        print("✓ Downloading NLTK data...", end=" ")
        try:
            import nltk
            
            # Create NLTK data directory if it doesn't exist
            nltk_data_dir = Path.home() / 'nltk_data'
            nltk_data_dir.mkdir(exist_ok=True)
            
            # Download required data
            required_data = ['punkt', 'stopwords', 'vader_lexicon', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
            
            for data in required_data:
                try:
                    nltk.download(data, quiet=True)
                except:
                    pass  # Some data might already exist
                    
            print("✓")
            return True
        except Exception as e:
            print(f"⚠️  Warning: {str(e)}")
            self.errors.append("NLTK data download incomplete (non-critical)")
            return True  # Non-critical error
            
    def create_directories(self):
        """Create required directories."""
        print("✓ Creating directories...", end=" ")
        
        directories = ['Data', 'Output', 'uploads', 'temp']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            
        print("✓")
        return True
        
    def create_env_file(self):
        """Create .env file with defaults."""
        env_file = Path('.env')
        if not env_file.exists():
            print("✓ Creating .env file...", end=" ")
            env_content = """# SUM Configuration
FLASK_APP=main.py
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
PORT=5001

# Optional: Redis for caching
# REDIS_URL=redis://localhost:6379/0

# Optional: API Keys for advanced features
# OPENAI_API_KEY=your-key-here
# ANTHROPIC_API_KEY=your-key-here
"""
            env_file.write_text(env_content)
            print("✓")
        return True
        
    def test_installation(self):
        """Test that SUM can be imported."""
        print("✓ Testing installation...", end=" ")
        try:
            # Test core imports
            import summarization_engine
            import main
            from flask import Flask
            import nltk
            print("✓")
            return True
        except ImportError as e:
            print(f"❌\n  Import error: {str(e)}")
            self.errors.append(f"Import test failed: {str(e)}")
            return False
            
    def print_next_steps(self):
        """Print instructions for running SUM."""
        print("\n" + "="*50)
        print("✅ Installation Complete!")
        print("="*50)
        
        if self.errors:
            print("\n⚠️  Warnings:")
            for error in self.errors:
                print(f"  - {error}")
                
        print("\n📚 Quick Start:")
        print("  1. Run SUM:")
        print(f"     {self.python_cmd} main.py")
        print("\n  2. Open your browser:")
        print("     http://localhost:5001")
        print("\n  3. Start summarizing!")
        
        print("\n💡 Pro Tips:")
        print("  - Use 'python sum_cli_simple.py' for command-line summarization")
        print("  - Check the README.md for API documentation")
        print("  - Join our community for support and updates")
        
        print("\n🚀 SUM - Making summarization synonymous with SUM")
        print("="*50 + "\n")
        
    def install(self):
        """Run the complete installation process."""
        self.print_banner()
        
        # Check Python version
        if not self.check_python_version():
            return False
            
        # Run installation steps
        steps = [
            self.upgrade_pip,
            self.install_requirements,
            self.download_nltk_data,
            self.create_directories,
            self.create_env_file,
            self.test_installation
        ]
        
        for step in steps:
            if not step():
                if step.__name__ not in ['download_nltk_data']:  # Non-critical
                    print("\n❌ Installation failed. Please check the errors above.")
                    return False
                    
        self.print_next_steps()
        return True


def main():
    """Run the installer."""
    installer = SUMInstaller()
    
    try:
        success = installer.install()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()