@echo off
REM SUM One-Click Installation Script for Windows

echo ==================================================
echo     SUM - One-Click Installation
echo     The Text Summarization Standard
echo ==================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo [OK] Python detected

REM Create virtual environment
echo [OK] Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo [OK] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo [OK] Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo [OK] Installing dependencies...
pip install -r requirements.txt

REM Download NLTK data
echo [OK] Downloading NLTK data...
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"

REM Create necessary directories
echo [OK] Creating directories...
if not exist "Data" mkdir Data
if not exist "Output" mkdir Output
if not exist "uploads" mkdir uploads
if not exist "temp" mkdir temp

REM Create .env file if it doesn't exist
if not exist ".env" (
    echo [OK] Creating .env file...
    (
        echo # SUM Configuration
        echo FLASK_APP=main.py
        echo FLASK_ENV=development
        echo SECRET_KEY=your-secret-key-here
        echo PORT=5001
        echo.
        echo # Optional: Redis for caching
        echo # REDIS_URL=redis://localhost:6379/0
        echo.
        echo # Optional: API Keys for advanced features
        echo # OPENAI_API_KEY=your-key-here
        echo # ANTHROPIC_API_KEY=your-key-here
    ) > .env
)

echo.
echo ==================================================
echo [SUCCESS] Installation Complete!
echo ==================================================
echo.
echo Quick Start:
echo   1. Activate the virtual environment:
echo      venv\Scripts\activate.bat
echo.
echo   2. Run SUM:
echo      python main.py
echo.
echo   3. Open your browser:
echo      http://localhost:5001
echo.
echo Pro Tips:
echo   - Use 'python sum_cli_simple.py' for command-line summarization
echo   - Check the README.md for API documentation
echo.
echo SUM - Making summarization synonymous with SUM
echo ==================================================
pause