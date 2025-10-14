#!/bin/bash
# AI Resume Screener - Run Script for Linux/macOS
# This script activates the virtual environment and runs the application

echo "================================================"
echo "🚀 STARTING AI-POWERED RESUME SCREENER"
echo "================================================"
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./setup.sh first to create the environment"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found!"
    echo "Please ensure you are in the correct directory"
    exit 1
fi

# Display configuration info
echo "📋 Configuration:"
python -c "from config import Config; print(Config.get_summary())"
echo

# Check if config is valid
if ! python -c "from config import Config; exit(0 if Config.validate_config() else 1)"; then
    echo
    echo "⚠️ Please update your configuration in config.py"
    echo "Press Enter to continue anyway, or Ctrl+C to exit"
    read
fi

# Start the application
echo
echo "🚀 Starting AI Resume Screener..."
echo "📱 The web interface will open in your browser"
echo "🔗 URL: http://127.0.0.1:7860"
echo
echo "⏹️ Press Ctrl+C to stop the application"
echo

python main.py

echo
echo "👋 AI Resume Screener stopped"