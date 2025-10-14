#!/bin/bash
# AI Resume Screener - Setup Script for Linux/macOS
# This script sets up the virtual environment and installs dependencies

echo "================================================"
echo "🔍 AI-POWERED RESUME SCREENER SETUP"
echo "================================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

echo "✅ Python found"
python3 --version

# Create virtual environment
echo
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "⚠️ Virtual environment already exists"
fi

# Activate virtual environment
echo
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo
echo "📈 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo
echo "📚 Installing required packages..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ All packages installed successfully"
else
    echo "❌ requirements.txt not found"
    echo "Please ensure requirements.txt is in the same directory"
    exit 1
fi

# Copy environment file
echo
echo "📄 Setting up environment configuration..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp ".env.example" ".env"
        echo "✅ Environment file created from template"
        echo "⚠️ Please edit .env file to set your API keys and preferences"
    else
        echo "⚠️ .env.example not found, creating basic .env file"
        cat > .env << EOF
NVIDIA_API_KEY=your-nvidia-api-key-here
HOST=127.0.0.1
PORT=7860
DEBUG=true
EOF
    fi
else
    echo "✅ Environment file already exists"
fi

# Create necessary directories
echo
echo "📁 Creating directories..."
mkdir -p temp logs
echo "✅ Directories created"

# Make scripts executable
chmod +x run.sh

# Setup complete
echo
echo "================================================"
echo "✅ SETUP COMPLETE!"
echo "================================================"
echo
echo "🚀 To run the application:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Set your NVIDIA API key in config.py"
echo "3. Run: python main.py"
echo
echo "Or simply run: ./run.sh"
echo
echo "📚 For detailed instructions, see README.md"
echo