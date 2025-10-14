---
title: AI-Powered_Resume_Screener
app_file: main.py
sdk: gradio
sdk_version: 5.49.1
---
# 🔍 AI-Powered Resume Screener

An intelligent candidate evaluation system using Deep Learning and Vector Similarity Search for local deployment.

## 📋 Overview

This AI-powered resume screener helps HR professionals and recruiters efficiently screen and analyze resumes using state-of-the-art machine learning techniques. The system provides:

- **PDF Resume Processing**: Automated text extraction from PDF documents
- **Semantic Search**: Find candidates using natural language queries
- **AI-Powered Analysis**: Generate summaries and answer questions about resumes
- **Vector Similarity**: FAISS-based efficient similarity search
- **Web Interface**: User-friendly Gradio-based interface

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (3.9 or 3.10 recommended)
- **4GB+ RAM** (8GB+ recommended)
- **Internet connection** (for initial model downloads)

### Installation

#### Windows

1. **Clone or download** this repository
2. **Run the setup script**:
   ```cmd
   setup.bat
   ```
3. **Configure API key** (optional):
   - Open `config.py`
   - Replace `"your-nvidia-api-key-here"` with your actual NVIDIA API key
4. **Start the application**:
   ```cmd
   run.bat
   ```

#### Linux/macOS

1. **Clone or download** this repository
2. **Make scripts executable**:
   ```bash
   chmod +x setup.sh run.sh
   ```
3. **Run the setup script**:
   ```bash
   ./setup.sh
   ```
4. **Configure API key** (optional):
   - Open `config.py`
   - Replace `"your-nvidia-api-key-here"` with your actual NVIDIA API key
5. **Start the application**:
   ```bash
   ./run.sh
   ```

## 📖 Detailed Installation Guide

### Step 1: Environment Setup

#### Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

#### Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

### Step 2: Configuration

#### Edit config.py
```python
# Set your NVIDIA API key (optional but recommended)
NVIDIA_API_KEY = "your-actual-api-key-here"

# Adjust server settings if needed
HOST = "127.0.0.1"
PORT = 7860
```

### Step 3: Launch Application

#### Manual Launch
```bash
# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Run application
python main.py
```

#### Using Run Scripts
```bash
# Windows
run.bat

# Linux/macOS
./run.sh
```

## 🎯 Usage Guide

### 1. **Access the Interface**
- Open your browser and go to: `http://127.0.0.1:7860`
- The web interface will load with multiple tabs

### 2. **Upload Resumes**
- Go to the **📤 Upload Resume** tab
- Select a PDF resume file
- Enter candidate name (optional)
- Click **Upload & Index**

### 3. **Search Candidates**
- Go to the **🔍 Search Candidates** tab
- Enter search queries like:
  - "Python developer with machine learning experience"
  - "Senior software engineer with leadership skills"
  - "Data scientist with PhD"
- Click **Search** to see ranked results

### 4. **Generate Summaries**
- Go to the **📋 Resume Summary** tab
- Select a candidate from the dropdown
- Click **Get Summary** for AI-generated resume summary

### 5. **Ask Questions**
- Go to the **🤖 AI Query** tab
- Ask natural language questions:
  - "Which candidates have Python experience?"
  - "Who has worked with machine learning?"
  - "Find candidates with leadership experience"

### 6. **View Statistics**
- Go to the **📊 Statistics** tab
- See system stats and indexed candidates

## 🔧 Configuration Options

### config.py Settings

```python
# Server Configuration
HOST = "127.0.0.1"      # Server host
PORT = 7860             # Server port
DEBUG = False           # Debug mode

# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
QA_MODEL = "distilbert-base-uncased-distilled-squad"

# Processing Limits
MAX_SUMMARY_LENGTH = 150
MAX_CONTEXT_LENGTH = 500
MAX_FILE_SIZE_MB = 10
```

### Environment Variables

You can also set configuration via environment variables:

```bash
# Set NVIDIA API key
export NVIDIA_API_KEY="your-api-key"

# Set environment
export FLASK_ENV="production"  # or "development", "test"
```

## 📁 Project Structure

```
AI-Powered Resume Screener/
├── main.py                 # Main application file
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── setup.bat              # Windows setup script
├── setup.sh               # Linux/macOS setup script
├── run.bat                # Windows run script
├── run.sh                 # Linux/macOS run script
├── README.md              # This file
├── IEEE_Documentation.md   # Academic documentation
├── venv/                  # Virtual environment (created during setup)
├── temp/                  # Temporary files directory
├── logs/                  # Log files directory
└── colab_notebooks/       # Original Colab notebooks
```

## 🔍 Features in Detail

### PDF Processing
- **PyPDF2** for text extraction
- Support for multi-page documents
- Error handling for corrupted files
- Automatic text cleaning and preprocessing

### Vector Search
- **FAISS** library for efficient similarity search
- **384-dimensional** embeddings from sentence-transformers
- **L2 distance** metric for similarity calculation
- Real-time indexing and search capabilities

### AI Models
- **Sentence Transformers**: all-MiniLM-L6-v2 for embeddings
- **BART**: facebook/bart-large-cnn for summarization
- **DistilBERT**: distilbert-base-uncased-distilled-squad for Q&A

### Web Interface
- **Gradio** framework for interactive UI
- **Tabbed interface** for different functionalities
- **Real-time** results and feedback
- **Mobile-responsive** design

## 🐛 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Change port in config.py
PORT = 7861  # or any available port
```

#### Model Download Issues
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/
python main.py
```

#### Memory Issues
```bash
# Reduce batch sizes or use smaller models
# Add swap space if needed (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### PDF Processing Errors
- Ensure PDF files are not corrupted
- Try different PDF files
- Check file permissions
- Verify file size (max 10MB by default)

### Performance Optimization

#### For Better Performance
1. **Use SSD** storage for faster model loading
2. **Increase RAM** for better caching
3. **Use GPU** (install `faiss-gpu` instead of `faiss-cpu`)
4. **Batch processing** for multiple files

#### Resource Requirements
- **Minimum**: 4GB RAM, 2GB storage
- **Recommended**: 8GB+ RAM, 5GB+ storage
- **GPU**: Optional but improves performance

## 🔒 Security Considerations

### Local Deployment
- Application runs on localhost by default
- No external network access required after setup
- Data stays on your local machine

### API Keys
- NVIDIA API key is optional
- Store keys securely in environment variables
- Never commit API keys to version control

### File Handling
- Only PDF files are supported
- Files are processed locally
- Temporary files are cleaned up automatically

## 📊 System Requirements

### Minimum Requirements
- **OS**: Windows 10, macOS 10.14, Ubuntu 18.04+
- **Python**: 3.8+
- **RAM**: 4GB
- **Storage**: 2GB free space
- **Network**: Internet for initial setup

### Recommended Requirements
- **OS**: Windows 11, macOS 12+, Ubuntu 20.04+
- **Python**: 3.9 or 3.10
- **RAM**: 8GB+
- **Storage**: 5GB+ free space (SSD preferred)
- **CPU**: Multi-core processor

## 🤝 Contributing

We welcome contributions! Please feel free to:

1. **Report bugs** by creating issues
2. **Suggest features** for future releases
3. **Submit pull requests** with improvements
4. **Share feedback** on usability

## 📄 License

This project is released under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face** for transformer models
- **Facebook AI** for FAISS library
- **Gradio** team for the web interface framework
- **PyPDF2** contributors for PDF processing

## 📞 Support

For support and questions:

1. **Check the troubleshooting section** above
2. **Review the configuration options**
3. **Check system requirements**
4. **Create an issue** if problems persist

## 🔄 Updates and Versioning

### Version 1.0.0 (Current)
- Initial release
- PDF processing and indexing
- Semantic search functionality
- AI-powered summarization and Q&A
- Gradio web interface

### Planned Features
- **Batch upload** for multiple resumes
- **Export functionality** for results
- **Advanced filtering** options
- **Integration APIs** for external systems
- **Docker containerization**

---

## 📝 Quick Reference Commands

### Setup (One-time)
```bash
# Windows
setup.bat

# Linux/macOS
./setup.sh
```

### Run Application
```bash
# Windows
run.bat

# Linux/macOS
./run.sh

# Manual
python main.py
```

### Stop Application
```bash
# Press Ctrl+C in terminal
```

### Update Dependencies
```bash
# Activate virtual environment first
pip install -r requirements.txt --upgrade
```

---

**🎉 Happy Resume Screening!**