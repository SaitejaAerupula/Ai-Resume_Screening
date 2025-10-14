"""
Configuration file for AI Resume Screener
Contains all configuration settings including API keys and system parameters
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for the AI Resume Screener application"""
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "7860"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # NVIDIA API Configuration
    # Set your NVIDIA API key here or as an environment variable
    NVIDIA_API_KEY: Optional[str] = os.getenv("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", "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")
    
    # Model Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    SUMMARIZATION_MODEL: str = os.getenv("SUMMARIZATION_MODEL", "facebook/bart-large-cnn")
    QA_MODEL: str = os.getenv("QA_MODEL", "distilbert-base-uncased-distilled-squad")
    
    # FAISS Configuration
    VECTOR_DIMENSION: int = int(os.getenv("VECTOR_DIMENSION", "384"))  # Dimension for all-MiniLM-L6-v2
    INDEX_TYPE: str = os.getenv("INDEX_TYPE", "IndexFlatL2")  # FAISS index type
    
    # Processing Configuration
    MAX_SUMMARY_LENGTH: int = int(os.getenv("MAX_SUMMARY_LENGTH", "150"))
    MIN_SUMMARY_LENGTH: int = int(os.getenv("MIN_SUMMARY_LENGTH", "50"))
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "500"))
    MAX_TEXT_LENGTH_FOR_SUMMARY: int = int(os.getenv("MAX_SUMMARY_TEXT_LENGTH", "1000"))
    
    # File Configuration
    SUPPORTED_FILE_TYPES: list = [".pdf"]
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    TEMP_DIR: str = os.getenv("TEMP_DIR", "temp")
    LOG_FILE: str = os.getenv("LOG_FILE", "resume_screener.log")
    
    # Search Configuration
    DEFAULT_SEARCH_RESULTS: int = int(os.getenv("DEFAULT_SEARCH_RESULTS", "5"))
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "10"))
    
    # UI Configuration
    GRADIO_THEME: str = os.getenv("GRADIO_THEME", "soft")
    SHARE_LINK: bool = os.getenv("SHARE_LINK", "False").lower() == "true"
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        if cls.NVIDIA_API_KEY == "your-nvidia-api-key-here":
            print("⚠️ WARNING: NVIDIA API key not set!")
            print("Please set your NVIDIA API key in config.py or as environment variable NVIDIA_API_KEY")
            return False
        
        if cls.PORT < 1024 or cls.PORT > 65535:
            print(f"⚠️ WARNING: Invalid port {cls.PORT}. Using default 7860")
            cls.PORT = 7860
        
        return True
    
    @classmethod
    def get_summary(cls) -> str:
        """Get configuration summary"""
        return f"""
🔧 Configuration Summary:
- Server: {cls.HOST}:{cls.PORT}
- NVIDIA API: {'✅ Configured' if cls.NVIDIA_API_KEY != 'your-nvidia-api-key-here' else '❌ Not set'}
- Embedding Model: {cls.EMBEDDING_MODEL}
- Vector Dimension: {cls.VECTOR_DIMENSION}
- Max File Size: {cls.MAX_FILE_SIZE_MB}MB
- Debug Mode: {'✅ Enabled' if cls.DEBUG else '❌ Disabled'}
        """.strip()


# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    SHARE_LINK = False


class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    SHARE_LINK = False


class TestConfig(Config):
    """Test environment configuration"""
    DEBUG = True
    PORT = 7861
    LOG_FILE = "test_resume_screener.log"


# Select configuration based on environment
ENV = os.getenv("FLASK_ENV", "development").lower()

if ENV == "production":
    Config = ProductionConfig
elif ENV == "test":
    Config = TestConfig
else:
    Config = DevelopmentConfig