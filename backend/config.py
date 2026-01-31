"""
Configuration settings for Medical AI Gateway 2.0
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "Medical AI Gateway 2.0"
    DEBUG: bool = True
    VERSION: str = "2.0.0"
    
    # Database
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/data/medical_gateway.db"
    
    # Google Cloud Vision
    GOOGLE_APPLICATION_CREDENTIALS: str = str(BASE_DIR / "kg-hackathon-e3f03b59d928.json")
    
    # Google Gemini AI
    GEMINI_API_KEY: Optional[str] = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    USE_GEMINI: bool = True  # Enable Gemini AI for prescription understanding
    
    # OpenAI (for LLM-based features)
    OPENAI_API_KEY: Optional[str] = None
    USE_LLM: bool = False  # Set to True if OpenAI key is available
    
    # File Storage
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    
    # Processing
    OCR_CONFIDENCE_THRESHOLD: float = 0.7
    ENTITY_CONFIDENCE_THRESHOLD: float = 0.6
    INTERACTION_SEVERITY_THRESHOLD: str = "moderate"
    
    # Audit
    ENABLE_AUDIT_LOGGING: bool = True
    AUDIT_LOG_PATH: Path = BASE_DIR / "data" / "audit_logs"
    
    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()

# Ensure directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
settings.AUDIT_LOG_PATH.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "data").mkdir(parents=True, exist_ok=True)
