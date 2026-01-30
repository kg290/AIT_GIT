"""
Production Configuration
Environment-based settings for deployment
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class ProductionSettings(BaseSettings):
    """Production settings with environment variable support"""
    
    # Application
    APP_NAME: str = "Hospital Prescription Management System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # Database
    DATABASE_URL: str = "sqlite:///./data/hospital_prescriptions.db"
    SQL_DEBUG: bool = False
    
    # Authentication
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "change-this-in-production-use-secrets")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480  # 8 hours
    
    # Google Cloud (OCR)
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    GOOGLE_CLOUD_PROJECT: Optional[str] = None
    
    # Gemini API
    GEMINI_API_KEY: Optional[str] = None
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    LOGS_DIR: Path = DATA_DIR / "logs"
    
    # Security
    ALLOWED_ORIGINS: list = ["*"]  # Restrict in production
    REQUIRE_HTTPS: bool = False
    
    # Features
    ENABLE_AUDIT_LOGGING: bool = True
    REQUIRE_PRESCRIPTION_VERIFICATION: bool = True
    OCR_CONFIDENCE_THRESHOLD: float = 0.7
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_production_settings() -> ProductionSettings:
    """Get cached settings instance"""
    return ProductionSettings()


# Create settings instance
production_settings = get_production_settings()

# Ensure directories exist
production_settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
production_settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
production_settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
production_settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
