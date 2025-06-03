"""Central configuration module for LLM Cockpit.

All application settings are defined here in one place.
"""

import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration.
    
    All settings are loaded from environment variables with sensible defaults.
    """
    
    # Flask settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Model settings
    MODELS_DIR: Path = Path(os.getenv("MODELS_DIR", "./models"))
    MODEL_CONFIG_PATH: Path = Path(os.getenv("MODEL_CONFIG_PATH", "./app/config_models.yaml"))
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "qwen2-7b-instruct")
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    DEFAULT_TOP_P: float = float(os.getenv("DEFAULT_TOP_P", "0.95"))
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "40"))
    DEFAULT_MAX_TOKENS: int = int(os.getenv("DEFAULT_MAX_TOKENS", "2048"))
    REPEAT_PENALTY: float = float(os.getenv("REPEAT_PENALTY", "1.1"))
    PRESENCE_PENALTY: float = float(os.getenv("PRESENCE_PENALTY", "0.0"))
    FREQUENCY_PENALTY: float = float(os.getenv("FREQUENCY_PENALTY", "0.0"))
    
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///llm_cockpit.db")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Vector store settings
    CHROMA_PERSIST_DIR: Path = Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # RAG settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_DOCS: int = int(os.getenv("TOP_K_DOCS", "6"))
    
    # Authentication settings
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", SECRET_KEY)
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "True").lower() == "true"
    RATE_LIMIT_DEFAULT: str = os.getenv("RATE_LIMIT_DEFAULT", "100/hour")
    
    # File upload settings
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "20"))
    ALLOWED_EXTENSIONS: set[str] = {
        ".txt", ".pdf", ".md", ".html", ".json", ".csv",
        ".py", ".js", ".java", ".cpp", ".c", ".rs"
    }
    
    # GPU monitoring
    GPU_MONITOR_INTERVAL: int = int(os.getenv("GPU_MONITOR_INTERVAL", "1"))
    
    # Logging
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = os.getenv(
        "LOG_LEVEL", "INFO"
    ).upper()  # type: ignore
    LOG_FORMAT: Literal["json", "console"] = os.getenv(
        "LOG_FORMAT", "console" if DEBUG else "json"
    )  # type: ignore
    
    # Voice settings (optional)
    STT_MODEL: str = os.getenv("STT_MODEL", "openai/whisper-base")
    TTS_MODEL: str = os.getenv("TTS_MODEL", "tts_models/en/vctk/vits")
    
    # Public exposure
    PUBLIC_MODE: bool = os.getenv("PUBLIC_MODE", "False").lower() == "true"
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Export configuration as dictionary.
        
        Returns:
            Configuration dictionary.
        """
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith("_") and key.isupper()
        }
    
    @classmethod
    def validate(cls) -> None:
        """Validate configuration settings.
        
        Raises:
            ValueError: If configuration is invalid.
        """
        if cls.PUBLIC_MODE and cls.SECRET_KEY == "dev-secret-key-change-in-production":
            raise ValueError("SECRET_KEY must be changed when PUBLIC_MODE is enabled")
        
        if not cls.MODELS_DIR.exists():
            cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        if not cls.CHROMA_PERSIST_DIR.exists():
            cls.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

        if not 0 <= cls.DEFAULT_TEMPERATURE <= 1:
            raise ValueError("DEFAULT_TEMPERATURE must be between 0 and 1")
        if not 0 <= cls.DEFAULT_TOP_P <= 1:
            raise ValueError("DEFAULT_TOP_P must be between 0 and 1")
        if cls.DEFAULT_TOP_K < 0:
            raise ValueError("DEFAULT_TOP_K must be non-negative")
        if cls.DEFAULT_MAX_TOKENS <= 0:
            raise ValueError("DEFAULT_MAX_TOKENS must be positive")
        if cls.REPEAT_PENALTY < 0:
            raise ValueError("REPEAT_PENALTY must be non-negative")
        if cls.PRESENCE_PENALTY < 0:
            raise ValueError("PRESENCE_PENALTY must be non-negative")
        if cls.FREQUENCY_PENALTY < 0:
            raise ValueError("FREQUENCY_PENALTY must be non-negative")
        if cls.CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if cls.CHUNK_OVERLAP < 0:
            raise ValueError("CHUNK_OVERLAP must be non-negative")
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
