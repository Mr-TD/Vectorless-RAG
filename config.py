"""
Vectorless RAG — Application Configuration
============================================
Loads settings from environment variables (.env file).
Provides separate configs for development and production.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


class Config:
    """Base configuration shared across all environments."""

    # --- Flask ---
    SECRET_KEY = os.getenv("SECRET_KEY", "fallback-secret-key")

    # --- File Storage ---
    UPLOAD_FOLDER = os.path.join(BASE_DIR, os.getenv("UPLOAD_FOLDER", "data/uploads"))
    INDEX_FOLDER = os.path.join(BASE_DIR, os.getenv("INDEX_FOLDER", "data/indexes"))
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH_MB", 50)) * 1024 * 1024  # Convert MB to bytes
    ALLOWED_EXTENSIONS = {"pdf", "txt", "md"}

    # --- NVIDIA NIM API ---
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
    NVIDIA_API_BASE_URL = os.getenv(
        "NVIDIA_API_BASE_URL",
        "https://integrate.api.nvidia.com/v1"
    )

    # --- LLM Settings ---
    LLM_MODEL = os.getenv("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 16384))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.7))
    LLM_TOP_P = float(os.getenv("LLM_TOP_P", 0.95))

    # --- RAG Settings (Vectorless) ---
    MAX_TRAVERSAL_DEPTH = int(os.getenv("MAX_TRAVERSAL_DEPTH", 10))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 120))

    # --- Traditional RAG Settings ---
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
    TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", 5))
    TRADITIONAL_INDEX_FOLDER = os.path.join(BASE_DIR, os.getenv("TRADITIONAL_INDEX_FOLDER", "data/traditional_indexes"))

    @staticmethod
    def init_app(app):
        """Hook for subclass-specific initialization."""
        # Ensure required directories exist
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        os.makedirs(app.config["INDEX_FOLDER"], exist_ok=True)
        os.makedirs(app.config["TRADITIONAL_INDEX_FOLDER"], exist_ok=True)


class DevelopmentConfig(Config):
    """Development-specific configuration."""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production-specific configuration."""
    DEBUG = False
    TESTING = False

    @staticmethod
    def init_app(app):
        Config.init_app(app)
        # In production, enforce that NVIDIA_API_KEY is set
        if not app.config.get("NVIDIA_API_KEY"):
            raise ValueError("NVIDIA_API_KEY must be set in production environment.")


class TestingConfig(Config):
    """Testing-specific configuration."""
    DEBUG = True
    TESTING = True
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "data/test_uploads")
    INDEX_FOLDER = os.path.join(BASE_DIR, "data/test_indexes")
    TRADITIONAL_INDEX_FOLDER = os.path.join(BASE_DIR, "data/test_traditional_indexes")


# Map environment names to config classes
config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}
