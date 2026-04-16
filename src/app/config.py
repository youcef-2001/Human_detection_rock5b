"""Configuration settings for the Flask application."""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv


logger = logging.getLogger(__name__)

# Load .env file from src directory
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    logger.warning(
        "No .env file found at %s. Falling back to process environment/defaults.",
        env_path,
    )


def _get_required_env(name: str) -> str:
    """Return a required env variable value or fail fast."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value.strip()


def _get_bool_env(name: str) -> bool:
    """Parse boolean env variable from true/false string."""
    return _get_required_env(name).lower() == "true"


def _get_int_env(name: str) -> int:
    """Parse integer env variable."""
    return int(_get_required_env(name))


def _get_float_env(name: str) -> float:
    """Parse float env variable."""
    return float(_get_required_env(name))


def _is_running_in_docker() -> bool:
    """Best-effort check to detect if the app runs inside a Docker container."""
    return Path("/.dockerenv").exists()


def _resolve_database_url() -> str:
    """
    Resolve runtime DB URL and enforce PostgreSQL only.

    If the app runs on host and DATABASE_URL points to docker service hostname
    "postgres", replace it with localhost so the same postgres container
    (and its persisted volume) remains reachable from host execution.
    """
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("Missing required environment variable: DATABASE_URL")

    if not (database_url.startswith("postgresql://") or database_url.startswith("postgresql+")):
        raise RuntimeError("DATABASE_URL must target PostgreSQL (sqlite is not supported)")

    if not _is_running_in_docker() and "@postgres:" in database_url:
        return database_url.replace("@postgres:", "@localhost:", 1)

    return database_url


def _resolve_test_database_url() -> str:
    """
    Resolve test DB URL and enforce PostgreSQL only.

    Tests use a dedicated PostgreSQL database to avoid any sqlite fallback and
    prevent destructive operations on the main database.
    """
    database_url = os.environ.get("TEST_DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("Missing required environment variable: TEST_DATABASE_URL")

    if not (database_url.startswith("postgresql://") or database_url.startswith("postgresql+")):
        raise RuntimeError("TEST_DATABASE_URL must target PostgreSQL (sqlite is not supported)")

    if not _is_running_in_docker() and "@postgres:" in database_url:
        return database_url.replace("@postgres:", "@localhost:", 1)

    return database_url


class Config:
    """Base configuration class with security defaults."""
    
    # Flask settings
    SECRET_KEY = _get_required_env("SECRET_KEY")
    DEBUG = _get_bool_env("FLASK_DEBUG")
    TESTING = False
    
    # Security settings
    JSON_SORT_KEYS = False
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload
    
    # API settings
    API_TITLE = "Human Detection API"
    API_VERSION = "1.0.0"
    
    # Model settings
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    RKNN_MODEL_PATH = _get_required_env("RKNN_MODEL_PATH")
    ONNX_MODEL_PATH = _get_required_env("ONNX_MODEL_PATH")
    
    # Inference settings
    CONFIDENCE_THRESHOLD = _get_float_env("CONFIDENCE_THRESHOLD")
    IOU_THRESHOLD = _get_float_env("IOU_THRESHOLD")
    IMG_SIZE = _get_int_env("IMG_SIZE")
    
    # Database settings (from .env)
    SQLALCHEMY_DATABASE_URI = _resolve_database_url()
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # Logging
    LOG_LEVEL = _get_required_env("LOG_LEVEL")


class DevelopmentConfig(Config):
    """Development environment configuration."""
    pass


class ProductionConfig(Config):
    """Production environment configuration."""
    pass


class TestingConfig(Config):
    """Testing environment configuration."""
    
    TESTING = True
    DEBUG = True
    SECRET_KEY = "test-secret-key"
    SQLALCHEMY_DATABASE_URI = _resolve_test_database_url()


def get_config():
    """
    Retrieve configuration based on environment.
    
    Returns:
        Config: Configuration object for the current environment.
    """
    env = _get_required_env("FLASK_ENV").lower()
    
    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()
