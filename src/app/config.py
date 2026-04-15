"""Configuration settings for the Flask application."""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Base configuration class with security defaults."""
    
    # Flask settings
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    DEBUG: bool = os.environ.get("DEBUG", "False").lower() == "true"
    TESTING: bool = False
    
    # Security settings
    JSON_SORT_KEYS: bool = False
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16MB max upload
    
    # API settings
    API_TITLE: str = "Human Detection API"
    API_VERSION: str = "1.0.0"
    
    # Model settings
    RKNN_MODEL_PATH: str = os.path.join(os.path.dirname(__file__), "..", "rknn", "Version6.rknn")
    ONNX_MODEL_PATH: str = os.path.join(os.path.dirname(__file__), "..", "onnx", "model.onnx")
    
    # Inference settings
    CONFIDENCE_THRESHOLD: float = 0.35
    IOU_THRESHOLD: float = 0.45
    IMG_SIZE: int = 320
    
    # WebSocket settings
    ESP32_WS_URI: str = os.environ.get("ESP32_WS_URI", "ws://10.28.26.7:81/")
    WS_RECONNECT_DELAY: int = 2  # seconds
    WS_MAX_SIZE: int = None  # unlimited


@dataclass
class DevelopmentConfig(Config):
    """Development environment configuration."""
    
    DEBUG: bool = True
    SECRET_KEY: str = "dev-secret-key"


@dataclass
class ProductionConfig(Config):
    """Production environment configuration."""
    
    DEBUG: bool = False
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "")
    
    def __post_init__(self):
        """Validate production settings."""
        if not self.SECRET_KEY:
            raise ValueError("SECRET_KEY env variable must be set in production")


@dataclass
class TestingConfig(Config):
    """Testing environment configuration."""
    
    TESTING: bool = True
    DEBUG: bool = True
    SECRET_KEY: str = "test-secret-key"
    ESP32_WS_URI: str = "ws://localhost:8765/"


def get_config() -> Config:
    """
    Retrieve configuration based on environment.
    
    Returns:
        Config: Configuration object for the current environment.
    """
    env = os.environ.get("FLASK_ENV", "development").lower()
    
    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()
