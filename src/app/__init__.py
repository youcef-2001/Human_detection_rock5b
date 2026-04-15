"""Flask application factory."""

import logging
from typing import Optional

from flask import Flask

from .config import get_config, Config
from .models import db
from .controllers import hello_bp, inference_bp, esp_nodes_bp, temperatures_bp, logging_bp, scenarios_bp
from .controllers.inference_controller import init_inference_service
from .services import InferenceService, WebSocketService


logger = logging.getLogger(__name__)


def create_app(config: Optional[Config] = None) -> Flask:
    """
    Create and configure Flask application.
    
    Args:
        config: Optional Config object. If None, uses environment-based config.
    
    Returns:
        Configured Flask application instance.
    """
    if config is None:
        config = get_config()
    
    app = Flask(__name__)
    app.config.from_object(config)
    
    # Configure logging
    _setup_logging()
    
    logger.info(f"Creating Flask app with config: {config.__class__.__name__}")
    
    # Initialize database
    db.init_app(app)
    with app.app_context():
        db.create_all()
        logger.info("Database initialized and tables created")
    
    # Initialize inference service
    try:
        inference_service = InferenceService()
        init_inference_service(inference_service)
        logger.info("Inference service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize inference service: {e}")
        raise
    
    # Register blueprints
    app.register_blueprint(hello_bp)
    app.register_blueprint(inference_bp)
    app.register_blueprint(esp_nodes_bp)
    app.register_blueprint(temperatures_bp)
    app.register_blueprint(logging_bp)
    app.register_blueprint(scenarios_bp)
    logger.info("Blueprints registered")
    
    # Initialize WebSocket service
    ws_service = WebSocketService(uri=config.ESP32_WS_URI)
    
    @app.before_request
    def before_request():
        """Initialize app context."""
        pass
    
    @app.teardown_appcontext
    def cleanup(exception=None):
        """Cleanup resources on app teardown."""
        if exception:
            logger.error(f"App context cleanup with exception: {exception}")
    
    @app.route("/health", methods=["GET"])
    def health_check():
        """
        Health check endpoint.
        
        Returns:
            JSON status response.
        """
        return {"status": "healthy"}, 200
    
    return app, inference_service, ws_service


def _setup_logging() -> None:
    """Configure logging with appropriate handlers."""
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
