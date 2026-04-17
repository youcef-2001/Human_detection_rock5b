"""Flask application factory."""

import logging
from typing import Optional, Tuple

from flask import Flask
from flask_cors import CORS

from .config import get_config, Config
from .models import db
from .schema_compat import ensure_postgres_schema_compat
from .controllers import (
    hello_bp,
    inference_bp,
    esp_nodes_bp,
    temperatures_bp,
    logging_bp,
    scenarios_bp,
    auth_bp,
    users_bp,
    network_bp,
)
from .controllers.inference_controller import init_inference_service
from .services import InferenceService


logger = logging.getLogger(__name__)


def create_app(config: Optional[Config] = None) -> Tuple[Flask, Optional[InferenceService]]:
    """
    Create and configure Flask application.
    
    Args:
        config: Optional Config object. If None, uses environment-based config.
    
    Returns:
        Tuple of configured Flask application and optional inference service.
    """
    if config is None:
        config = get_config()
    
    app = Flask(__name__)
    app.config.from_object(config)

    CORS(
        app,
        resources={r"/api/*": {"origins": "*"}},
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )
    
    # Configure logging
    _setup_logging()
    
    logger.info(f"Creating Flask app with config: {config.__class__.__name__}")
    
    # Initialize database
    db.init_app(app)
    with app.app_context():
        db.create_all()
        ensure_postgres_schema_compat(db)
        logger.info("Database initialized and tables created")
    
    # Initialize inference service
    inference_service = None
    try:
        inference_service = InferenceService(config.RKNN_MODEL_PATH, config.ONNX_MODEL_PATH, config.CONFIDENCE_THRESHOLD, config.IOU_THRESHOLD)
        init_inference_service(inference_service)
        logger.info("Inference service initialized")
    except Exception as e:
        error_message = f"Inference initialization failed: {e}"
        logger.warning(error_message)
    
    # Register blueprints
    app.register_blueprint(hello_bp)
    app.register_blueprint(inference_bp)
    app.register_blueprint(esp_nodes_bp)
    app.register_blueprint(temperatures_bp)
    app.register_blueprint(logging_bp)
    app.register_blueprint(scenarios_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(users_bp)
    app.register_blueprint(network_bp)
    logger.info("Blueprints registered")
    
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
    
    return app, inference_service


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
