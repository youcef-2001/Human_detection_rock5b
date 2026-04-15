"""Flask application factory."""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional

from flask import Flask
from sqlalchemy import inspect, text

from .config import get_config, Config
from .models import db, ESPNode, Logging, Scenario, Temperature
from .controllers import hello_bp, inference_bp, esp_nodes_bp, temperatures_bp, logging_bp, scenarios_bp
from .controllers.inference_controller import init_inference_service
from .controllers.esp_nodes_controller import init_esp_fleet_service
from .services import InferenceService, ESPFleetWebSocketService


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
        _ensure_schema_compatibility()
        logger.info("Database initialized and tables created")
        _seed_sample_data(app)
    
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
    
    # Initialize fleet WebSocket service
    ws_service = ESPFleetWebSocketService(
        app=app,
        scan_subnet=getattr(config, "NETWORK_SCAN_SUBNET", None),
        auto_scan_on_startup=getattr(config, "AUTO_SCAN_ON_STARTUP", True),
        persist_interval_seconds=getattr(config, "TEMPERATURE_SAVE_INTERVAL_SECONDS", 900),
        flush_interval_seconds=getattr(config, "WS_FLUSH_INTERVAL_SECONDS", 5),
    )
    init_esp_fleet_service(ws_service)
    
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


def _seed_sample_data(app: Flask) -> None:
    """Populate a development database with representative data."""
    if not app.config.get("ENABLE_SAMPLE_DATA", False):
        return

    if ESPNode.query.first() is not None:
        return

    salon = ESPNode(ip_address="10.105.139.24", room_name="Salon")
    cuisine = ESPNode(ip_address="10.105.139.25", room_name="Cuisine")
    chambre = ESPNode(ip_address="10.105.139.26", room_name="Chambre")

    db.session.add_all([salon, cuisine, chambre])
    db.session.flush()

    now = datetime.utcnow()
    db.session.add_all([
        Temperature(
            esp_node_id=salon.id,
            event_key="seed-salon-1",
            temperature=21.6,
            measured_at=now - timedelta(minutes=8),
        ),
        Temperature(
            esp_node_id=salon.id,
            event_key="seed-salon-2",
            temperature=22.1,
            measured_at=now - timedelta(minutes=2),
        ),
        Temperature(
            esp_node_id=cuisine.id,
            event_key="seed-cuisine-1",
            temperature=24.3,
            measured_at=now - timedelta(minutes=12),
        ),
        Temperature(
            esp_node_id=chambre.id,
            event_key="seed-chambre-1",
            temperature=19.4,
            measured_at=now - timedelta(minutes=5),
        ),
    ])

    active_scenario = Scenario(
        name="Présence soirée",
        description="Active le confort sur le salon et la cuisine.",
        is_active=True,
    )
    active_scenario.esp_nodes.extend([salon, cuisine])

    night_scenario = Scenario(
        name="Nuit",
        description="Réduit la consigne dans la chambre pendant la nuit.",
        is_active=False,
    )
    night_scenario.esp_nodes.append(chambre)

    db.session.add_all([active_scenario, night_scenario])

    db.session.add_all([
        Logging(
            log_type="user",
            action_log="Scénario 'Présence soirée' activé depuis l'application.",
            concerned_column="scenarios",
        ),
        Logging(
            log_type="system",
            action_log="ESP32 Salon détecté sur 10.105.139.24.",
            concerned_column="esp_nodes",
        ),
        Logging(
            log_type="user",
            action_log="Pièce 'Cuisine' synchronisée avec le backend.",
            concerned_column="esp_nodes",
        ),
    ])

    db.session.commit()


def _ensure_schema_compatibility() -> None:
    """Best-effort schema adjustments for existing deployments."""
    inspector = inspect(db.engine)

    if inspector.has_table("esp_nodes"):
        existing_columns = {column["name"] for column in inspector.get_columns("esp_nodes")}

        if "node_uid" not in existing_columns:
            db.session.execute(text("ALTER TABLE esp_nodes ADD COLUMN node_uid VARCHAR(64)"))
            db.session.commit()

        if "updated_at" not in existing_columns:
            db.session.execute(text("ALTER TABLE esp_nodes ADD COLUMN updated_at TIMESTAMP"))
            db.session.execute(text("UPDATE esp_nodes SET updated_at = CURRENT_TIMESTAMP WHERE updated_at IS NULL"))
            db.session.commit()

        missing_uid_rows = db.session.execute(text("SELECT id FROM esp_nodes WHERE node_uid IS NULL")).fetchall()
        for row in missing_uid_rows:
            db.session.execute(
                text("UPDATE esp_nodes SET node_uid = :uid WHERE id = :id"),
                {"uid": uuid.uuid4().hex, "id": row.id},
            )
        if missing_uid_rows:
            db.session.commit()

        existing_indexes = {index["name"] for index in inspector.get_indexes("esp_nodes")}
        if "ix_esp_nodes_node_uid" not in existing_indexes:
            db.session.execute(text("CREATE UNIQUE INDEX ix_esp_nodes_node_uid ON esp_nodes (node_uid)"))
            db.session.commit()

    if inspector.has_table("scenarios"):
        scenario_columns = {column["name"] for column in inspector.get_columns("scenarios")}
        if "updated_at" not in scenario_columns:
            db.session.execute(text("ALTER TABLE scenarios ADD COLUMN updated_at TIMESTAMP"))
            db.session.execute(text("UPDATE scenarios SET updated_at = CURRENT_TIMESTAMP WHERE updated_at IS NULL"))
            db.session.commit()


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
