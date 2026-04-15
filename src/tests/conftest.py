"""Test configuration and fixtures."""

import os
import sys
import pytest
import numpy as np
from pathlib import Path

# Add project root to path so ``src`` package is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.app.config import TestingConfig
from src.app import create_app
from src.app.models import db


@pytest.fixture
def config():
    """Provide testing configuration."""
    return TestingConfig()


@pytest.fixture
def app():
    """Create Flask app with testing config."""
    config = TestingConfig()
    app, inference_service, ws_service = create_app(config)
    
    with app.app_context():
        # Create all tables
        db.create_all()
        yield app
        # Cleanup
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def app_context(app):
    """Push app context for models access."""
    with app.app_context():
        yield app


@pytest.fixture
def create_esp_node(client):
    """Factory fixture to create ESP nodes."""
    def _create_node(ip_address="192.168.1.100", room_name="Test Room"):
        response = client.post('/api/esp-nodes', json={
            "ip_address": ip_address,
            "room_name": room_name
        })
        return response.get_json(), response.status_code
    
    return _create_node


@pytest.fixture
def sample_esp_node(create_esp_node):
    """Create a sample ESP node."""
    node, status = create_esp_node("192.168.1.100", "Living Room")
    return node


@pytest.fixture
def create_temperature(client):
    """Factory fixture to create temperature records."""
    def _create_temp(esp_node_id, event_key=None, temperature=25.5, measured_at="2024-04-15T10:30:00"):
        import time
        if event_key is None:
            event_key = f"sensor_{esp_node_id}_{int(time.time() * 1000)}"
        
        response = client.post('/api/temperatures', json={
            "esp_node_id": esp_node_id,
            "event_key": event_key,
            "temperature": temperature,
            "measured_at": measured_at
        })
        return response.get_json(), response.status_code
    
    return _create_temp


@pytest.fixture
def create_scenario(client):
    """Factory fixture to create scenarios."""
    def _create_scenario(name=None, description="Test", is_active=True, esp_node_ids=None):
        import time
        if name is None:
            name = f"Scenario_{int(time.time() * 1000)}"
        
        response = client.post('/api/scenarios', json={
            "name": name,
            "description": description,
            "is_active": is_active,
            "esp_node_ids": esp_node_ids or []
        })
        return response.get_json(), response.status_code
    
    return _create_scenario


@pytest.fixture
def create_log(client):
    """Factory fixture to create audit logs."""
    def _create_log(log_type="user", action_log="Test action", concerned_column=None):
        response = client.post('/api/logging', json={
            "log_type": log_type,
            "action_log": action_log,
            "concerned_column": concerned_column
        })
        return response.get_json(), response.status_code
    
    return _create_log


@pytest.fixture
def sample_bgr_image():
    """Create sample BGR image (320x240)."""
    return np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)


@pytest.fixture
def sample_thermal_frame():
    """Create sample thermal frame (24x32 float32)."""
    # Generate realistic thermal data in range [5, 55]
    return np.random.uniform(5.0, 55.0, (24, 32)).astype(np.float32)


@pytest.fixture
def sample_gray_image():
    """Create sample grayscale image."""
    return np.random.randint(0, 256, (240, 320), dtype=np.uint8)
