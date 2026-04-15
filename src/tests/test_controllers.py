"""Unit tests for Flask controllers."""

import json
import pytest
from io import BytesIO

from src.app.config import TestingConfig
from src.app import create_app
from src.app.controllers.inference_controller import init_inference_service
from src.app.services import InferenceService


@pytest.fixture
def app():
    """Create Flask app for testing."""
    config = TestingConfig()
    app, inference_service, ws_service = create_app(config)
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Create Flask test client."""
    return app.test_client()


class TestHelloController:
    """Test Hello World endpoint."""
    
    def test_hello_world_endpoint(self, client):
        """Test /hello/ returns greeting."""
        response = client.get("/hello/")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["message"] == "Hello World"
    
    def test_hello_world_json_format(self, client):
        """Test /hello/ response is valid JSON."""
        response = client.get("/hello/")
        assert response.content_type == "application/json"


class TestHealthCheck:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test /health returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"


class TestInferenceController:
    """Test inference detection endpoint."""
    
    def test_inference_endpoint_missing_image(self, client):
        """Test /inference/detect with missing image."""
        response = client.post("/inference/detect")
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
        assert "image" in data["error"].lower()
    
    def test_inference_endpoint_empty_file(self, client):
        """Test /inference/detect with empty file."""
        data = {
            'image': (BytesIO(b''), 'empty.jpg')
        }
        response = client.post(
            "/inference/detect",
            data=data,
            content_type='multipart/form-data'
        )
        assert response.status_code == 400
    
    def test_inference_endpoint_invalid_thermal_size(self, client):
        """Test /inference/detect with invalid thermal data size."""
        # Send data that's not multiple of 4 (float32)
        invalid_thermal = b'\x00\x00\x00'
        data = {
            'image': (BytesIO(invalid_thermal), 'thermal.bin')
        }
        response = client.post(
            "/inference/detect",
            data=data,
            content_type='multipart/form-data'
        )
        assert response.status_code == 400


class TestAppInitialization:
    """Test Flask app initialization."""
    
    def test_app_creation(self, app):
        """Test app is created successfully."""
        assert app is not None
        assert app.config['TESTING'] is True
    
    def test_blueprints_registered(self, app):
        """Test blueprints are registered."""
        registered_blueprints = app.blueprints.keys()
        assert 'hello' in registered_blueprints
        assert 'inference' in registered_blueprints
    
    def test_config_loaded(self, app):
        """Test configuration is loaded correctly."""
        assert app.config['API_TITLE'] == "Human Detection API"
        assert app.config['CONFIDENCE_THRESHOLD'] == 0.35


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
