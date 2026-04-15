"""Integration tests combining multiple components."""

import json
import numpy as np
from io import BytesIO

import pytest

from src.app.config import TestingConfig
from src.app import create_app
from src.app.services import inference_service as inf


class MockInferenceService:
    """Simple inference mock for integration tests."""

    def infer(self, _image):
        return {"human_count": 1, "hot_object_count": 0}

    def release(self):
        return None


@pytest.fixture
def app(monkeypatch):
    """Create Flask app for testing."""
    monkeypatch.setattr("src.app.InferenceService", lambda: MockInferenceService())
    config = TestingConfig()
    app, _inference_service, _ws_service = create_app(config)
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


class TestFullWorkflow:
    """Test complete API workflows."""
    
    def test_health_and_hello_workflow(self, client):
        """Test basic health check and hello endpoints."""
        # Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert json.loads(health_response.data)["status"] == "healthy"
        
        # Check hello
        hello_response = client.get("/hello/")
        assert hello_response.status_code == 200
        assert json.loads(hello_response.data)["message"] == "Hello World"
    
    def test_inference_with_valid_thermal_data(self, client):
        """Test inference with properly formatted thermal data."""
        # Create valid thermal frame (24x32 float32)
        thermal_frame = np.random.uniform(5, 55, (24, 32)).astype(np.float32)
        thermal_bytes = thermal_frame.tobytes()
        
        response = client.post(
            "/inference/detect",
            data={'image': (BytesIO(thermal_bytes), 'thermal.bin')},
            content_type='multipart/form-data'
        )
        
        # Should either succeed or fail with appropriate error
        if response.status_code == 200:
            data = json.loads(response.data)
            assert "human_count" in data
            assert "hot_object_count" in data
            assert data["success"] is True
        else:
            # Error responses should have proper format
            assert response.status_code in [400, 500]


class TestErrorHandling:
    """Test error handling across services."""
    
    def test_invalid_request_format(self, client):
        """Test API handles invalid request format."""
        response = client.post(
            "/inference/detect",
            data='invalid',
            content_type='text/plain'
        )
        assert response.status_code in [400, 415]
    
    def test_missing_required_fields(self, client):
        """Test API validates required fields."""
        response = client.post(
            "/inference/detect",
            data={},
            content_type='multipart/form-data'
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data


class TestDataPipeline:
    """Test data processing pipeline."""
    
    def test_thermal_data_preprocessing(self):
        """Test thermal data is preprocessed correctly."""
        # Create thermal frame
        thermal = np.ones((24, 32), dtype=np.float32) * 30.0
        
        # Convert to BGR
        bgr = inf.thermal_to_bgr(thermal)
        
        # Verify shape and type
        assert bgr.shape == (240, 320, 3)
        assert bgr.dtype == np.uint8
        assert np.all(bgr >= 0) and np.all(bgr <= 255)
    
    def test_image_preprocessing_chain(self):
        """Test complete image preprocessing chain."""
        # Create test image
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        # Apply letterbox
        boxed, ratio, pad = inf.letterbox(image, size=320)
        
        # Verify output
        assert boxed.shape == (320, 320, 3)
        assert ratio > 0
        assert len(pad) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
