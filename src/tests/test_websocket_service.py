"""Unit tests for WebSocket service."""

import numpy as np
import json
import base64
import io
import pytest

from app.services.websocket_service import WebSocketService


class TestWebSocketService:
    """Test WebSocket monitoring service."""
    
    def test_service_initialization(self):
        """Test WebSocket service initializes correctly."""
        service = WebSocketService(uri="ws://localhost:8765/")
        assert service.uri == "ws://localhost:8765/"
        assert not service._running
    
    def test_service_callback_setting(self):
        """Test callback can be registered."""
        def dummy_callback(frame):
            pass
        
        service = WebSocketService(uri="ws://localhost:8765/")
        service.set_frame_callback(dummy_callback)
        assert service.on_frame_callback == dummy_callback
    
    def test_decode_npy_payload_from_bytes(self):
        """Test decoding NPY format from bytes."""
        thermal_frame = np.random.uniform(5, 55, (24, 32)).astype(np.float32)
        
        # Create NPY bytes
        buffer = io.BytesIO()
        np.save(buffer, thermal_frame)
        npy_bytes = buffer.getvalue()
        
        decoded = WebSocketService._decode_payload(npy_bytes)
        assert decoded is not None
        assert decoded.shape == (24, 32)
    
    def test_decode_float32_payload_from_bytes(self):
        """Test decoding raw float32 binary."""
        thermal_frame = np.random.uniform(5, 55, (24, 32)).astype(np.float32)
        binary_data = thermal_frame.tobytes()
        
        decoded = WebSocketService._decode_payload(binary_data)
        assert decoded is not None
        assert decoded.shape == (24, 32)
    
    def test_decode_npy_base64_json(self):
        """Test decoding base64 NPY in JSON."""
        thermal_frame = np.random.uniform(5, 55, (24, 32)).astype(np.float32)
        
        # Create NPY bytes and encode
        buffer = io.BytesIO()
        np.save(buffer, thermal_frame)
        npy_bytes = buffer.getvalue()
        npy_base64 = base64.b64encode(npy_bytes).decode('utf-8')
        
        payload = json.dumps({"npy_base64": npy_base64})
        decoded = WebSocketService._decode_payload(payload)
        assert decoded is not None
        assert decoded.shape == (24, 32)
    
    def test_decode_float32_base64_json(self):
        """Test decoding base64 float32 in JSON."""
        thermal_frame = np.random.uniform(5, 55, (24, 32)).astype(np.float32)
        
        # Encode float32 as base64
        binary_data = thermal_frame.tobytes()
        f32_base64 = base64.b64encode(binary_data).decode('utf-8')
        
        payload = json.dumps({"float32_base64": f32_base64})
        decoded = WebSocketService._decode_payload(payload)
        assert decoded is not None
        assert decoded.shape == (24, 32)
    
    def test_decode_invalid_json(self):
        """Test decoding invalid JSON returns None."""
        payload = "not valid json {"
        decoded = WebSocketService._decode_payload(payload)
        assert decoded is None
    
    def test_decode_invalid_binary_size(self):
        """Test decoding binary with invalid size."""
        # 3 bytes is not multiple of 4
        invalid_binary = b'\x00\x00\x00'
        decoded = WebSocketService._decode_payload(invalid_binary)
        assert decoded is None
    
    def test_decode_wrong_thermal_size(self):
        """Test decoding float32 with wrong thermal size."""
        # Create wrong size array
        wrong_size_data = np.zeros(100, dtype=np.float32).tobytes()
        decoded = WebSocketService._decode_payload(wrong_size_data)
        assert decoded is not None  # Returns as-is without reshape


class TestWebSocketStartStop:
    """Test WebSocket service lifecycle."""
    
    def test_service_cannot_start_twice(self):
        """Test service cannot be started twice."""
        service = WebSocketService(uri="ws://localhost:8765/")
        service._running = True  # Simulate running state
        
        with pytest.raises(RuntimeError):
            service.start()
    
    def test_service_stop_when_not_running(self):
        """Test stop is safe when service not running."""
        service = WebSocketService(uri="ws://localhost:8765/")
        # Should not raise
        service.stop()
        assert not service._running


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
