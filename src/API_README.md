# Human Detection API

Flask web application for human and hot object detection using RKNN (RK3588 NPU) or ONNX (CPU) inference backends.

## Features

- **Dual Inference Backends**: Automatically selects NPU (RK3588) or CPU ONNX based on platform
- **RESTful API**: Simple HTTP endpoints for detection
- **WebSocket Monitoring**: Background listener for ESP32 thermal data streams
- **Professional Logging**: Structured logging throughout application
- **Type Hints**: Full type annotation for code clarity
- **Test Coverage**: Comprehensive unit and integration tests (TDD approach)
- **Security**: CSRF protection, input validation, secure defaults

## Project Structure

```
.
├── app/
│   ├── __init__.py              # Flask app factory
│   ├── config.py                # Configuration management
│   ├── main.py                  # Server entry point
│   ├── controllers/             # API endpoint handlers
│   │   ├── hello_controller.py  # /hello/ endpoint
│   │   └── inference_controller.py  # /inference/detect endpoint
│   ├── services/                # Business logic services
│   │   ├── inference_service.py # Detector backends (NPU/ONNX)
│   │   └── websocket_service.py # ESP32 WebSocket listener
│   └── utils/                   # Utilities
│       └── platform_detector.py # Hardware detection
├── tests/                       # Test suite
│   ├── conftest.py             # Pytest fixtures
│   ├── test_inference_service.py   # Detector tests
│   ├── test_controllers.py      # API endpoint tests
│   ├── test_websocket_service.py   # WebSocket tests
│   └── test_integration.py      # End-to-end workflow tests
├── run.py                       # Application launcher
├── requirements.txt             # Python dependencies
└── pytest.ini                   # Test configuration
```

## Installation

### Prerequisites

- Python 3.8+
- For RK3588: RKNN toolkit2 installed (separate)
- For x86/x64: ONNX Runtime (included in requirements)

### Setup

1. Clone or navigate to project directory
2. Create Python environment:
   ```bash
   # If using conda
   conda create -n detection python=3.11
   conda activate detection
   
   # Or with venv
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. For RK3588 NPU support, install RKNN toolkit2 separately following Radxa documentation

## Usage

### Start Server

```bash
# Development server with debug mode
python run.py --debug

# Production server
python run.py --host 0.0.0.0 --port 5000

# Custom configuration via environment variables
export FLASK_ENV=production
export SECRET_KEY="your-secret-key-here"
export ESP32_WS_URI="ws://192.168.1.100:81/"
python run.py
```

### API Endpoints

#### Health Check
```bash
GET /health
# Response: {"status": "healthy"}
```

#### Hello World
```bash
GET /hello/
# Response: {"message": "Hello World"}
```

#### Image Detection
```bash
POST /inference/detect
Content-Type: multipart/form-data

# Upload image file (JPEG, PNG) or binary thermal data (24x32 float32)
# Request:
file: <image_file.jpg or thermal.bin>

# Response (200 OK):
{
    "human_count": 2,
    "hot_object_count": 1,
    "success": true
}

# Error Response (400/500):
{
    "error": "description of error",
    "success": false
}
```

### Example: Python Client

```python
import requests
import numpy as np

# Detect in image
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/inference/detect',
        files={'image': f}
    )
    print(response.json())

# Detect in thermal frame (24x32 float32)
thermal_frame = np.random.uniform(5, 55, (24, 32)).astype(np.float32)
response = requests.post(
    'http://localhost:5000/inference/detect',
    files={'image': ('thermal.bin', thermal_frame.tobytes())}
)
print(response.json())
```

## Testing

### Run All Tests
```bash
pytest
```

### Run Specific Test File
```bash
pytest tests/test_controllers.py -v
```

### With Coverage Report
```bash
pytest --cov=app tests/
```

### Run Only Unit Tests
```bash
pytest tests/test_inference_service.py tests/test_controllers.py -v
```

### Integration Tests
```bash
pytest tests/test_integration.py -v
```

## Configuration

Configuration is managed through `app/config.py`:

- **DevelopmentConfig**: Debug mode, verbose logging
- **ProductionConfig**: Secure defaults, requires SECRET_KEY env var
- **TestingConfig**: Test-specific settings

Environment Variables:
- `FLASK_ENV`: `development`, `production`, or `testing`
- `SECRET_KEY`: Session secret (required for production)
- `ESP32_WS_URI`: WebSocket address of ESP32 (default: ws://10.28.26.7:81/)
- `FLASK_HOST`: Server host (default: 0.0.0.0)
- `FLASK_PORT`: Server port (default: 5000)
- `FLASK_DEBUG`: Enable debug mode (default: false)

## Architecture

### Inference Service (Factory Pattern)

Automatically selects appropriate detector:

```python
service = InferenceService(
    rknn_model_path="path/to/model.rknn",
    onnx_model_path="path/to/model.onnx"
)

# Returns HumanDetectorNPU on RK3588
# Returns HumanDetectorCPU on x86/x64
result = service.infer(image)
```

### WebSocket Service (Background Thread)

Continuously monitors ESP32 thermal stream:

```python
service = WebSocketService(uri="ws://esp32:81/")
service.start()  # Runs in background thread

service.set_frame_callback(lambda frame: process(frame))
```

### Code Style

- Type hints on all functions
- Docstrings following Google style
- Singleton pattern for expensive resources
- Factory pattern for polymorphic creation
- Separated concerns (services, controllers, utils)

## Performance

- NPU inference (RK3588): ~10-20ms per frame
- CPU inference (ONNX): ~50-100ms per frame (depends on system)
- Memory: ~200MB per detector instance
- WebSocket: Continuous monitoring with <2ms latency

## Security Considerations

- CSRF tokens enabled
- Input validation on all endpoints
- Max file size: 16MB
- Secure headers configured
- No default credentials
- Production requires explicit SECRET_KEY

## Troubleshooting

### Import Errors
If `rknnlite` import fails on x86:
- Normal on non-RK3588 systems
- ONNX backend will be used automatically
- Install onnxruntime: `pip install onnxruntime`

### Model Not Found
```python
FileNotFoundError: [Errno 2] No such file or directory: 'path/to/model.rknn'
```
- Check model paths in config
- Ensure models exist in `rknn/` and `onnx/` directories

### WebSocket Connection Failed
```
WARNING: WebSocket error: [Errno 111] Connection refused
```
- Verify ESP32 IP address and port
- Check ESP32 is running and reachable
- Service will auto-reconnect with backoff

## License

[Specify your license here]

## Contributing

Follow PEP 8 style guide, write tests for new features, and update documentation.
