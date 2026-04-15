```
Human Detection API - Project Structure
========================================

src/
│
├── app/                                   # Main Flask application package
│   ├── __init__.py                       # Flask factory (create_app)
│   ├── config.py                         # Configuration classes (Dev/Prod/Test)
│   ├── main.py                           # Server entry point with signal handling
│   │
│   ├── controllers/                      # API controllers (blueprints)
│   │   ├── __init__.py
│   │   ├── hello_controller.py           # GET /hello/ endpoint
│   │   └── inference_controller.py       # POST /inference/detect endpoint
│   │
│   ├── services/                         # Business logic services
│   │   ├── __init__.py
│   │   ├── inference_service.py          # Core detection logic
│   │   │   ├── InferenceService (factory)
│   │   │   ├── HumanDetectorBase (abstract)
│   │   │   ├── HumanDetectorNPU (RK3588)
│   │   │   └── HumanDetectorCPU (ONNX)
│   │   └── websocket_service.py          # ESP32 background listener
│   │
│   └── utils/                            # Utility modules
│       ├── __init__.py
│       └── platform_detector.py          # Hardware detection (NPU/CPU)
│
├── tests/                                # Test suite (TDD approach)
│   ├── __init__.py
│   ├── conftest.py                       # Pytest fixtures
│   ├── test_inference_service.py         # Detector & NMS tests
│   ├── test_controllers.py               # API endpoint tests
│   ├── test_websocket_service.py         # WebSocket decode tests
│   └── test_integration.py               # End-to-end workflow tests
│
│
├── run.py                                # Application launcher
├── pytest.ini                            # Pytest configuration
├── Makefile                              # Development commands
├── Dockerfile                            # Container image
├── docker-compose.yml                    # Multi-container setup
├── .env.example                          # Environment variables template
└── API_README.md                         # Comprehensive API documentation


ARCHITECTURE PATTERNS
====================

1. FACTORY PATTERN (InferenceService)
   - Creates appropriate detector based on platform
   - Hides implementation complexity from controllers
   - Single responsibility principle

2. SINGLETON PATTERN (Detectors, PlatformDetector)
   - Ensures single instance of expensive resources
   - Lazy initialization on first access
   - Thread-safe (for this use case)

3. BLUEPRINT PATTERN (Flask Blueprints)
   - Modular route registration
   - Separation of concerns
   - Easy testing and reusability

4. OBSERVER PATTERN (WebSocket)
   - Background listener notifies on frame arrival
   - Decoupled from main request/response cycle
   - Callback-based event handling

5. DEPENDENCY INJECTION
   - Services passed to controllers
   - Easier testing and mocking
   - Loose coupling between layers


COMMUNICATION FLOW
==================

Client Request
    ↓
Flask Blueprint Controller
    ↓
InferenceService (factory)
    ↓
HumanDetectorNPU/CPU (backend)
    ├→ Preprocess image (letterbox)
    ├→ Run inference
    └→ Postprocess (NMS)
    ↓
Response to Client

ESP32 WebSocket Stream (Background)
    ↓
WebSocketService (async listener)
    ↓
Frame Callback (if registered)
    ↓
Custom Processing / Storage


KEY FEATURES
============

✓ Automatic Platform Detection
  - NPU backend on RK3588
  - CPU ONNX on x86/x64
  - Transparent to caller

✓ Secure Design
  - CSRF protection ready
  - Input validation
  - Max file size limits
  - No hardcoded secrets

✓ Professional Logging
  - Structured log messages
  - Configurable levels
  - File and console output

✓ Comprehensive Testing
  - Unit tests for each service
  - Integration tests for workflows
  - Fixtures for common data
  - 80%+ code coverage target

✓ Clean Code
  - Type hints throughout
  - Docstrings (Google style)
  - Single responsibility
  - DRY principle

✓ Development Tools
  - Makefile for common tasks
  - Docker support
  - Environment variable config
  - Pre-commit checks


DEPLOYMENT OPTIONS
==================

1. Local Development
   python run.py --debug

2. Production Server
   export SECRET_KEY="..."
   python run.py

3. Docker Container
   docker build -t human-detection .
   docker run -p 5000:5000 human-detection

4. Docker Compose (with override for ESP32 URI)
   docker-compose up -d


TESTING EXECUTION
=================

Run all tests:
  pytest tests/ -v

Run with coverage:
  pytest tests/ --cov=app

Run specific module:
  pytest tests/test_inference_service.py -v

Run integration only:
  pytest tests/test_integration.py -v
```
