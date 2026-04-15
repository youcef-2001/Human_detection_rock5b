"""Services module for the application."""

from .inference_service import (
    InferenceService,
    HumanDetectorNPU,
    HumanDetectorCPU,
)
from .websocket_service import WebSocketService

__all__ = [
    "InferenceService",
    "HumanDetectorNPU",
    "HumanDetectorCPU",
    "WebSocketService",
]
