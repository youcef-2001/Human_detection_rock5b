"""Services module for the application."""

from .inference_service import (
    InferenceService,
    HumanDetectorNPU,
    HumanDetectorCPU,
)

__all__ = [
    "InferenceService",
    "HumanDetectorNPU",
    "HumanDetectorCPU",
]
