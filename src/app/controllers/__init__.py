"""Controllers module for Flask blueprints."""

from .hello_controller import hello_bp
from .inference_controller import inference_bp

__all__ = ["hello_bp", "inference_bp"]
