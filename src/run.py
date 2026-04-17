#!/usr/bin/env python
"""Application launcher script with development server."""

import os
import argparse
import sys
from pathlib import Path


def _ensure_python_311() -> None:
    """Fail fast when the interpreter is not Python 3.11."""
    if sys.version_info[:2] != (3, 11):
        current = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        raise RuntimeError(
            "This project requires Python 3.11.x. "
            f"Current interpreter: {current}. "
            "Create/activate a 3.11 virtual environment and retry."
        )


_ensure_python_311()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    # Keep site-packages precedence to avoid shadowing third-party modules
    # by similarly named top-level project directories (e.g. onnx/).
    sys.path.append(str(PROJECT_ROOT))

from src.app.main import run


def main():
    """
    Parse arguments and launch the application.
    
    Supports command-line configuration for host, port, and debug mode.
    """
    parser = argparse.ArgumentParser(
        description="Human Detection API Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("FLASK_HOST", "0.0.0.0"),
        help="Server host address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("FLASK_PORT", 5000)),
        help="Server port number (default: 5000)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=os.environ.get("FLASK_DEBUG", "").lower() == "true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
