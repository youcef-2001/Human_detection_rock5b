#!/usr/bin/env python
"""Application launcher script with development server."""

import os
import argparse

from app.main import run


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
