"""Application entry point."""

import logging
import signal
import sys

from .config import get_config
from . import create_app


logger = logging.getLogger(__name__)


def run(host: str = "0.0.0.0", port: int = 5000, debug: bool = False) -> None:
    """
    Run the Flask development server with graceful shutdown.
    
    Args:
        host: Server host address.
        port: Server port number.
        debug: Enable Flask debug mode.
    """
    config = get_config()
    app, inference_service = create_app(config)
    
    
    def shutdown_handler(signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Shutdown signal received")
        inference_service.release()
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    logger.info(f"Starting server on {host}:{port}")
    
    try:
        app.run(host=host, port=port, debug=debug, use_reloader=False)
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        inference_service.release()


if __name__ == "__main__":
    run()
