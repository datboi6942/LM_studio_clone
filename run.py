"""Main entry point for LLM Cockpit."""

import argparse
import sys

from app import create_app, socketio
from app.config import Config


def main() -> None:
    """Run the LLM Cockpit server."""
    parser = argparse.ArgumentParser(description="LLM Cockpit - Local LLM Interface")
    parser.add_argument(
        "--host",
        default=Config.HOST,
        help=f"Host to bind to (default: {Config.HOST})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=Config.PORT,
        help=f"Port to bind to (default: {Config.PORT})"
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Enable public mode (bind to 0.0.0.0)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Override config with CLI args
    if args.public:
        Config.PUBLIC_MODE = True
        Config.HOST = "0.0.0.0"
    
    if args.debug:
        Config.DEBUG = True
    
    # Create and run app
    app = create_app()
    
    # Run with SocketIO for WebSocket support
    socketio.run(
        app,
        host=args.host,
        port=args.port,
        debug=Config.DEBUG,
        use_reloader=Config.DEBUG,
        log_output=True
    )


if __name__ == "__main__":
    main() 
