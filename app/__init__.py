"""LLM Cockpit - A powerful local LLM interface."""

import structlog
from flask import Flask
from pathlib import Path
from flask_socketio import SocketIO

from .config import Config
from .services.logging import setup_logging

socketio = SocketIO()


def create_app(config_override: dict | None = None) -> Flask:
    """Create and configure the Flask application.
    
    Args:
        config_override: Optional configuration overrides.
        
    Returns:
        Configured Flask application.
    """
    # Setup logging first
    setup_logging(Config.LOG_LEVEL, Config.LOG_FORMAT)
    logger = structlog.get_logger(__name__)
    
    # Create Flask app
    app = Flask(__name__)
    app.config.from_object(Config)
    
    if config_override:
        app.config.update(config_override)
    
    # Configure Flask for streaming
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.config['RESPONSE_TIMEOUT'] = 300  # 5 minutes for long generations
    
    # Validate configuration
    Config.validate()
    
    # Initialize extensions
    socketio.init_app(app, cors_allowed_origins="*" if Config.PUBLIC_MODE else None)
    
    # Register blueprints
    from .routes import api, chat
    app.register_blueprint(api.bp)
    app.register_blueprint(chat.bp)
    
    # Initialize model registry
    from .services.model_registry import ModelRegistry
    app.model_registry = ModelRegistry()

    # Initialize chat store using SQLite
    from .services.chat_store import ChatStore
    db_path = Path(Config.DATABASE_URL.replace("sqlite:///", ""))
    app.chat_store = ChatStore(db_path)
    
    # Load models on startup
    with app.app_context():
        try:
            app.model_registry.load_models()
        except Exception as e:
            logger.error("Failed to load models", error=str(e))
    
    # Start background tasks
    if Config.GPU_MONITOR_INTERVAL > 0:
        from .services.gpu_monitor import start_gpu_monitoring
        start_gpu_monitoring(socketio)
    
    logger.info(
        "LLM Cockpit initialized",
        host=Config.HOST,
        port=Config.PORT,
        public_mode=Config.PUBLIC_MODE
    )
    
    return app 
