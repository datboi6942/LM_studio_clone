"""Model registry for managing multiple models."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

from ..config import Config
from ..models import BaseModelRunner, LlamaCppRunner

logger = structlog.get_logger(__name__)


class ModelRegistry:
    """Registry for managing multiple models with hot-swapping."""
    
    def __init__(self) -> None:
        """Initialize model registry."""
        self.models: Dict[str, BaseModelRunner] = {}
        self.active_model: Optional[str] = None
        self.config_path = Config.MODEL_CONFIG_PATH
    
    def load_models(self) -> None:
        """Load all models from configuration file."""
        if not self.config_path.exists():
            logger.warning("Model config not found, creating default")
            self._create_default_config()
        
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
        
        for model_id, model_config in config.items():
            try:
                self._register_model(model_id, model_config)
            except Exception as e:
                logger.error("Failed to register model", model_id=model_id, error=str(e))
        
        # Set default active model
        if not self.active_model and self.models:
            self.active_model = Config.DEFAULT_MODEL if Config.DEFAULT_MODEL in self.models else list(self.models.keys())[0]
            logger.info("Set active model", model_id=self.active_model)
    
    def _register_model(self, model_id: str, config: Dict[str, Any]) -> None:
        """Register a model.
        
        Args:
            model_id: Unique model identifier.
            config: Model configuration.
        """
        engine = config.get("engine", "llama_cpp")
        config["model_id"] = model_id
        
        # Check if model file exists
        model_path = Path(config.get("path", ""))
        if not model_path.is_absolute():
            model_path = Config.MODELS_DIR / model_path
        
        if not model_path.exists():
            logger.warning(
                "Model file not found, skipping registration", 
                model_id=model_id, 
                path=str(model_path)
            )
            return
        
        # Update config with absolute path
        config["path"] = str(model_path)
        
        if engine == "llama_cpp":
            runner = LlamaCppRunner(**config)
        else:
            raise ValueError(f"Unknown engine: {engine}")
        
        self.models[model_id] = runner
        logger.info("Registered model", model_id=model_id, engine=engine, path=str(model_path))
    
    def get_model(self, model_id: Optional[str] = None) -> BaseModelRunner:
        """Get a model by ID or return active model.
        
        Args:
            model_id: Model ID, or None for active model.
            
        Returns:
            Model runner.
            
        Raises:
            KeyError: If model not found.
            RuntimeError: If no models loaded.
        """
        if not self.models:
            raise RuntimeError("No models loaded")
        
        if model_id:
            if model_id not in self.models:
                raise KeyError(f"Model not found: {model_id}")
            return self.models[model_id]
        
        if not self.active_model:
            raise RuntimeError("No active model set")
        
        return self.models[self.active_model]
    
    def load_model(self, model_id: str) -> None:
        """Load a model into memory.
        
        Args:
            model_id: Model to load.
        """
        model = self.get_model(model_id)
        if not model.is_loaded:
            model.load()
            logger.info("Loaded model", model_id=model_id)
    
    def unload_model(self, model_id: str) -> None:
        """Unload a model from memory.
        
        Args:
            model_id: Model to unload.
        """
        model = self.get_model(model_id)
        if model.is_loaded:
            model.unload()
            logger.info("Unloaded model", model_id=model_id)
    
    def switch_model(self, model_id: str) -> None:
        """Switch active model (with hot-swap).
        
        Args:
            model_id: Model to switch to.
        """
        if model_id not in self.models:
            raise KeyError(f"Model not found: {model_id}")
        
        # Unload current model if different
        if self.active_model and self.active_model != model_id:
            current = self.models[self.active_model]
            if current.is_loaded:
                current.unload()
        
        # Load new model
        self.active_model = model_id
        new_model = self.models[model_id]
        if not new_model.is_loaded:
            new_model.load()
        
        logger.info("Switched active model", model_id=model_id)
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all registered models with their info.
        
        Returns:
            Dictionary of model info.
        """
        return {
            model_id: {
                **model.get_model_info(),
                "active": model_id == self.active_model
            }
            for model_id, model in self.models.items()
        }
    
    def _create_default_config(self) -> None:
        """Create default model configuration file."""
        default_config = {
            "qwen2-7b-instruct": {
                "path": str(Config.MODELS_DIR / "qwen2-7b-instruct-q4_k_m.gguf"),
                "engine": "llama_cpp",
                "n_gpu_layers": 35,
                "rope_freq_base": 20000
            }
        }
        
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False) 