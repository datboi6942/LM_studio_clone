"""Model registry for managing multiple models."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import re

import structlog

from app.config import Config
from app.models import BaseModelRunner, LlamaCppRunner

logger = structlog.get_logger(__name__)


class ModelRegistry:
    """Registry for managing multiple models with hot-swapping."""
    
    def __init__(self) -> None:
        """Initialize model registry."""
        self.models: Dict[str, BaseModelRunner] = {}
        self.active_model: Optional[str] = None
        self.config_path = Config.MODEL_CONFIG_PATH
    
    def load_models(self) -> None:
        """Load all models by scanning the models directory dynamically."""
        # Clear existing models
        self.models.clear()
        self.active_model = None
        
        # Scan models directory for GGUF files
        models_dir = Config.MODELS_DIR
        if not models_dir.exists():
            logger.warning("Models directory not found", path=str(models_dir))
            models_dir.mkdir(parents=True, exist_ok=True)
            return
        
        # Find all GGUF files recursively
        gguf_files = list(models_dir.rglob("*.gguf"))
        
        if not gguf_files:
            logger.info("No GGUF model files found in models directory")
            return
        
        # Define quantization preference order (higher quality first)
        quantization_preference = ['Q8_0', 'Q6_K', 'Q5_K_M', 'Q5_0', 'Q4_K_S', 'Q4_K_M', 'Q3_K_L', 'Q3_K_M', 'Q3_K_S', 'Q2_K']
        
        # Track found models by base name to prefer higher quality quantizations
        found_models = {}
        
        # Register each model found
        for model_file in gguf_files:
            try:
                # Generate model ID from file path
                model_id = self._generate_model_id(model_file)
                
                # Extract base model name (without quantization)
                base_name = model_id
                for quant in quantization_preference:
                    if f"_{quant.lower()}" in model_id:
                        base_name = model_id.replace(f"_{quant.lower()}", '')
                        break
                
                # Check if we already have this base model with a better quantization
                if base_name in found_models:
                    # Compare quantization levels
                    existing_quant = None
                    new_quant = None
                    
                    for quant in quantization_preference:
                        if f"_{quant.lower()}" in found_models[base_name]['id']:
                            existing_quant = quant
                        if f"_{quant.lower()}" in model_id:
                            new_quant = quant
                    
                    # Skip if existing model has better or same quantization
                    if existing_quant and new_quant:
                        existing_rank = quantization_preference.index(existing_quant) if existing_quant in quantization_preference else 999
                        new_rank = quantization_preference.index(new_quant) if new_quant in quantization_preference else 999
                        if existing_rank <= new_rank:
                            continue
                
                # Store this model
                found_models[base_name] = {'id': model_id, 'file': model_file}
                
            except Exception as e:
                logger.error("Failed to process model", path=str(model_file), error=str(e))
        
        # Sort by base name, then by quantization preference
        sorted_models = sorted(
            found_models.values(),
            key=lambda m: (
                # Prioritize Phi-3 models
                0 if 'phi' in str(m['file']).lower() else 1,
                # Then by quantization quality
                quantization_preference.index(m['id'].split('_')[-1]) if m['id'].split('_')[-1] in quantization_preference else len(quantization_preference)
            )
        )
        
        # Now register the best quantization for each model
        for model_info in sorted_models:
            try:
                self._register_model_from_file(model_info['file'], {})
            except Exception as e:
                logger.error("Failed to register model", path=str(model_info['file']), error=str(e))
        
        # Set the first model as active (preferably Phi-3)
        if sorted_models and not self.active_model:
            self.set_active_model(sorted_models[0]['id'])
        
        logger.info(f"Loaded {len(self.models)} models dynamically")
    
    def set_active_model(self, model_id: str) -> None:
        """Set the active model.
        
        Args:
            model_id: Model ID to set as active.
        """
        if model_id not in self.models:
            raise KeyError(f"Model not found: {model_id}")
        
        self.active_model = model_id
        logger.info("Set active model", model_id=model_id)
    
    def _register_model_from_file(self, model_path: Path, custom_configs: Dict[str, Any]) -> None:
        """Register a model from a GGUF file.
        
        Args:
            model_path: Path to the GGUF file.
            custom_configs: Any custom configurations from config file.
        """
        # Generate model ID from file path
        model_id = self._generate_model_id(model_path)
        
        # Create default config
        config = {
            "model_id": model_id,
            "path": str(model_path),
            "engine": "llama_cpp",
            "n_gpu_layers": 35,  # Default value
        }
        
        # Override with any custom configs
        if model_id in custom_configs:
            config.update(custom_configs[model_id])
            # Ensure path is still correct (in case config has wrong path)
            config["path"] = str(model_path)
        
        # Create model runner
        runner = LlamaCppRunner(**config)
        self.models[model_id] = runner
        
        logger.info("Registered model", model_id=model_id, path=str(model_path))
    
    def _generate_model_id(self, model_path: Path) -> str:
        """Generate a clean model ID from file path.
        
        Args:
            model_path: Path to model file.
            
        Returns:
            Clean model identifier.
        """
        # Get relative path from models directory
        models_dir = Config.MODELS_DIR
        try:
            rel_path = model_path.relative_to(models_dir)
        except ValueError:
            # Fallback to just filename if not under models dir
            rel_path = model_path
        
        # Remove .gguf extension and convert to clean ID
        name_parts = []
        
        # Add parent directory names (but not "models")
        for part in rel_path.parent.parts:
            if part.lower() != "models":
                name_parts.append(part)
        
        # Add filename without extension
        filename = rel_path.stem
        name_parts.append(filename)
        
        # Join and clean up
        model_id = "_".join(name_parts)
        model_id = re.sub(r'[^a-zA-Z0-9_]', '_', model_id)  # Replace special chars
        model_id = re.sub(r'_+', '_', model_id)  # Collapse multiple underscores
        model_id = model_id.strip('_').lower()  # Remove leading/trailing underscores and lowercase
        
        return model_id or "unknown_model"
    
    def _register_model(self, model_id: str, config: Dict[str, Any]) -> None:
        """Register a model from config (legacy method for backward compatibility).
        
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
    
    def reload_models(self) -> None:
        """Reload models from configuration file.
        
        This allows dynamic model updates without restarting the application.
        """
        logger.info("Reloading models from configuration")
        
        # Keep track of currently loaded models
        previously_loaded = {
            model_id: model.is_loaded 
            for model_id, model in self.models.items()
        }
        
        # Clear current registry
        self.models.clear()
        self.active_model = None
        
        # Reload from config
        self.load_models()
        
        # Restore load state for models that still exist
        for model_id, was_loaded in previously_loaded.items():
            if model_id in self.models and was_loaded:
                try:
                    self.models[model_id].load()
                    logger.info("Restored loaded state", model_id=model_id)
                except Exception as e:
                    logger.error("Failed to restore model load state", 
                               model_id=model_id, error=str(e))
    
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