"""OpenAI-compatible API routes."""

import json
from typing import Any, Dict, Iterator, Optional

import structlog
from flask import Blueprint, Response, current_app, jsonify, request

from ..models import CompletionChunk

logger = structlog.get_logger(__name__)

bp = Blueprint("api", __name__, url_prefix="/api")


def make_sse(data: Dict[str, Any]) -> str:
    """Format data for Server-Sent Events.
    
    Args:
        data: Data to send.
        
    Returns:
        SSE-formatted string.
    """
    return f"data: {json.dumps(data)}\n\n"


@bp.route("/v1/models", methods=["GET"])
def list_models() -> Any:
    """List available models."""
    models = current_app.model_registry.list_models()
    
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 1700000000,
                "owned_by": "user",
                **info
            }
            for model_id, info in models.items()
        ]
    })


@bp.route("/v1/chat/completions", methods=["POST"])
def chat_completions() -> Any:
    """Create chat completion (OpenAI-compatible)."""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract parameters
        model_id = data.get("model")
        messages = data.get("messages", [])
        stream = data.get("stream", False)
        
        # Generation parameters
        kwargs = {
            "temperature": data.get("temperature", 0.7),
            "top_p": data.get("top_p", 0.95),
            "top_k": data.get("top_k", 40),
            "max_tokens": data.get("max_tokens", 2048),
        }
        
        # Get model
        try:
            model = current_app.model_registry.get_model(model_id)
        except (KeyError, RuntimeError) as e:
            return jsonify({"error": str(e)}), 404
        
        # Load model if needed
        if not model.is_loaded:
            model.load()
        
        # Generate response
        if stream:
            def generate() -> Iterator[str]:
                try:
                    for chunk in model.chat(messages, stream=True, **kwargs):
                        yield make_sse(chunk)
                    yield make_sse({"done": True})
                except Exception as e:
                    logger.error("Streaming error", error=str(e))
                    yield make_sse({"error": str(e)})
            
            return Response(
                generate(),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # Non-streaming response
            chunks = list(model.chat(messages, stream=False, **kwargs))
            if chunks:
                return jsonify(chunks[-1])
            else:
                return jsonify({"error": "No response generated"}), 500
                
    except Exception as e:
        logger.error("Chat completion error", error=str(e))
        return jsonify({"error": str(e)}), 500


@bp.route("/v1/models/<model_id>/load", methods=["POST"])
def load_model(model_id: str) -> Any:
    """Load a model into memory."""
    try:
        current_app.model_registry.load_model(model_id)
        return jsonify({"status": "loaded", "model": model_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/v1/models/<model_id>/unload", methods=["POST"])
def unload_model(model_id: str) -> Any:
    """Unload a model from memory."""
    try:
        current_app.model_registry.unload_model(model_id)
        return jsonify({"status": "unloaded", "model": model_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/v1/models/switch", methods=["POST"])
def switch_model() -> Any:
    """Switch active model."""
    data = request.json
    if not data or "model" not in data:
        return jsonify({"error": "Model ID required"}), 400
    
    try:
        current_app.model_registry.switch_model(data["model"])
        return jsonify({"status": "switched", "model": data["model"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/health", methods=["GET"])
def health_check() -> Any:
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "models_loaded": len([
            m for m in current_app.model_registry.models.values()
            if m.is_loaded
        ]),
        "active_model": current_app.model_registry.active_model
    }) 