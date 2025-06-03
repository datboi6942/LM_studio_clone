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
        chat_id = data.get("chat_id")
        
        # Debug logging
        logger.info("Chat completion request", model_id=model_id, messages_count=len(messages), stream=stream)
        
        # Validate model_id
        if not model_id:
            logger.error("No model specified in request")
            return jsonify({"error": "No model specified"}), 400
        
        # Generation parameters
        from ..config import Config

        kwargs = {
            "temperature": data.get("temperature", Config.DEFAULT_TEMPERATURE),
            "top_p": data.get("top_p", Config.DEFAULT_TOP_P),
            "top_k": data.get("top_k", Config.DEFAULT_TOP_K),
            "max_tokens": data.get("max_tokens", Config.DEFAULT_MAX_TOKENS),
            "repeat_penalty": data.get("repeat_penalty", Config.REPEAT_PENALTY),
            "presence_penalty": data.get("presence_penalty", Config.PRESENCE_PENALTY),
            "frequency_penalty": data.get("frequency_penalty", Config.FREQUENCY_PENALTY),
            "stop": data.get("stop"),
        }
        
        # Get model
        try:
            model = current_app.model_registry.get_model(model_id)
        except (KeyError, RuntimeError) as e:
            logger.error("Model not found", model_id=model_id, error=str(e))
            return jsonify({"error": f"Model not found: {model_id}"}), 404
        
        # Load model if needed
        if not model.is_loaded:
            logger.info("Auto-loading model", model_id=model_id)
            model.load()

        if chat_id and messages:
            last = messages[-1]
            if last.get("role") == "user":
                current_app.chat_store.add_message(chat_id, "user", last.get("content", ""))
        
        # Generate response
        if stream:
            def generate() -> Iterator[str]:
                try:
                    assistant_text = ""
                    for chunk in model.chat(messages, stream=True, **kwargs):
                        # Send the chunk immediately with proper SSE format
                        yield f"data: {json.dumps(chunk)}\n\n"
                        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                        if delta:
                            assistant_text += delta
                    # Send a final [DONE] message to indicate completion
                    yield "data: [DONE]\n\n"
                    if chat_id and assistant_text:
                        current_app.chat_store.add_message(chat_id, "assistant", assistant_text)
                except Exception as e:
                    logger.error("Streaming error", error=str(e))
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
            response = Response(
                generate(),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "X-Accel-Buffering": "no",  # Disable Nginx buffering
                    "Connection": "keep-alive",
                    "Transfer-Encoding": "chunked"
                }
            )
            response.implicit_sequence_conversion = False  # Disable Flask's buffering
            return response
        else:
            # Non-streaming response
            chunks = list(model.chat(messages, stream=False, **kwargs))
            if chunks:
                if chat_id:
                    assistant_content = chunks[-1].get("choices", [{}])[0].get("message", {}).get("content", "")
                    if assistant_content:
                        current_app.chat_store.add_message(chat_id, "assistant", assistant_content)
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