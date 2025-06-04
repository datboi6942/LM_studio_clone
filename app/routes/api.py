"""OpenAI-compatible API routes."""

import json
from typing import Any, Dict, Iterator, Optional

import structlog
from flask import Blueprint, Response, current_app, jsonify, request

from app.models import CompletionChunk

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
            logger.info("=== MODEL ACCESS DEBUG START ===")
            logger.info("Getting model", model_id=model_id)
            model = current_app.model_registry.get_model(model_id)
            logger.info("Model retrieved successfully", model=model, is_loaded=model.is_loaded)
        except (KeyError, RuntimeError) as e:
            logger.error("Model not found", model_id=model_id, error=str(e))
            return jsonify({"error": f"Model not found: {model_id}"}), 404
        
        # Load model if needed
        if not model.is_loaded:
            logger.info("Auto-loading model", model_id=model_id)
            try:
                model.load()
                logger.info("Model loaded successfully", model_id=model_id, is_loaded=model.is_loaded)
            except Exception as load_error:
                logger.error("Failed to load model", model_id=model_id, error=str(load_error), exc_info=True)
                return jsonify({"error": f"Failed to load model: {load_error}"}), 500
        else:
            logger.info("Model already loaded", model_id=model_id)
        
        logger.info("=== MODEL ACCESS DEBUG END ===")
        
        # Generate response
        if stream:
            def generate() -> Iterator[str]:
                try:
                    logger.info("=== ROUTE GENERATE START ===")
                    logger.info("Starting streaming generation in route", model_id=model_id, messages_count=len(messages))
                    logger.info("About to call model.chat", model=model, is_loaded=model.is_loaded)
                    
                    chunk_count = 0
                    for chunk in model.chat(messages, stream=True, **kwargs):
                        chunk_count += 1
                        # Log what we're sending
                        logger.info(f"Route received chunk #{chunk_count}", chunk=chunk)
                        # Send the chunk immediately with proper SSE format
                        sse_data = f"data: {json.dumps(chunk)}\n\n"
                        logger.info(f"Route sending SSE data: {sse_data.strip()}")
                        yield sse_data
                    # Send a final [DONE] message to indicate completion
                    logger.info(f"Route stream completed with {chunk_count} chunks")
                    yield "data: [DONE]\n\n"
                    logger.info("=== ROUTE GENERATE END ===")
                except Exception as e:
                    logger.error("Route streaming error", error=str(e), exc_info=True)
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


@bp.route("/test/stream", methods=["GET"])
def test_stream() -> Any:
    """Test SSE streaming endpoint."""
    def generate():
        import time
        for i in range(5):
            time.sleep(0.5)
            yield f"data: {json.dumps({'message': f'Test message {i+1}', 'index': i})}\n\n"
        yield "data: [DONE]\n\n"
    
    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    ) 