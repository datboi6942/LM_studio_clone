"""Chat UI routes."""

from flask import Blueprint, render_template, request, jsonify, session, send_from_directory, current_app
from flask_socketio import emit
import json
from datetime import datetime
from typing import List, Dict, Any
import os
import subprocess
import threading

bp = Blueprint("chat", __name__, url_prefix="/")


@bp.route("/favicon.ico")
def favicon():
    """Serve favicon."""
    return send_from_directory('static', 'favicon.ico', mimetype='image/vnd.microsoft.icon')


@bp.route("/")
def index():
    """Main chat interface."""
    return render_template("chat.html")


@bp.route("/chat/list")
def list_chats():
    """Get chat list for sidebar."""
    folder = request.args.get("folder", "")
    
    # Return empty list since we don't have a database yet
    # In a real implementation, this would query the database
    chats: List[Dict[str, Any]] = []
    
    return render_template("components/chat_list.html", chats=chats)


@bp.route("/chat/folders")
def list_folders():
    """Get folder list for sidebar."""
    # Return empty list since we don't have a database yet
    # In a real implementation, this would query the database
    folders: List[Dict[str, Any]] = []
    
    return render_template("components/folder_list.html", folders=folders)


@bp.route("/chat/search")
def search_chats():
    """Search chats."""
    query = request.args.get("q", "")
    folder = request.args.get("folder", "")
    
    # Return empty results since we don't have a database yet
    chats: List[Dict[str, Any]] = []
    
    return render_template("components/chat_list.html", chats=chats)


@bp.route("/chat/create", methods=["POST"])
def create_chat():
    """Create a new chat."""
    # For now, just reload the page to start a new chat
    return jsonify({
        "success": True,
        "chat_id": None,
        "redirect": "/"
    })


@bp.route("/chat/folder/create", methods=["POST"])
def create_folder():
    """Create a new folder."""
    name = request.form.get("name", "").strip()
    
    if not name:
        return jsonify({"error": "Folder name is required"}), 400
    
    # For now, just return the folder (it won't persist)
    folder = {"name": name, "count": 0}
    return render_template("components/folder_item.html", folder=folder)


@bp.route("/api/chat/<int:chat_id>")
def get_chat(chat_id):
    """Get chat details."""
    # Return empty chat since we don't have persistence yet
    chat = {
        "id": chat_id,
        "title": "New Chat",
        "messages": [],
        "rag_files": []
    }
    
    return jsonify(chat)


@bp.route("/api/models")
def get_models():
    """Get available models from model registry."""
    try:
        model_registry = current_app.model_registry
        models_info = model_registry.list_models()
        
        # Convert to the format expected by the frontend
        models = []
        for model_id, info in models_info.items():
            models.append({
                "name": model_id,
                "display_name": info.get("display_name", model_id),
                "type": info.get("engine", "unknown"),
                "size": info.get("size", "unknown"),
                "context_length": info.get("context_length", 0),
                "loaded": info.get("loaded", False),
                "active": info.get("active", False)
            })
        
        return jsonify(models)
    
    except Exception as e:
        current_app.logger.error(f"Failed to get models: {e}")
        return jsonify([]), 500


@bp.route("/api/models/<model_name>/info")
def get_model_info(model_name):
    """Get model information from registry."""
    try:
        model_registry = current_app.model_registry
        model = model_registry.get_model(model_name)
        info = model.get_model_info()
        
        return jsonify({
            "name": model_name,
            "type": info.get("engine", "unknown"),
            "size": info.get("size", "unknown"),
            "context_length": info.get("context_length", 0),
            "loaded": model.is_loaded if hasattr(model, 'is_loaded') else False,
            "memory_usage": info.get("memory_usage", "unknown")
        })
    
    except Exception as e:
        current_app.logger.error(f"Failed to get model info for {model_name}: {e}")
        return jsonify({
            "name": model_name,
            "type": "unknown",
            "size": "unknown", 
            "context_length": 0,
            "loaded": False,
            "memory_usage": "unknown"
        }), 404


@bp.route("/api/models/<model_name>/load", methods=["POST"])
def load_model(model_name):
    """Load a model."""
    try:
        model_registry = current_app.model_registry
        model_registry.load_model(model_name)
        return jsonify({"success": True, "message": f"Model {model_name} loaded successfully"})
    
    except Exception as e:
        current_app.logger.error(f"Failed to load model {model_name}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/api/models/<model_name>/unload", methods=["POST"]) 
def unload_model(model_name):
    """Unload a model."""
    try:
        model_registry = current_app.model_registry
        model_registry.unload_model(model_name)
        return jsonify({"success": True, "message": f"Model {model_name} unloaded successfully"})
    
    except Exception as e:
        current_app.logger.error(f"Failed to unload model {model_name}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/api/models/<model_name>/switch", methods=["POST"])
def switch_model(model_name):
    """Switch active model."""
    try:
        model_registry = current_app.model_registry
        model_registry.switch_model(model_name)
        return jsonify({"success": True, "message": f"Switched to model {model_name}"})
    
    except Exception as e:
        current_app.logger.error(f"Failed to switch to model {model_name}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/api/models/download", methods=["POST"])
def download_model():
    """Download a model from HuggingFace."""
    try:
        data = request.json
        model_name = data.get("model_name")
        
        if not model_name:
            return jsonify({"success": False, "error": "Model name is required"}), 400
        
        # Start download in background
        def download_task():
            with current_app.app_context():
                try:
                    # Use the download_model.py script
                    script_path = os.path.join(os.getcwd(), "download_model.py")
                    result = subprocess.run([
                        "python", script_path, model_name
                    ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
                    
                    if result.returncode == 0:
                        current_app.logger.info(f"Model {model_name} downloaded successfully")
                        # TODO: Add model to registry
                    else:
                        current_app.logger.error(f"Failed to download model {model_name}: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    current_app.logger.error(f"Download timeout for model {model_name}")
                except Exception as e:
                    current_app.logger.error(f"Download error for model {model_name}: {e}")
        
        # Start download in background thread
        thread = threading.Thread(target=download_task)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "success": True, 
            "message": f"Started downloading {model_name}. Check logs for progress."
        })
    
    except Exception as e:
        current_app.logger.error(f"Failed to start download: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/api/models/available")
def get_available_models():
    """Get list of models available for download."""
    # Popular small models that work well for testing
    available_models = [
        {
            "name": "microsoft/DialoGPT-medium",
            "display_name": "DialoGPT Medium",
            "size": "1.5GB",
            "description": "Conversational model, good for chat"
        },
        {
            "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
            "display_name": "TinyLlama 1.1B Chat",
            "size": "2.2GB",
            "description": "Small but capable chat model"
        },
        {
            "name": "microsoft/DialoGPT-small",
            "display_name": "DialoGPT Small", 
            "size": "500MB",
            "description": "Lightweight conversational model"
        }
    ]
    
    return jsonify(available_models) 