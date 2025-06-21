"""Chat UI routes."""

import os
import re
import threading
from typing import Any

from flask import (
    Blueprint,
    current_app,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

bp = Blueprint("chat", __name__, url_prefix="/")


@bp.route("/favicon.ico")
def favicon():
    """Serve favicon."""
    return send_from_directory(
        "static", "favicon.ico", mimetype="image/vnd.microsoft.icon"
    )


@bp.route("/")
def index():
    """Main chat interface."""
    from ..config import Config

    return render_template("chat.html", config=Config)


@bp.route("/chat/list")
def list_chats():
    """Get chat list for sidebar."""

    folder = request.args.get("folder")
    search = request.args.get("q")
    chats = current_app.db.list_chats(folder=folder, search=search)

    return render_template("components/chat_list.html", chats=chats)


@bp.route("/chat/folders")
def list_folders():
    """Get folder list for sidebar."""
    folders = current_app.db.list_folders()

    return render_template("components/folder_list.html", folders=folders)


@bp.route("/chat/search")
def search_chats():
    """Search chats."""

    query = request.args.get("q", "").strip()
    chats = current_app.db.list_chats(search=query)

    return render_template("components/chat_list.html", chats=chats)


@bp.route("/chat/create", methods=["POST"])
def create_chat():
    """Create a new chat."""
    title = request.form.get("title", "New Chat")
    folder = request.form.get("folder") or None
    chat_id = current_app.db.create_chat(title=title, folder=folder)

    return jsonify({"success": True, "chat_id": chat_id})


@bp.route("/chat/folder/create", methods=["POST"])
def create_folder():
    """Create a new folder."""
    name = request.form.get("name", "").strip()
    if not re.match(r"^[\w\- ]+$", name):
        return jsonify({"error": "Invalid folder name"}), 400

    if not name:
        return jsonify({"error": "Folder name is required"}), 400

    current_app.db.create_folder(name)
    folder = {"name": name, "count": 0}
    return render_template("components/folder_item.html", folder=folder)


@bp.route("/chat/<int:chat_id>/delete", methods=["POST"])
def delete_chat(chat_id: int):
    """Delete a chat."""
    current_app.db.delete_chat(chat_id)
    return jsonify({"success": True})


@bp.route("/chat/<int:chat_id>/rename", methods=["POST"])
def rename_chat(chat_id: int):
    """Rename a chat."""
    title = request.form.get("title", "New Chat").strip()
    if not title:
        return jsonify({"error": "Title required"}), 400
    current_app.db.rename_chat(chat_id, title)
    return jsonify({"success": True})


@bp.route("/api/chat/<int:chat_id>")
def get_chat(chat_id):
    """Get chat details."""
    try:
        chat = current_app.db.get_chat(chat_id)
        return jsonify(chat)
    except KeyError:
        return jsonify({"error": "Chat not found"}), 404


@bp.route("/api/chat/<int:chat_id>/messages", methods=["POST"])
def add_message(chat_id: int):
    """Append a message to a chat."""
    data = request.get_json() or {}
    role = data.get("role")
    content = data.get("content", "")
    if role not in {"user", "assistant"} or not content:
        return jsonify({"error": "Invalid payload"}), 400
    try:
        current_app.db.add_message(chat_id, role, content)
    except KeyError:
        return jsonify({"error": "Chat not found"}), 404
    return jsonify({"success": True})


@bp.route("/api/models")
def get_models():
    """Get available models from model registry."""
    try:
        model_registry = current_app.model_registry
        models_info = model_registry.list_models()

        # Convert to the format expected by the frontend
        models = []
        for model_id, info in models_info.items():
            models.append(
                {
                    "name": model_id,
                    "display_name": info.get("display_name", model_id),
                    "type": info.get("engine", "unknown"),
                    "size": info.get("size", "unknown"),
                    "context_length": info.get("context_length", 0),
                    "loaded": info.get("loaded", False),
                    "active": info.get("active", False),
                }
            )

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

        return jsonify(
            {
                "name": model_name,
                "type": info.get("engine", "unknown"),
                "size": info.get("size", "unknown"),
                "context_length": info.get("context_length", 0),
                "loaded": model.is_loaded if hasattr(model, "is_loaded") else False,
                "memory_usage": info.get("memory_usage", "unknown"),
            }
        )

    except Exception as e:
        current_app.logger.error(f"Failed to get model info for {model_name}: {e}")
        return (
            jsonify(
                {
                    "name": model_name,
                    "type": "unknown",
                    "size": "unknown",
                    "context_length": 0,
                    "loaded": False,
                    "memory_usage": "unknown",
                }
            ),
            404,
        )


@bp.route("/api/models/<model_name>/load", methods=["POST"])
def load_model(model_name):
    """Load a model."""
    try:
        model_registry = current_app.model_registry
        model_registry.load_model(model_name)

        # Emit socket event to notify frontend of successful load
        socketio = current_app.extensions.get("socketio")
        if socketio:
            socketio.emit("model_loaded", {"model_id": model_name, "status": "loaded"})

        return jsonify(
            {
                "success": True,
                "message": f"Model {model_name} loaded successfully",
                "loaded": True,
            }
        )

    except Exception as e:
        current_app.logger.error(f"Failed to load model {model_name}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/api/models/<model_name>/unload", methods=["POST"])
def unload_model(model_name):
    """Unload a model."""
    try:
        model_registry = current_app.model_registry
        model_registry.unload_model(model_name)

        # Emit socket event to notify frontend of successful unload
        socketio = current_app.extensions.get("socketio")
        if socketio:
            socketio.emit(
                "model_unloaded", {"model_id": model_name, "status": "unloaded"}
            )

        return jsonify(
            {
                "success": True,
                "message": f"Model {model_name} unloaded successfully",
                "loaded": False,
            }
        )

    except Exception as e:
        current_app.logger.error(f"Failed to unload model {model_name}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/api/models/<model_name>/switch", methods=["POST"])
def switch_model(model_name):
    """Switch active model."""
    try:
        model_registry = current_app.model_registry
        model_registry.switch_model(model_name)

        # Emit socket event to notify frontend of model switch
        socketio = current_app.extensions.get("socketio")
        if socketio:
            socketio.emit(
                "model_switched", {"model_id": model_name, "status": "active"}
            )

        return jsonify(
            {
                "success": True,
                "message": f"Switched to model {model_name}",
                "active": True,
            }
        )

    except Exception as e:
        current_app.logger.error(f"Failed to switch to model {model_name}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/api/models/search", methods=["GET"])
def search_huggingface_models():
    """Search HuggingFace for available models."""
    try:
        query = request.args.get("q", "").strip()
        if not query:
            return jsonify({"success": False, "error": "Search query is required"}), 400

        from huggingface_hub import HfApi, list_repo_files

        api = HfApi()

        # Search for models
        models = api.list_models(
            search=query,
            filter=["gguf", "pytorch"],  # Focus on models likely to work
            limit=10,
            sort="downloads",
        )

        results = []
        for model in models:
            try:
                # Get files in the repository
                files = list_repo_files(model.id, token=os.getenv("HF_TOKEN"))

                # Find GGUF files and other model files
                gguf_files = [f for f in files if f.endswith(".gguf")]
                other_files = [
                    f for f in files if f.endswith((".safetensors", ".bin", ".pt"))
                ]

                if gguf_files or other_files:
                    # Extract quantization info from GGUF files
                    quantizations = []
                    for gguf_file in gguf_files:
                        # Extract quantization from filename (e.g., Q4_K_M, Q5_0, etc.)
                        parts = gguf_file.lower().split(".")
                        for part in parts:
                            if any(
                                q in part
                                for q in ["q4", "q5", "q6", "q8", "f16", "f32"]
                            ):
                                size_mb = "Unknown"
                                try:
                                    # Try to get file size (this might fail for private repos)
                                    file_info = api.get_paths_info(
                                        model.id, [gguf_file]
                                    )
                                    if file_info and len(file_info) > 0:
                                        size_bytes = (
                                            file_info[0].size
                                            if hasattr(file_info[0], "size")
                                            else None
                                        )
                                        if size_bytes:
                                            size_mb = (
                                                f"{size_bytes / (1024*1024):.0f}MB"
                                            )
                                except Exception:
                                    pass

                                quantizations.append(
                                    {
                                        "name": part.upper(),
                                        "file": gguf_file,
                                        "size": size_mb,
                                    }
                                )
                                break

                    # If no GGUF files, check if it's convertible
                    if not quantizations and other_files:
                        quantizations.append(
                            {
                                "name": "Convert to GGUF",
                                "file": "convert",
                                "size": "Will convert",
                            }
                        )

                    if quantizations:
                        results.append(
                            {
                                "id": model.id,
                                "name": model.id.split("/")[-1],
                                "full_name": model.id,
                                "downloads": getattr(model, "downloads", 0),
                                "description": (
                                    getattr(model, "card_data", {}).get(
                                        "description", ""
                                    )
                                    if hasattr(model, "card_data") and model.card_data
                                    else ""
                                ),
                                "quantizations": quantizations[
                                    :5
                                ],  # Limit to top 5 quantizations
                            }
                        )

            except Exception as e:
                current_app.logger.warning(
                    f"Failed to get files for model {model.id}: {e}"
                )
                continue

        return jsonify(
            {"success": True, "models": results[:10]}  # Limit to top 10 results
        )

    except Exception as e:
        current_app.logger.error(f"Failed to search HuggingFace: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/api/models/download", methods=["POST"])
def download_model():
    """Download a specific model file from HuggingFace."""
    try:
        data = request.json
        model_name = data.get("model_name")
        specific_file = data.get("file")  # Specific file to download
        if model_name and (".." in model_name or model_name.startswith("/")):
            return jsonify({"success": False, "error": "Invalid model name"}), 400
        if specific_file and (".." in specific_file or specific_file.startswith("/")):
            return jsonify({"success": False, "error": "Invalid file name"}), 400
        quantization = data.get("quantization", "Q4_K_M")  # Fallback quantization

        if not model_name:
            return jsonify({"success": False, "error": "Model name is required"}), 400

        # Get the current app for context (fix Flask context issue)
        app = current_app._get_current_object()

        # Start download in background
        def download_task():
            try:
                # Import download function
                import sys
                from pathlib import Path

                from huggingface_hub import hf_hub_download

                app.logger.info(
                    f"Starting download of {model_name}, file: {specific_file}"
                )

                if specific_file and specific_file != "convert":
                    # Download specific file
                    try:
                        local_file = hf_hub_download(
                            repo_id=model_name,
                            filename=specific_file,
                            token=os.getenv("HF_TOKEN"),
                            local_dir=f"./models/{model_name.replace('/', '_')}",
                        )
                        model_path = Path(local_file)
                        app.logger.info(f"Downloaded specific file to: {model_path}")

                    except Exception as e:
                        app.logger.error(f"Failed to download specific file: {e}")
                        return
                else:
                    # Fallback to old method for conversion
                    project_root = Path(__file__).parent.parent.parent
                    sys.path.insert(0, str(project_root))

                    from download_model import download_model as dl_model

                    model_path = dl_model(model_name, quantization=quantization)

                if model_path and model_path.exists():
                    app.logger.info(
                        f"Model {model_name} downloaded successfully to {model_path}"
                    )

                    # Since we now have dynamic model discovery, just reload the registry
                    # No need to update config files manually
                    with app.app_context():
                        try:
                            app.model_registry.reload_models()
                            app.logger.info(
                                "Model registry reloaded, new model should be available"
                            )
                        except Exception as e:
                            app.logger.error(f"Failed to reload model registry: {e}")
                else:
                    app.logger.error(
                        f"Download completed but model file not found: {model_path}"
                    )

            except Exception as e:
                app.logger.error(f"Download error for model {model_name}: {e}")

        # Start download in background thread
        thread = threading.Thread(target=download_task)
        thread.daemon = True
        thread.start()

        return jsonify(
            {
                "success": True,
                "message": f"Started downloading {model_name} ({specific_file if specific_file else 'auto-convert'}). The model will appear in the dropdown once download completes.",
            }
        )

    except Exception as e:
        current_app.logger.error(f"Failed to start download: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/api/models/refresh", methods=["POST"])
def refresh_models():
    """Manually refresh the model registry."""
    try:
        current_app.model_registry.reload_models()
        return jsonify({"success": True, "message": "Models refreshed successfully"})
    except Exception as e:
        current_app.logger.error(f"Failed to refresh models: {e}")
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
            "description": "Conversational model, good for chat",
        },
        {
            "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "display_name": "TinyLlama 1.1B Chat",
            "size": "2.2GB",
            "description": "Small but capable chat model",
        },
        {
            "name": "microsoft/DialoGPT-small",
            "display_name": "DialoGPT Small",
            "size": "500MB",
            "description": "Lightweight conversational model",
        },
    ]

    return jsonify(available_models)
