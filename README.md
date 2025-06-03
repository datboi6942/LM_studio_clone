# LLM Cockpit 🚀

A powerful local LLM interface that runs GGUF/Safetensors models with GPU acceleration, real-time streaming, and advanced features beyond typical LLM runners.

## Features

### Core Features (LM Studio Parity)
- ✅ Run GGUF / Safetensors models on CPU + GPU
- ✅ Auto-download and convert from HuggingFace
- ✅ OpenAI-compatible REST API with SSE streaming
- ✅ Multi-model hot-swap capability
- ✅ Serve on LAN with security controls

### Extra Features
- 🔥 Real-time GPU VRAM/temperature dashboard
- 🎯 Function calling & tool-use support
- 🗣️ Voice chat with Whisper STT + TTS (optional)
- 📚 RAG pipeline with persistent vector DB
- 🔌 Plugin system via Python entry-points
- 🤖 Agent orchestration playground
- 🎛️ Fine-grained generation controls (temperature, penalties, etc.) adjustable from the web UI

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-cockpit.git
cd llm-cockpit

# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

## Quick Start

### 1. Download a Model

```bash
# Download a GGUF model from HuggingFace
poetry run download-model TheBloke/Llama-2-7B-Chat-GGUF

# Or with specific quantization
poetry run download-model TheBloke/CodeLlama-13B-GGUF --quantization Q5_K_M
```

### 2. Run the Server

```bash
# Start the server (localhost only)
poetry run llm-cockpit

# Or enable LAN access
poetry run llm-cockpit --public

# With debug mode
poetry run llm-cockpit --debug
```

### 3. Test the API

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2-7b-instruct",
    "stream": true,
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello! Can you explain quantum computing?"}
    ]
  }'
```

## API Documentation

### Chat Completions (OpenAI-compatible)

```
POST /api/v1/chat/completions
```

Request body:
```json
{
  "model": "model-id",
  "messages": [
    {"role": "system", "content": "System prompt"},
    {"role": "user", "content": "User message"}
  ],
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 2048
}
```

### Model Management

```
GET  /api/v1/models              # List available models
POST /api/v1/models/{id}/load    # Load model into memory
POST /api/v1/models/{id}/unload  # Unload model from memory
POST /api/v1/models/switch       # Switch active model
```

### Health Check

```
GET /api/health
```

## Configuration

Configuration can be set via environment variables or `.env` file:

```bash
# .env file
SECRET_KEY=your-secret-key-here
DEBUG=False
HOST=127.0.0.1
PORT=8000
MODELS_DIR=./models
DEFAULT_MODEL=qwen2-7b-instruct
DEFAULT_TEMPERATURE=0.7
DEFAULT_TOP_P=0.95
DEFAULT_TOP_K=40
DEFAULT_MAX_TOKENS=2048
REPEAT_PENALTY=1.1
PRESENCE_PENALTY=0.0
FREQUENCY_PENALTY=0.0
GPU_MONITOR_INTERVAL=1
LOG_LEVEL=INFO
PUBLIC_MODE=False
```

These values serve as defaults. You can adjust them at runtime from the sidebar
"Settings" panel without editing the `.env` file.

## Model Configuration

Models are configured in `app/config_models.yaml`:

```yaml
llama2-7b-chat:
  path: ./models/llama-2-7b-chat.Q4_K_M.gguf
  engine: llama_cpp
  n_gpu_layers: 35
  n_ctx: 4096

codellama-13b:
  path: ./models/codellama-13b-instruct.Q5_K_M.gguf
  engine: llama_cpp
  n_gpu_layers: 40
  rope_freq_base: 1000000
```

## GPU Support

### NVIDIA GPU Setup

```bash
# Install CUDA 12.x
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/\
cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-4

# Compile llama-cpp with CUDA support
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
```

## Development

### Running Tests

```bash
# Run all tests with coverage
poetry run pytest

# Run specific test file
poetry run pytest tests/test_models.py

# Run with verbose output
poetry run pytest -v
```

### Code Quality

```bash
# Format code
poetry run black app tests

# Lint code
poetry run ruff check app tests

# Type check
poetry run mypy app
```

### Building Documentation

```bash
# Generate API docs
poetry run pdoc --html --output-dir docs app
```

## Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-dev

COPY . .
EXPOSE 8000

CMD ["poetry", "run", "llm-cockpit", "--host", "0.0.0.0"]
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Built with Flask, llama-cpp-python, and love
- Inspired by LM Studio, but with more features
- GPU monitoring powered by pynvml
- Vector search by Chroma 