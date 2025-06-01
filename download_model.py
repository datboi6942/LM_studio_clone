"""Download and convert models from HuggingFace."""

import argparse
import os
import subprocess
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download


def download_model(
    repo_id: str,
    revision: str = "main",
    quantization: str = "Q4_K_M",
    token: Optional[str] = None
) -> Path:
    """Download model from HuggingFace.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., "TheBloke/Llama-2-7B-GGUF").
        revision: Git revision to download.
        quantization: Quantization type for GGUF conversion.
        token: HuggingFace API token.
        
    Returns:
        Path to downloaded/converted model.
    """
    # Get HF token from env if not provided
    if token is None:
        token = os.getenv("HF_TOKEN")
    
    print(f"Downloading {repo_id} (revision: {revision})...")
    
    # Download model files
    local_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=["*.gguf", "*.safetensors", "*.bin", "config.json", "tokenizer*"],
        token=token,
        local_dir=f"./models/{repo_id.replace('/', '_')}"
    )
    
    local_path = Path(local_path)
    print(f"Downloaded to: {local_path}")
    
    # Check if GGUF file exists
    gguf_files = list(local_path.glob("*.gguf"))
    if gguf_files:
        print(f"Found GGUF file: {gguf_files[0]}")
        return gguf_files[0]
    
    # Check for safetensors/bin files to convert
    model_files = list(local_path.glob("*.safetensors")) + list(local_path.glob("*.bin"))
    if not model_files:
        print("No model files found to convert")
        return local_path
    
    # Convert to GGUF
    print(f"Converting to GGUF with quantization {quantization}...")
    output_file = local_path / f"{repo_id.split('/')[-1]}-{quantization}.gguf"
    
    try:
        subprocess.run([
            "python", "-m", "llama_cpp.convert",
            "--outtype", quantization.lower(),
            "--outfile", str(output_file),
            str(local_path)
        ], check=True)
        
        print(f"Converted to: {output_file}")
        return output_file
        
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {e}")
        return local_path


def main() -> None:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Download and convert models from HuggingFace"
    )
    parser.add_argument(
        "repo",
        help="HuggingFace repository ID (e.g., TheBloke/Llama-2-7B-GGUF)"
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Git revision to download (default: main)"
    )
    parser.add_argument(
        "--quantization",
        default="Q4_K_M",
        help="Quantization type for GGUF conversion (default: Q4_K_M)"
    )
    parser.add_argument(
        "--token",
        help="HuggingFace API token (or set HF_TOKEN env var)"
    )
    
    args = parser.parse_args()
    
    model_path = download_model(
        repo_id=args.repo,
        revision=args.revision,
        quantization=args.quantization,
        token=args.token
    )
    
    print(f"\nModel ready at: {model_path}")
    print("\nTo use this model, add it to app/config_models.yaml:")
    print(f"""
{args.repo.split('/')[-1].lower()}:
  path: {model_path}
  engine: llama_cpp
  n_gpu_layers: 35  # Adjust based on your GPU
""")


if __name__ == "__main__":
    main() 