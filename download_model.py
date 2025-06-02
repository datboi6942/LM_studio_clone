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
        # For multiple GGUF files, try to find a good one or pick the first
        chosen_gguf = gguf_files[0]
        for gguf_file in gguf_files:
            # Prefer Q4_K_M or similar quantization if available
            if any(quant in gguf_file.name.lower() for quant in ['q4_k_m', 'q4_k', 'q4']):
                chosen_gguf = gguf_file
                break
        
        print(f"Found GGUF file: {chosen_gguf}")
        return chosen_gguf
    
    # Check for safetensors/bin files to convert
    model_files = list(local_path.glob("*.safetensors")) + list(local_path.glob("*.bin"))
    if not model_files:
        print("No model files found to convert")
        return local_path
    
    # Try to convert to GGUF
    print(f"Converting to GGUF with quantization {quantization}...")
    output_file = local_path / f"{repo_id.split('/')[-1]}-{quantization}.gguf"
    
    # Try different conversion methods
    conversion_successful = False
    
    # Method 1: Try llama_cpp.convert
    try:
        print("Attempting conversion with llama_cpp.convert...")
        subprocess.run([
            "python", "-m", "llama_cpp.convert",
            "--outtype", quantization.lower(),
            "--outfile", str(output_file),
            str(local_path)
        ], check=True)
        
        print(f"Converted to: {output_file}")
        conversion_successful = True
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"llama_cpp.convert failed: {e}")
    
    # Method 2: Try alternative conversion if available
    if not conversion_successful:
        try:
            print("Attempting conversion with alternative method...")
            # Check if we have convert.py in llama.cpp
            import llama_cpp
            llama_cpp_path = Path(llama_cpp.__file__).parent
            convert_script = llama_cpp_path / "convert.py"
            
            if convert_script.exists():
                subprocess.run([
                    "python", str(convert_script),
                    str(local_path),
                    "--outtype", quantization.lower(),
                    "--outfile", str(output_file)
                ], check=True)
                
                print(f"Converted to: {output_file}")
                conversion_successful = True
                
        except Exception as e:
            print(f"Alternative conversion failed: {e}")
    
    # If conversion failed, just return the directory with the original files
    if not conversion_successful:
        print("Conversion failed, but model files are available for manual setup")
        print(f"Original files at: {local_path}")
        
        # Try to find the main model file to return
        main_file = None
        for pattern in ["pytorch_model.bin", "model.safetensors", "*.bin", "*.safetensors"]:
            matches = list(local_path.glob(pattern))
            if matches:
                main_file = matches[0]
                break
        
        return main_file if main_file else local_path
    
    return output_file


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