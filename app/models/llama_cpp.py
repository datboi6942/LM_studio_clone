"""Llama.cpp model runner implementation."""

import json
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import structlog
from llama_cpp import Llama

from .base import BaseModelRunner, CompletionChunk, Message

logger = structlog.get_logger(__name__)


class LlamaCppRunner(BaseModelRunner):
    """Runner for llama-cpp-python models (GGUF format)."""
    
    def __init__(
        self,
        path: str,
        n_gpu_layers: int = 0,
        n_ctx: int = 4096,
        n_batch: int = 512,
        rope_freq_base: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        """Initialize llama-cpp runner.
        
        Args:
            path: Path to GGUF model file.
            n_gpu_layers: Number of layers to offload to GPU.
            n_ctx: Context window size.
            n_batch: Batch size for prompt processing.
            rope_freq_base: RoPE frequency base (model-specific).
            **kwargs: Additional configuration.
        """
        super().__init__(**kwargs)
        self.model_path = Path(path)
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.rope_freq_base = rope_freq_base
        self.llm: Optional[Llama] = None
    
    def load(self) -> None:
        """Load the GGUF model into memory."""
        if self.is_loaded:
            return
        
        logger.info("Loading GGUF model", path=str(self.model_path))
        
        # Validate model file exists and is readable
        if not self.model_path.exists():
            raise RuntimeError(f"Model file not found: {self.model_path}")
        
        if not self.model_path.is_file():
            raise RuntimeError(f"Model path is not a file: {self.model_path}")
        
        try:
            # Use conservative parameters to avoid crashes
            kwargs: Dict[str, Any] = {
                "model_path": str(self.model_path),
                "n_gpu_layers": max(0, self.n_gpu_layers),  # Ensure non-negative
                "n_ctx": max(512, min(8192, self.n_ctx)),  # Clamp context size
                "n_batch": max(1, min(2048, self.n_batch)),  # Clamp batch size
                "verbose": False,
                "use_mmap": True,  # Use memory mapping for efficiency
                "use_mlock": False,  # Don't lock memory (can cause issues)
                "n_threads": None,  # Let llama.cpp decide
            }
            
            # Only add rope_freq_base if it's a reasonable value
            if self.rope_freq_base and self.rope_freq_base > 0:
                kwargs["rope_freq_base"] = float(self.rope_freq_base)
            
            logger.info("Loading model with parameters", **kwargs)
            
            self.llm = Llama(**kwargs)
            self.is_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error("Failed to load model", error=str(e), path=str(self.model_path))
            if self.llm:
                del self.llm
                self.llm = None
            self.is_loaded = False
            raise RuntimeError(f"Failed to load model: {e}")
    
    def unload(self) -> None:
        """Unload the model from memory."""
        if self.llm:
            del self.llm
            self.llm = None
            self.is_loaded = False
            logger.info("Model unloaded")
    
    def chat(
        self,
        messages: List[Message],
        stream: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        max_tokens: int = 2048,
        **kwargs: Any
    ) -> Iterator[CompletionChunk]:
        """Generate chat completions.
        
        Args:
            messages: Chat messages.
            stream: Whether to stream response.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional parameters.
            
        Yields:
            Completion chunks.
        """
        logger.info("🚀 CHAT METHOD CALLED", model_id=getattr(self, 'model_id', 'unknown'), stream=stream)
        
        if not self.is_loaded or not self.llm:
            logger.error("💥 Model not loaded when chat called", is_loaded=self.is_loaded, llm=bool(self.llm))
            raise RuntimeError("Model not loaded")
        
        try:
            # Add comprehensive debugging
            logger.info("=== CHAT METHOD DEBUG START ===")
            logger.info("Parameters", 
                       stream=stream, 
                       temperature=temperature, 
                       top_p=top_p, 
                       top_k=top_k, 
                       max_tokens=max_tokens,
                       messages_count=len(messages))
            logger.info("Messages", messages=messages)
            logger.info("Model path", path=str(self.model_path))
            logger.info("Model loaded", is_loaded=self.is_loaded)
            
            self.validate_messages(messages)
            
            # Convert messages to prompt
            prompt = self._format_messages(messages)
            logger.info("Generated prompt for chat", prompt_length=len(prompt), prompt=prompt[:200] + "..." if len(prompt) > 200 else prompt)
            
            # Validate parameters
            temperature = max(0.01, min(2.0, temperature))  # Clamp to valid range
            top_p = max(0.01, min(1.0, top_p))  # Clamp to valid range
            top_k = max(1, min(100, top_k))  # Clamp to valid range
            max_tokens = max(1, min(4096, max_tokens))  # Clamp to valid range
            
            # Generate completion
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created = int(time.time())
            
            logger.info("Starting completion", 
                       stream=stream, 
                       temperature=temperature, 
                       top_p=top_p, 
                       top_k=top_k, 
                       max_tokens=max_tokens,
                       completion_id=completion_id)
            
            # Use simple text completion instead of chat completion to avoid crashes
            if stream:
                try:
                    logger.info("Starting streaming generation")
                    
                    # Try to use create_completion method first (more reliable)
                    try:
                        logger.debug("Trying create_completion method for streaming")
                        stream_iter = self.llm.create_completion(
                            prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            stream=True,
                            echo=False,
                            stop=["</s>", "<|end|>", "<|endoftext|>", "\n\nUser:", "\n\nHuman:"]
                        )
                        logger.info("create_completion stream created successfully")
                    except Exception as e:
                        logger.warning(f"create_completion failed, falling back to __call__: {e}")
                        # Fallback to the simple call method
                        stream_iter = self.llm(
                            prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            stream=True,
                            echo=False,
                            stop=["</s>", "<|end|>", "<|endoftext|>", "\n\nUser:", "\n\nHuman:"]
                        )
                        logger.info("__call__ stream created successfully")
                    
                    chunk_count = 0
                    total_content = ""
                    has_generated_content = False
                    
                    logger.info("Starting to iterate over stream...")
                    
                    for chunk in stream_iter:
                        chunk_count += 1
                        logger.info(f"=== CHUNK {chunk_count} ===")
                        logger.info("Raw chunk", chunk=chunk)
                        
                        # Extract text from chunk
                        text = ""
                        finish_reason = None
                        
                        if isinstance(chunk, dict):
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                choice = chunk["choices"][0]
                                text = choice.get("text", "")
                                finish_reason = choice.get("finish_reason")
                                logger.info("Extracted from choices", text=f"'{text}'", finish_reason=finish_reason)
                            elif "text" in chunk:
                                text = chunk["text"]
                                logger.info("Extracted from text field", text=f"'{text}'")
                        
                        if text:
                            total_content += text
                            has_generated_content = True
                            logger.info(f"✅ Generated text: '{text}' (chunk #{chunk_count}, total chars: {len(total_content)})")
                        else:
                            logger.warning(f"❌ Chunk {chunk_count} has no text content")
                        
                        # Format and yield the chunk
                        formatted_chunk = self._format_chunk(chunk, completion_id, created)
                        logger.info("Formatted chunk", formatted_chunk=formatted_chunk)
                        yield formatted_chunk
                        
                        # Check if we should stop
                        if finish_reason and finish_reason != "null":
                            logger.info(f"🛑 Stopping due to finish reason: {finish_reason}")
                            break
                    
                    logger.info(f"Stream iteration completed: {chunk_count} chunks, {len(total_content)} chars, has_content: {has_generated_content}")
                    
                    # If no content was generated, try non-streaming as fallback
                    if not has_generated_content:
                        logger.warning("❌ No content generated during streaming, trying non-streaming fallback")
                        try:
                            logger.info("Attempting non-streaming fallback...")
                            # Try non-streaming completion
                            response = self.llm.create_completion(
                                prompt,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                top_k=top_k,
                                stream=False,
                                echo=False,
                                stop=["</s>", "<|end|>", "<|endoftext|>", "\n\nUser:", "\n\nHuman:"]
                            )
                            
                            logger.info("Non-streaming response", response=response)
                            
                            if response.get("choices") and response["choices"][0].get("text", "").strip():
                                text = response["choices"][0]["text"].strip()
                                logger.info(f"✅ Non-streaming fallback generated: '{text[:100]}...'")
                                
                                # Yield this as a streaming chunk
                                yield {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": self.model_id,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {
                                            "content": text
                                        },
                                        "finish_reason": "stop"
                                    }]
                                }
                            else:
                                logger.error("❌ Even non-streaming fallback failed to generate content")
                                # Yield helpful error message
                                yield {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": self.model_id,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {
                                            "content": "I'm having trouble generating a response. This might be due to the model configuration or prompt format. Please try:\n1. Using a different prompt\n2. Adjusting the temperature settings\n3. Switching to a different model"
                                        },
                                        "finish_reason": "stop"
                                    }]
                                }
                                
                        except Exception as fallback_error:
                            logger.error("Non-streaming fallback failed", error=str(fallback_error), exc_info=True)
                            yield {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": self.model_id,
                                "choices": [{
                                    "index": 0,
                                    "delta": {
                                        "content": f"Model error: {str(fallback_error)}"
                                    },
                                    "finish_reason": "error"
                                }]
                            }
                    
                    logger.info("=== CHAT METHOD DEBUG END (STREAMING) ===")
                    
                except Exception as e:
                    logger.error("Streaming completion error", error=str(e), exc_info=True)
                    # Yield error message as completion
                    yield {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": self.model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "content": f"Error during generation: {str(e)}"
                            },
                            "finish_reason": "error"
                        }]
                    }
            else:
                try:
                    # Use the simple call method without streaming
                    response = self.llm(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        stream=False,
                        echo=False,  # Don't echo the prompt
                        # Don't use stop tokens for now to avoid issues
                    )
                    
                    yield self._format_completion(response, completion_id, created)
                    
                except Exception as e:
                    logger.error("Non-streaming completion error", error=str(e))
                    # Yield error message as completion
                    yield {
                        "id": completion_id,
                        "object": "chat.completion",
                        "created": created,
                        "model": self.model_id,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": f"Error: {str(e)}"
                            },
                            "finish_reason": "error"
                        }]
                    }
                    
        except Exception as e:
            logger.error("Chat method error", error=str(e))
            # Yield error message as completion
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created = int(time.time())
            
            yield {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": self.model_id,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Error: {str(e)}"
                    },
                    "finish_reason": "error"
                }]
            }
    
    async def achat(
        self,
        messages: List[Message],
        stream: bool = False,
        **kwargs: Any
    ) -> AsyncIterator[CompletionChunk]:
        """Async chat generation (uses sync version internally)."""
        # For now, we'll use the sync version
        # TODO: Implement true async with thread pool
        for chunk in self.chat(messages, stream=stream, **kwargs):
            yield chunk
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "id": self.model_id,
            "path": str(self.model_path),
            "engine": "llama-cpp",
            "loaded": self.is_loaded,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
        }
    
    def _format_messages(self, messages: List[Message]) -> str:
        """Format messages into a prompt string.
        
        Args:
            messages: Chat messages.
            
        Returns:
            Formatted prompt.
        """
        # Check model type based on model path
        is_tinyllama = "tinyllama" in str(self.model_path).lower()
        is_phi = "phi" in str(self.model_path).lower()
        
        if is_tinyllama:
            # Use TinyLlama's chat template format
            prompt = "<|system|>\nYou are a helpful AI assistant.</s>\n"
            
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                
                if role == "system":
                    prompt = f"<|system|>\n{content}</s>\n"
                elif role == "user":
                    prompt += f"<|user|>\n{content}</s>\n"
                elif role == "assistant":
                    prompt += f"<|assistant|>\n{content}</s>\n"
            
            # Add the assistant prompt marker
            prompt += "<|assistant|>\n"
        elif is_phi:
            # Use Phi-3's chat template format
            prompt = "<|system|>\nYou are a helpful AI assistant.<|end|>\n"
            
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                
                if role == "system":
                    prompt = f"<|system|>\n{content}<|end|>\n"
                elif role == "user":
                    prompt += f"<|user|>\n{content}<|end|>\n"
                elif role == "assistant":
                    prompt += f"<|assistant|>\n{content}<|end|>\n"
            
            # Add the assistant prompt marker
            prompt += "<|assistant|>\n"
        else:
            # Use a simple but effective format for other models
            prompt = ""
            
            # Add a system prompt if not present
            has_system = any(msg["role"] == "system" for msg in messages)
            if not has_system:
                prompt += "You are a helpful AI assistant.\n\n"
            
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                
                if role == "system":
                    prompt += f"{content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n"
            
            # Add the assistant prompt to encourage a response
            prompt += "Assistant:"
        
        logger.debug(f"Generated prompt:\n{prompt}")
        return prompt
    
    def _format_chunk(
        self,
        chunk: Dict[str, Any],
        completion_id: str,
        created: int
    ) -> CompletionChunk:
        """Format streaming chunk."""
        # Log the raw chunk for debugging
        logger.debug("Raw chunk from llama.cpp", chunk=chunk)
        
        # Extract text content from the chunk
        text = ""
        finish_reason = None
        
        # Handle different chunk formats
        if isinstance(chunk, dict):
            if "choices" in chunk and len(chunk["choices"]) > 0:
                choice = chunk["choices"][0]
                text = choice.get("text", "")
                finish_reason = choice.get("finish_reason")
            elif "text" in chunk:
                # Direct text format
                text = chunk["text"]
            elif "content" in chunk:
                # Direct content format
                text = chunk["content"]
        
        # Create the formatted response
        formatted = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.model_id,
            "choices": [{
                "index": 0,
                "delta": {
                    "content": text
                },
                "finish_reason": finish_reason
            }]
        }
        
        # Log what we're sending
        if text:
            logger.debug(f"Formatted chunk with text: '{text[:50]}...' (length: {len(text)})")
        else:
            logger.debug("Formatted chunk with no text", finish_reason=finish_reason)
        
        return formatted
    
    def _format_completion(
        self,
        response: Dict[str, Any],
        completion_id: str,
        created: int
    ) -> CompletionChunk:
        """Format non-streaming completion."""
        # llama-cpp-python returns the response in the correct format already
        # We just need to convert it to the OpenAI chat completion format
        text = ""
        finish_reason = "stop"
        if response.get("choices") and len(response["choices"]) > 0:
            text = response["choices"][0].get("text", "")
            finish_reason = response["choices"][0].get("finish_reason", "stop")
        
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": self.model_id,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text
                },
                "finish_reason": finish_reason
            }],
            "usage": response.get("usage", {})
        } 