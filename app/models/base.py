"""Base model interface for inference engines."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, TypedDict


class Message(TypedDict):
    """Chat message format."""
    role: str
    content: str


class CompletionChunk(TypedDict):
    """Streaming completion chunk format."""
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]


class ModelConfig(TypedDict, total=False):
    """Model configuration."""
    path: str
    engine: str
    gpu_layers: int
    n_gpu_layers: int
    max_total_tokens: int
    rope_freq_base: int
    temperature: float
    top_p: float
    top_k: int


class BaseModelRunner(ABC):
    """Abstract base class for model runners.
    
    All inference engines must implement this interface.
    """
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the model runner.
        
        Args:
            **kwargs: Model-specific configuration.
        """
        self.config = kwargs
        self.model_id = kwargs.get("model_id", "unknown")
        self.is_loaded = False
    
    @abstractmethod
    def load(self) -> None:
        """Load the model into memory.
        
        Raises:
            RuntimeError: If model loading fails.
        """
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload the model from memory."""
        pass
    
    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        stream: bool = False,
        **kwargs: Any
    ) -> Iterator[CompletionChunk]:
        """Generate chat completions.
        
        Args:
            messages: List of chat messages.
            stream: Whether to stream the response.
            **kwargs: Additional generation parameters.
            
        Yields:
            Completion chunks if streaming, otherwise single completion.
        """
        pass
    
    @abstractmethod
    async def achat(
        self,
        messages: List[Message],
        stream: bool = False,
        **kwargs: Any
    ) -> AsyncIterator[CompletionChunk]:
        """Async version of chat.
        
        Args:
            messages: List of chat messages.
            stream: Whether to stream the response.
            **kwargs: Additional generation parameters.
            
        Yields:
            Completion chunks if streaming, otherwise single completion.
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Model metadata including name, parameters, context length, etc.
        """
        pass
    
    def validate_messages(self, messages: List[Message]) -> None:
        """Validate message format.
        
        Args:
            messages: Messages to validate.
            
        Raises:
            ValueError: If messages are invalid.
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        for msg in messages:
            if "role" not in msg or "content" not in msg:
                raise ValueError("Each message must have 'role' and 'content'")
            
            if msg["role"] not in ["system", "user", "assistant", "function"]:
                raise ValueError(f"Invalid role: {msg['role']}") 