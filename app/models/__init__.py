from __future__ import annotations

from typing import Any, Dict, Iterator, List


class CompletionChunk(Dict[str, Any]):
    """Simple dict subclass representing a completion chunk."""


class BaseModelRunner:
    """Minimal base class for model runners."""

    def __init__(self, **config: Any) -> None:
        self.config = config
        self.is_loaded = False

    def load(self) -> None:
        self.is_loaded = True

    def unload(self) -> None:
        self.is_loaded = False

    def chat(self, messages: List[Dict[str, Any]], stream: bool = False, **kwargs: Any) -> Iterator[CompletionChunk]:
        """Return a dummy completion."""
        chunk: CompletionChunk = {
            "choices": [{"message": {"content": ""}}]
        }
        if stream:
            yield chunk
        else:
            yield chunk

    def get_model_info(self) -> Dict[str, Any]:
        return {"engine": "dummy"}


class LlamaCppRunner(BaseModelRunner):
    """Placeholder runner used for tests."""
    pass
