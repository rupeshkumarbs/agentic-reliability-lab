from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ProviderError(Exception):
    """Non-retryable provider failure (bad request, auth, unknown model)."""


class TransientError(ProviderError):
    """Retryable failure: rate limit, overload, 5xx, network."""


@dataclass
class ModelResponse:
    model: str
    content: list[dict]  # Messages API content blocks, echoed back unchanged in history
    stop_reason: str
    input_tokens: int
    output_tokens: int
    latency_ms: float

    @property
    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.content if b.get("type") == "text")

    @property
    def tool_calls(self) -> list[dict]:
        return [b for b in self.content if b.get("type") == "tool_use"]


class Provider(Protocol):
    name: str

    def complete(
        self, *, model: str, system: str, messages: list[dict], tools: list[dict], max_tokens: int
    ) -> ModelResponse: ...
