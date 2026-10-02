"""Model providers. All speak the Messages API content-block shape internally."""

from .base import ModelResponse, Provider, ProviderError, TransientError

__all__ = ["ModelResponse", "Provider", "ProviderError", "TransientError", "make_provider"]


def make_provider(name: str, **kwargs) -> Provider:
    if name == "reference":
        from .reference import ReferenceModel

        return ReferenceModel(**kwargs)
    if name in ("anthropic", "bedrock"):
        from .claude import ClaudeProvider

        return ClaudeProvider(platform=name, **kwargs)
    raise ValueError(f"unknown provider {name!r} (reference | anthropic | bedrock)")
