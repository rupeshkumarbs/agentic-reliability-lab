"""Claude via the official Anthropic Python SDK (Claude API or Amazon Bedrock).

- Claude API: ``anthropic.Anthropic()`` resolves credentials from the environment
  (ANTHROPIC_API_KEY, or an ``ant auth login`` profile).
- Bedrock: ``anthropic.AnthropicBedrockMantle(aws_region=...)``; model ids take an
  ``anthropic.`` prefix.

Opus 5.5 runs adaptive thinking (it cannot be disabled), so assistant content is
kept as returned (thinking blocks included) and echoed back unchanged on the next
turn. Effort is set explicitly because Opus 5.5 defaults to "medium".
"""

from __future__ import annotations

import time

from .base import ModelResponse, ProviderError, TransientError


class ClaudeProvider:
    def __init__(self, platform: str = "anthropic", aws_region: str | None = None, effort: str = "medium") -> None:
        import anthropic  # optional dependency: pip install anthropic

        self._anthropic = anthropic
        self.platform = platform
        self.name = f"claude-{platform}"
        self.effort = effort
        if platform == "bedrock":
            self.client = anthropic.AnthropicBedrockMantle(aws_region=aws_region or "ap-south-1")
        else:
            self.client = anthropic.Anthropic()

    def _model_id(self, model: str) -> str:
        return f"anthropic.{model}" if self.platform == "bedrock" and not model.startswith("anthropic.") else model

    def complete(self, *, model: str, system: str, messages: list[dict], tools: list[dict], max_tokens: int) -> ModelResponse:
        a = self._anthropic
        is_haiku = model.removeprefix("anthropic.") == "claude-haiku-4-5"
        params = dict(model=self._model_id(model), max_tokens=max_tokens, system=system, messages=messages, tools=tools)
        if not is_haiku:  # Haiku 4.5 has no effort control
            params["output_config"] = {"effort": self.effort}
        started = time.perf_counter()
        try:
            if self.platform == "anthropic" and not is_haiku:
                # Server-side refusal fallback, routed by refusal category (Claude API only).
                response = self.client.beta.messages.create(
                    **params, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
                )
            else:
                response = self.client.messages.create(**params)
        except (a.RateLimitError, a.InternalServerError, a.APIConnectionError) as exc:
            raise TransientError(str(exc)) from exc
        except a.APIStatusError as exc:
            if exc.status_code >= 500:
                raise TransientError(str(exc)) from exc
            raise ProviderError(f"{exc.status_code}: {exc.message}") from exc

        return ModelResponse(
            model=response.model,
            content=[block.model_dump(mode="json", exclude_none=True) for block in response.content],
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
