"""Model gateway: one choke point for every model call.

- routes   : logical route name -> ordered model list (primary, then fallbacks)
- retries  : exponential backoff on transient errors
- breaker  : per-model circuit breaker; an open breaker skips straight to the fallback
- cache    : exact-match response cache (deterministic replays, eval reruns)
- budget   : hard per-session USD ceiling, checked before every call
- ledger   : every call recorded with tokens, cost and latency (FinOps)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Callable

from .pricing import Ledger, LedgerEntry, cost_usd
from .providers.base import ModelResponse, Provider, TransientError
from .tracing import Tracer

DEFAULT_ROUTES: dict[str, list[str]] = {
    "agent": ["claude-opus-5-5", "claude-sonnet-5-5"],
    "agent-economy": ["claude-haiku-4-5", "claude-sonnet-5-5"],
    "judge": ["claude-opus-5-5"],
}


class BudgetExceeded(Exception):
    pass


class AllModelsFailed(Exception):
    pass


@dataclass
class _Breaker:
    failures: int = 0
    opened_at: float | None = None


@dataclass
class ModelGateway:
    provider: Provider
    routes: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_ROUTES))
    ledger: Ledger = field(default_factory=Ledger)
    session_budget_usd: float = 0.50
    max_retries: int = 2
    backoff_s: float = 0.2
    breaker_threshold: int = 3
    breaker_cooldown_s: float = 30.0
    use_cache: bool = False
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    _breakers: dict[str, _Breaker] = field(default_factory=dict)
    _cache: dict[str, ModelResponse] = field(default_factory=dict)

    def _breaker_open(self, model: str) -> bool:
        b = self._breakers.get(model)
        if not b or b.opened_at is None:
            return False
        if self.clock() - b.opened_at >= self.breaker_cooldown_s:
            b.opened_at, b.failures = None, 0  # half-open: allow a trial call
            return False
        return True

    def _record_failure(self, model: str) -> None:
        b = self._breakers.setdefault(model, _Breaker())
        b.failures += 1
        if b.failures >= self.breaker_threshold:
            b.opened_at = self.clock()

    def complete(
        self, *, route: str, session: str, system: str, messages: list[dict], tools: list[dict],
        tracer: Tracer, max_tokens: int = 16000,
    ) -> ModelResponse:
        if self.ledger.session_cost(session) >= self.session_budget_usd:
            raise BudgetExceeded(f"session {session} reached {self.session_budget_usd:.2f} USD")

        key = hashlib.sha256(json.dumps([route, system, messages, tools], sort_keys=True).encode()).hexdigest()
        if self.use_cache and key in self._cache:
            cached = self._cache[key]
            self.ledger.record(LedgerEntry(session, route, cached.model, 0, 0, 0.0, 0.0, cached=True))
            tracer.event("gateway.cache_hit", route=route)
            return cached

        for model in self.routes[route]:
            if self._breaker_open(model):
                tracer.event("gateway.breaker_open", model=model)
                continue
            for attempt in range(self.max_retries + 1):
                with tracer.span("llm.call", **{"gen_ai.request.model": model, "route": route, "attempt": attempt}) as span:
                    try:
                        resp = self.provider.complete(model=model, system=system, messages=messages, tools=tools, max_tokens=max_tokens)
                    except TransientError as exc:
                        span.status = "error"
                        span.attributes["error"] = str(exc)[:80]
                        self._record_failure(model)
                        if attempt < self.max_retries and not self._breaker_open(model):
                            self.sleep(self.backoff_s * 2 ** attempt)
                            continue
                        break
                    cost = cost_usd(model, resp.input_tokens, resp.output_tokens)
                    span.attributes.update({
                        "gen_ai.usage.input_tokens": resp.input_tokens,
                        "gen_ai.usage.output_tokens": resp.output_tokens,
                        "gen_ai.response.finish_reason": resp.stop_reason,
                        "cost_usd": round(cost, 6),
                        "latency_ms": round(resp.latency_ms),
                    })
                self._breakers.pop(model, None)
                self.ledger.record(LedgerEntry(session, route, model, resp.input_tokens, resp.output_tokens, cost, resp.latency_ms))
                if self.use_cache:
                    self._cache[key] = resp
                return resp
            tracer.event("gateway.fallback", from_model=model)
        raise AllModelsFailed(f"every model on route {route!r} failed")
