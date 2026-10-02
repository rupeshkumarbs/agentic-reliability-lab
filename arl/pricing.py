"""Model price table and per-call cost accounting (FinOps for AI).

Prices are Claude API first-party list prices in USD per 1M tokens (as of
September 2026). Bedrock / Vertex pricing is set by those platforms; override
with ``PRICES`` when routing there.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

PRICES: dict[str, tuple[float, float]] = {
    # model id: (input USD / 1M tokens, output USD / 1M tokens)
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    base = model.removeprefix("anthropic.")  # Bedrock ids carry an "anthropic." prefix
    if base not in PRICES:
        raise KeyError(f"no price for model {model!r}; add it to arl.pricing.PRICES")
    price_in, price_out = PRICES[base]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


@dataclass
class LedgerEntry:
    session: str
    route: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    cached: bool = False


@dataclass
class Ledger:
    """Append-only record of every model call; the source for unit economics."""

    entries: list[LedgerEntry] = field(default_factory=list)

    def record(self, entry: LedgerEntry) -> None:
        self.entries.append(entry)

    def session_cost(self, session: str) -> float:
        return sum(e.cost_usd for e in self.entries if e.session == session)

    def total_cost(self) -> float:
        return sum(e.cost_usd for e in self.entries)

    def by_model(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
        for e in self.entries:
            row = out[e.model]
            row["calls"] += 1
            row["input_tokens"] += e.input_tokens
            row["output_tokens"] += e.output_tokens
            row["cost_usd"] += e.cost_usd
        return dict(out)
