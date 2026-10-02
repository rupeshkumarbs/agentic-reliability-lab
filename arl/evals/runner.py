"""Run the eval suite for one agent version and decide whether it may ship."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..agent import AgentConfig, SupportAgent, deny_all
from ..gateway import ModelGateway
from ..pricing import Ledger
from ..providers import Provider
from ..tools import ROOT, SupportBackend
from ..tracing import Tracer
from .graders import grade

CASES = ROOT / "evals" / "cases.jsonl"


def load_cases(path: Path = CASES) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass
class CaseResult:
    id: str
    category: str
    critical: bool
    passed: bool
    cost_usd: float
    latency_ms: float
    steps: int
    failed_checks: list[str]
    answer: str


@dataclass
class Scorecard:
    agent: str
    provider: str
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return sum(c.passed for c in self.cases) / len(self.cases)

    @property
    def critical_failures(self) -> list[str]:
        return [c.id for c in self.cases if c.critical and not c.passed]

    @property
    def avg_cost_usd(self) -> float:
        return statistics.fmean(c.cost_usd for c in self.cases)

    @property
    def p95_latency_ms(self) -> float:
        ordered = sorted(c.latency_ms for c in self.cases)
        return ordered[max(0, int(round(0.95 * len(ordered))) - 1)]

    def summary(self) -> dict:
        by_cat: dict[str, list[bool]] = {}
        for c in self.cases:
            by_cat.setdefault(c.category, []).append(c.passed)
        return {
            "agent": self.agent,
            "provider": self.provider,
            "cases": len(self.cases),
            "pass_rate": round(self.pass_rate, 4),
            "critical_failures": self.critical_failures,
            "by_category": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_cat.items())},
            "avg_cost_usd": round(self.avg_cost_usd, 6),
            "p95_latency_ms": round(self.p95_latency_ms),
            "avg_steps": round(statistics.fmean(c.steps for c in self.cases), 2),
        }


def run_suite(config: AgentConfig, provider: Provider, *, cases: list[dict] | None = None,
              trace_dir: Path | None = None) -> Scorecard:
    ledger = Ledger()
    gateway = ModelGateway(provider=provider, ledger=ledger, sleep=lambda s: None)
    card = Scorecard(agent=f"{config.name}@{config.version}", provider=provider.name)
    for case in cases or load_cases():
        backend = SupportBackend()
        approver = (lambda *a: True) if case.get("approver") == "approve" else deny_all
        agent = SupportAgent(config, gateway, backend, approver)
        tracer = Tracer(trace_id=case["id"])
        result = agent.run(case["input"], session=case["id"], tracer=tracer)
        cost = ledger.session_cost(case["id"])
        latency = sum(e.latency_ms for e in ledger.entries if e.session == case["id"])
        checks = grade(case["expect"], result, backend, cost)
        card.cases.append(CaseResult(
            id=case["id"], category=case["category"], critical=case.get("critical", False),
            passed=all(c.passed for c in checks), cost_usd=cost, latency_ms=latency, steps=result.steps,
            failed_checks=[f"{c.name} ({c.detail})" for c in checks if not c.passed], answer=result.answer,
        ))
        if trace_dir:
            tracer.export_jsonl(trace_dir / f"{case['id']}.jsonl")
    return card


@dataclass
class GateDecision:
    ship: bool
    reasons: list[str]


def release_gate(card: Scorecard, baseline: dict | None, *, max_pass_rate_drop: float = 0.0,
                 max_cost_increase: float = 0.25) -> GateDecision:
    reasons: list[str] = []
    if card.critical_failures:
        reasons.append(f"critical (safety) cases failed: {', '.join(card.critical_failures)}")
    if baseline:
        if card.pass_rate < baseline["pass_rate"] - max_pass_rate_drop:
            reasons.append(f"pass rate {card.pass_rate:.1%} below baseline {baseline['pass_rate']:.1%}")
        limit = baseline["avg_cost_usd"] * (1 + max_cost_increase)
        if card.avg_cost_usd > limit:
            reasons.append(f"avg cost/case {card.avg_cost_usd:.5f} USD exceeds {limit:.5f} (baseline +{max_cost_increase:.0%})")
    return GateDecision(ship=not reasons, reasons=reasons)


def render_markdown(card: Scorecard, baseline: dict | None, decision: GateDecision) -> str:
    s = card.summary()
    lines = [f"# Eval scorecard: {s['agent']} ({s['provider']})", ""]
    lines.append(f"**Release gate: {'PASS - ship' if decision.ship else 'BLOCKED'}**")
    lines += [f"- {r}" for r in decision.reasons]
    lines += ["", "| Metric | This version | Baseline |", "|---|---|---|"]
    b = baseline or {}
    lines.append(f"| Pass rate | {s['pass_rate']:.1%} | {b['pass_rate']:.1%} |" if b else f"| Pass rate | {s['pass_rate']:.1%} | - |")
    lines.append(f"| Critical failures | {len(s['critical_failures'])} | {len(b.get('critical_failures', [])) if b else '-'} |")
    lines.append(f"| Avg cost / case (USD) | {s['avg_cost_usd']:.5f} | {b['avg_cost_usd']:.5f} |" if b else f"| Avg cost / case (USD) | {s['avg_cost_usd']:.5f} | - |")
    lines.append(f"| p95 latency (ms) | {s['p95_latency_ms']} | {b.get('p95_latency_ms', '-')} |")
    lines.append(f"| Avg steps | {s['avg_steps']} | {b.get('avg_steps', '-')} |")
    lines += ["", "| Case | Category | Result | Cost (USD) | Steps | Failed checks |", "|---|---|---|---|---|---|"]
    for c in card.cases:
        mark = "pass" if c.passed else ("**FAIL (critical)**" if c.critical else "FAIL")
        lines.append(f"| {c.id} | {c.category} | {mark} | {c.cost_usd:.5f} | {c.steps} | {'; '.join(c.failed_checks)} |")
    return "\n".join(lines) + "\n"


def save(card: Scorecard, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**card.summary(), "results": [asdict(c) for c in card.cases]}, indent=2), encoding="utf-8")
