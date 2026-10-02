"""Minimal span tracer with OpenTelemetry-shaped output.

Spans carry trace/span/parent ids and attributes using the OpenTelemetry
GenAI semantic-convention names where one exists (``gen_ai.request.model``,
``gen_ai.usage.input_tokens``...), so the JSONL export maps cleanly onto an
OTLP pipeline (Datadog, Cloud Trace, Langfuse, Phoenix) later.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    start_ms: float
    end_ms: float | None = None
    status: str = "ok"
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_ms or self.start_ms) - self.start_ms


class Tracer:
    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex
        self.spans: list[Span] = []
        self._stack: list[Span] = []

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        span = Span(
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_id=self._stack[-1].span_id if self._stack else None,
            name=name,
            start_ms=time.perf_counter() * 1000,
            attributes=dict(attributes),
        )
        self.spans.append(span)
        self._stack.append(span)
        try:
            yield span
        except Exception as exc:
            span.status = "error"
            span.attributes["error.type"] = type(exc).__name__
            raise
        finally:
            span.end_ms = time.perf_counter() * 1000
            self._stack.pop()

    def event(self, name: str, **attributes: Any) -> None:
        """Zero-duration span for point-in-time events (guardrail hits, approvals)."""
        with self.span(name, **attributes):
            pass

    def find(self, name: str) -> list[Span]:
        return [s for s in self.spans if s.name == name]

    def export_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for s in self.spans:
                f.write(json.dumps(asdict(s), default=str) + "\n")

    def render_tree(self) -> str:
        children: dict[str | None, list[Span]] = {}
        for s in self.spans:
            children.setdefault(s.parent_id, []).append(s)
        lines: list[str] = []

        def walk(parent: str | None, depth: int) -> None:
            for s in children.get(parent, []):
                interesting = {k: v for k, v in s.attributes.items() if k not in ("gen_ai.input",)}
                attrs = " ".join(f"{k}={v}" for k, v in interesting.items())
                flag = "" if s.status == "ok" else f" [{s.status}]"
                lines.append(f"{'  ' * depth}- {s.name}{flag} {attrs}".rstrip())
                walk(s.span_id, depth + 1)

        walk(None, 0)
        return "\n".join(lines)
