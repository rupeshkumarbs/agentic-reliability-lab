"""Support tools: specs (strict JSON schemas) and an in-memory backend.

Each eval case gets a fresh backend, so side effects (refunds, escalations)
are observable and isolated per case.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from .retrieval import BM25, load_chunks

ROOT = Path(__file__).resolve().parent.parent

TOOL_SPECS: list[dict] = [
    {
        "name": "lookup_order",
        "description": "Look up an order by ID. Returns status, category, amount, delivery age, refund state and the account email.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "Order ID like ORD-1234"}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_policy",
        "description": "Search the store policy knowledge base (refunds, shipping, privacy, promotions).",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "issue_refund",
        "description": "Issue a refund to the original payment method. High-risk: may require specialist approval.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "amount_inr": {"type": "integer", "minimum": 1},
                "reason": {"type": "string"},
            },
            "required": ["order_id", "amount_inr", "reason"],
            "additionalProperties": False,
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Hand the conversation to a human specialist with a short summary.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        },
    },
]
SPECS_BY_NAME = {t["name"]: t for t in TOOL_SPECS}

_ORDERS = json.loads((ROOT / "data" / "orders.json").read_text(encoding="utf-8"))
_RETRIEVER = BM25(load_chunks(ROOT / "kb"))


class SupportBackend:
    def __init__(self) -> None:
        self.orders = {o["order_id"]: copy.deepcopy(o) for o in _ORDERS}
        self.refunds: list[dict] = []
        self.escalations: list[str] = []

    def execute(self, name: str, args: dict) -> tuple[str, bool]:
        """Run a tool; returns (JSON result, is_error)."""
        handler = getattr(self, f"_{name}", None)
        if handler is None:
            return json.dumps({"error": f"unknown tool {name}"}), True
        result = handler(**args)
        return json.dumps(result), "error" in result

    def _lookup_order(self, order_id: str) -> dict:
        order = self.orders.get(order_id.strip().upper())
        return dict(order) if order else {"error": f"order {order_id} not found"}

    def _search_policy(self, query: str) -> dict:
        hits = _RETRIEVER.search(query, k=2)
        return {"results": [{"source": c.source, "title": c.title, "text": c.text} for c, _ in hits]}

    def _issue_refund(self, order_id: str, amount_inr: int, reason: str) -> dict:
        order = self.orders.get(order_id)
        if order is None:
            return {"error": f"order {order_id} not found"}
        if order["status"] != "delivered":
            return {"error": "order not delivered"}
        if order["refunded"]:
            return {"error": "order already refunded"}
        if amount_inr > order["amount_inr"]:
            return {"error": "refund exceeds order amount"}
        order["refunded"] = True
        refund = {"refund_id": f"RF-{len(self.refunds) + 1:04d}", "order_id": order_id, "amount_inr": amount_inr}
        self.refunds.append(refund)
        return refund

    def _escalate_to_human(self, summary: str) -> dict:
        self.escalations.append(summary)
        return {"ticket": f"ESC-{len(self.escalations):04d}", "eta_hours": 24}
