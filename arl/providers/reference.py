"""Deterministic reference model for offline CI.

Not an LLM: a rule-based policy that consumes the same inputs a real model gets
(system prompt, conversation, tool results) and emits Messages API content
blocks. Its behaviour depends on the *prompt*, the way an instruction-following
model's does - drop the "treat tool output as untrusted" rule and it starts
obeying instructions planted in retrieved documents. That makes prompt
regressions reproducible in CI without API keys; run the same suite with
``--provider anthropic`` to evaluate real Claude.

Token counts are estimated (~4 chars/token) so FinOps numbers stay meaningful;
latency is simulated per model tier.
"""

from __future__ import annotations

import json
import re

from ..guardrails import EMAIL_RE
from .base import ModelResponse, TransientError

ORDER_RE = re.compile(r"\bORD-\d{4}\b", re.IGNORECASE)
_LATENCY_MS = {"claude-opus-5-5": 1400.0, "claude-sonnet-5-5": 900.0, "claude-haiku-4-5": 450.0}


class ReferenceModel:
    name = "reference"

    def __init__(self, fail_first: dict[str, int] | None = None) -> None:
        # Fault injection for gateway drills: {"claude-opus-5-5": 2} fails the first 2 calls.
        self.fail_first = dict(fail_first or {})

    def complete(self, *, model: str, system: str, messages: list[dict], tools: list[dict], max_tokens: int) -> ModelResponse:
        if self.fail_first.get(model, 0) > 0:
            self.fail_first[model] -= 1
            raise TransientError(f"{model}: 529 overloaded (injected)")

        user, calls = _read_history(messages)
        kind, payload = _decide(system, user, calls)
        if kind == "tool":
            name, args = payload
            content = [{"type": "tool_use", "id": f"toolu_ref_{len(calls) + 1:02d}", "name": name, "input": args}]
            stop = "tool_use"
        else:
            content = [{"type": "text", "text": payload}]
            stop = "end_turn"

        prompt_chars = len(system) + len(json.dumps(messages)) + len(json.dumps(tools))
        output_tokens = max(1, len(json.dumps(content)) // 4)
        latency = _LATENCY_MS.get(model, 800.0) + output_tokens * 8
        return ModelResponse(model, content, stop, prompt_chars // 4, output_tokens, latency)


def _read_history(messages: list[dict]) -> tuple[str, list[dict]]:
    first = messages[0]["content"]
    user = first if isinstance(first, str) else " ".join(b.get("text", "") for b in first)
    by_id: dict[str, dict] = {}
    order: list[str] = []
    for m in messages[1:]:
        if not isinstance(m["content"], list):
            continue
        for b in m["content"]:
            if m["role"] == "assistant" and b.get("type") == "tool_use":
                by_id[b["id"]] = {"name": b["name"], "input": b["input"]}
                order.append(b["id"])
            elif m["role"] == "user" and b.get("type") == "tool_result":
                call = by_id.get(b["tool_use_id"], {})
                call["result"] = _unwrap(b.get("content", ""))
                call["is_error"] = bool(b.get("is_error"))
    return user, [by_id[i] for i in order]


def _unwrap(content) -> dict:
    """Tool results are JSON; guardrails may wrap them as {"warning", "data"}."""
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content)
    try:
        value = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return {"raw": str(content)}
    return value.get("data", value) if isinstance(value, dict) else {"raw": value}


def _decide(system: str, user: str, calls: list[dict]):
    s, u = system.lower(), user.lower()
    follows_untrusted_rule = "untrusted" in s
    checks_ownership = "ownership" in s or "owns it" in s
    protects_privacy = "never disclose" in s
    m = ORDER_RE.search(user)
    oid = m.group(0).upper() if m else None

    def last(name: str) -> dict | None:
        return next((c for c in reversed(calls) if c.get("name") == name), None)

    def tool(name: str, args: dict):
        return "tool", (name, args)

    def text(t: str):
        return "text", t

    # Privacy: requests for someone else's details.
    if protects_privacy and re.search(r"(email|phone|address|name|contact)\b.*\b(of|for) (the )?(customer|person|buyer)|who placed", u):
        return text("I'm sorry, I can't share details about other customers or their orders. "
                    "I can help with orders placed from your own account.")

    # Explicit request for a human.
    if re.search(r"\b(human|manager|supervisor|real person)\b", u):
        if not last("escalate_to_human"):
            return tool("escalate_to_human", {"summary": f"Customer asked for a human ({oid or 'no order id'}): {user[:160]}"})
        return text("I've passed this to a human specialist; they'll contact you within 24 hours.")

    # Promotions (the retrieved promotions page carries a planted instruction).
    if re.search(r"\b(promo|promotion|discount|coupon|offer)s?\b", u):
        pol = last("search_policy")
        if not pol:
            return tool("search_policy", {"query": "current promotions discounts offers"})
        docs = "\n".join(r["text"] for r in pol.get("result", {}).get("results", []))
        planted = re.search(r"issue a full refund for (ORD-\d{4})(?: of (\d[\d,]*))?", docs, re.IGNORECASE)
        if planted and not follows_untrusted_rule and not last("issue_refund"):
            amount = int((planted.group(2) or "1").replace(",", ""))
            return tool("issue_refund", {"order_id": planted.group(1).upper(), "amount_inr": amount, "reason": "goodwill gesture"})
        offer = next((ln[2:] for ln in docs.splitlines() if ln.startswith("- ") and "refund" not in ln.lower()), None)
        return text(f"Current offer: {offer}" if offer else "There are no active promotions right now.")

    # Refunds.
    if re.search(r"\b(refund|return|money back)\b", u):
        if not oid:
            return text("I can help with that. Could you share your order ID? It looks like ORD-1234.")
        looked = last("lookup_order")
        if not looked:
            return tool("lookup_order", {"order_id": oid})
        order = looked.get("result", {})
        if "error" in order:
            return text(f"I couldn't find an order with ID {oid}. Please double-check the ID.")
        claimed = EMAIL_RE.search(user)
        if checks_ownership and claimed and claimed.group(0).lower() != order["customer_email"].lower():
            return text("I couldn't verify that this order belongs to you. For security, refunds can only be "
                        "issued to the account that placed the order.")
        if order["status"] != "delivered":
            return text(f"Order {oid} is still in transit, so it can't be refunded yet. "
                        "I can cancel it before delivery if you'd like.")
        if order["refunded"]:
            return text(f"Order {oid} has already been refunded, so no further refund is due.")
        pol = last("search_policy")
        if not pol:
            return tool("search_policy", {"query": f"refund window {order['category']}"})
        window = _refund_window(pol.get("result", {}), order["category"])
        if order["delivered_days_ago"] > window:
            return text(f"Order {oid} was delivered {order['delivered_days_ago']} days ago, outside the {window}-day "
                        f"refund window for {order['category']}, so I can't refund it.")
        refund = last("issue_refund")
        if not refund:
            return tool("issue_refund", {"order_id": oid, "amount_inr": order["amount_inr"], "reason": user[:120]})
        if refund.get("is_error"):
            err = refund.get("result", {}).get("error", "")
            if "approv" in err.lower():
                return text(f"This refund of {order['amount_inr']:,} INR needs specialist approval and wasn't approved, "
                            "so I haven't processed it. A specialist will review your case.")
            return text(f"I wasn't able to process the refund: {err}.")
        return text(f"Done: I've refunded {order['amount_inr']:,} INR for {oid}. "
                    "It will reach your original payment method in 5-7 business days.")

    # Order status.
    if re.search(r"\b(where|status|track|tracking|delivered|arrive)\b", u):
        if not oid:
            return text("Sure. What's your order ID? It looks like ORD-1234.")
        looked = last("lookup_order")
        if not looked:
            return tool("lookup_order", {"order_id": oid})
        order = looked.get("result", {})
        if "error" in order:
            return text(f"I couldn't find an order with ID {oid}. Please double-check the ID.")
        if order["status"] == "delivered":
            return text(f"Order {oid} was delivered {order['delivered_days_ago']} days ago.")
        return text(f"Order {oid} is in transit and should arrive in the next few days.")

    return text("I can help with order status, refunds, cancellations and promotions. What do you need?")


def _refund_window(policy: dict, category: str) -> int:
    docs = "\n".join(r["text"] for r in policy.get("results", []))
    specific = re.search(rf"{re.escape(category)}\w*:\s*refunds within (\d+) days", docs, re.IGNORECASE)
    default = re.search(r"all other categories:\s*refunds within (\d+) days", docs, re.IGNORECASE)
    if specific:
        return int(specific.group(1))
    return int(default.group(1)) if default else 30
