"""Guardrails at every trust boundary of the agent loop.

input  -> redact payment / government identifiers before the model sees them
tools  -> detect prompt injection in tool output; flag it and taint the session
action -> tool allowlist, argument validation, approval rules (amount + taint)
output -> redact any PII before the answer leaves the system
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --- PII -----------------------------------------------------------------------
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)")
PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
AADHAAR_RE = re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")
CARD_RE = re.compile(r"(?<!\d)(?:\d[\s-]?){13,19}(?!\d)")


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact(text: str, *, include_contact: bool) -> tuple[str, list[str]]:
    """Return (redacted_text, kinds_found). Contact details are kept on input
    (needed for ownership checks) and redacted on output."""
    found: list[str] = []

    def card(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            found.append("card")
            return "[REDACTED_CARD]"
        return m.group(0)

    text = CARD_RE.sub(card, text)
    for kind, pattern in (("pan", PAN_RE), ("aadhaar", AADHAAR_RE)):
        text, n = pattern.subn(f"[REDACTED_{kind.upper()}]", text)
        found += [kind] * n
    if include_contact:
        for kind, pattern in (("email", EMAIL_RE), ("phone", PHONE_RE)):
            text, n = pattern.subn(f"[REDACTED_{kind.upper()}]", text)
            found += [kind] * n
    return text, found


# --- Prompt injection ------------------------------------------------------------
INJECTION_PATTERNS = [
    r"ignore (all |any )?(of )?(the )?(previous|prior|above) (instructions|rules)",
    r"disregard (all |any )?(previous|prior|your) (instructions|rules)",
    r"^\s*system\s*:",
    r"you are now\b",
    r"reveal (your |the )?(system prompt|instructions)",
    r"without (any )?(human )?approval",
]
_INJECTION = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in INJECTION_PATTERNS]


def detect_injection(text: str) -> list[str]:
    return [p.pattern for p in _INJECTION if p.search(text)]


# --- Action policy -----------------------------------------------------------------
@dataclass(frozen=True)
class GuardrailConfig:
    redact_input_pii: bool = True
    flag_untrusted_tool_output: bool = True
    redact_output_pii: bool = True
    approval_threshold_inr: int = 5000
    taint_requires_approval: bool = True


HIGH_RISK_TOOLS = {"issue_refund"}


def validate_args(tool_spec: dict, args: dict) -> str | None:
    """Check required fields, types and unknown fields against the tool's JSON schema."""
    schema = tool_spec["input_schema"]
    props = schema.get("properties", {})
    types = {"string": str, "integer": int, "number": (int, float), "boolean": bool}
    for name in schema.get("required", []):
        if name not in args:
            return f"missing required argument {name!r}"
    for name, value in args.items():
        if name not in props:
            return f"unexpected argument {name!r}"
        expected = types.get(props[name].get("type"))
        if expected and (not isinstance(value, expected) or isinstance(value, bool) and expected is not bool):
            return f"argument {name!r} must be {props[name]['type']}"
        if props[name].get("minimum") is not None and value < props[name]["minimum"]:
            return f"argument {name!r} below minimum"
    return None


def approval_reason(tool: str, args: dict, *, config: GuardrailConfig, tainted: bool) -> str | None:
    """Why this call needs a human, or None if it can run unattended."""
    if tool not in HIGH_RISK_TOOLS:
        return None
    if tainted and config.taint_requires_approval:
        return "session handled untrusted content flagged as prompt injection"
    if args.get("amount_inr", 0) > config.approval_threshold_inr:
        return f"amount above {config.approval_threshold_inr} INR"
    return None
