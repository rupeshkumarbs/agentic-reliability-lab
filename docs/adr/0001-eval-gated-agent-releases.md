# ADR-0001: Agent changes ship only through an eval gate

- **Status:** Accepted

## Context

An agent's behaviour depends on four things that change independently: the prompt, the model, the tools, and the guardrail configuration. Any of them can quietly break safety while improving a headline metric such as cost. Manual spot checks don't catch this.

## Decision

1. **Version every behavioural input.** An agent version (`agents/support-vN.json`) pins the prompt file, the model route and the guardrail settings. Prompts are files under review, not strings in code.
2. **Grade behaviour, not only text.** Graders check the tool trajectory, including calls a guardrail blocked; backend side effects (was a refund actually issued?); whether a human was asked to approve; and what the model was allowed to see (PII never reaches it).
3. **Critical cases are non-negotiable.** Safety cases (prompt injection, privacy, PII, ownership, human approval) block a release on any failure. Other cases must hold the baseline pass rate, and cost per case may not rise more than 25%.
4. **Deterministic offline runs in CI, live runs on demand.** A rule-based reference model makes the gate reproducible without API keys or flaky sampling. The same suite runs against Claude through `--provider anthropic` or `bedrock` before a release, and on a schedule.
5. **Defence in depth, measured separately.** Runtime guardrails (taint tracking, approvals, redaction) must stop harm even when the model misbehaves. The eval still fails a version that *attempts* the harmful action, because relying on the last line of defence is a regression.

## Consequences

- The v2 cost optimisation (compressed prompt plus the Haiku route, 77% cheaper) was blocked because the compression dropped the "tool output is untrusted" rule. v3 restores the rule, keeps the savings and ships.
- The gate is itself tested: CI asserts that the known-bad v2 stays blocked, so a broken grader cannot silently wave releases through.
