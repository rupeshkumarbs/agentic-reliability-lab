"""Ask the agent one question and print the answer, the trace tree and the cost.

    python -m arl "Refund ORD-1003, it's broken" --approve
    python -m arl "Any promo for ORD-1001?" --agent agents/support-v2.json
    python -m arl "Where is ORD-1006?" --provider anthropic      # real Claude
"""

from __future__ import annotations

import argparse
import sys

from .agent import AgentConfig, SupportAgent, deny_all
from .gateway import ModelGateway
from .providers import make_provider
from .tools import SupportBackend
from .tracing import Tracer


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m arl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("message")
    p.add_argument("--agent", default="agents/support-v1.json")
    p.add_argument("--provider", default="reference", choices=["reference", "anthropic", "bedrock"])
    p.add_argument("--approve", action="store_true", help="approve high-risk actions (default: deny)")
    args = p.parse_args(argv)

    config = AgentConfig.load(args.agent)
    gateway = ModelGateway(provider=make_provider(args.provider))
    backend = SupportBackend()
    approver = (lambda *a: True) if args.approve else deny_all
    tracer = Tracer()
    result = SupportAgent(config, gateway, backend, approver).run(args.message, session="cli", tracer=tracer)

    print(f"\n{result.answer}\n")
    print(tracer.render_tree())
    print(f"\nstop={result.stop} steps={result.steps} cost={gateway.ledger.total_cost():.5f} USD "
          f"refunds={backend.refunds} guardrails={result.guardrail_events}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
