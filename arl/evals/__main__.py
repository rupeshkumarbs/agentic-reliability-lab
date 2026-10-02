"""CLI: python -m arl.evals --agent agents/support-v2.json --baseline evals/baselines/support.json

Exit code 0 = gate passed (ship), 1 = blocked. Use in CI as a required check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..agent import AgentConfig
from ..providers import make_provider
from ..tools import ROOT
from .runner import release_gate, render_markdown, run_suite, save


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m arl.evals", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", required=True, help="agent config, e.g. agents/support-v1.json")
    p.add_argument("--provider", default="reference", choices=["reference", "anthropic", "bedrock"])
    p.add_argument("--baseline", type=Path, help="baseline scorecard JSON to compare against")
    p.add_argument("--out", type=Path, default=ROOT / "out", help="directory for scorecard + traces")
    p.add_argument("--write-baseline", type=Path, help="save this run as the new baseline (only if the gate passes)")
    args = p.parse_args(argv)

    config = AgentConfig.load(args.agent)
    card = run_suite(config, make_provider(args.provider), trace_dir=args.out / "traces" / config.version)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline and args.baseline.exists() else None
    decision = release_gate(card, baseline)

    report = render_markdown(card, baseline, decision)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"scorecard-{config.version}.md").write_text(report, encoding="utf-8")
    save(card, args.out / f"scorecard-{config.version}.json")
    print(report)

    if args.write_baseline:
        if decision.ship:
            save(card, args.write_baseline)
            print(f"Baseline updated: {args.write_baseline}")
        else:
            print("Baseline NOT updated: gate is blocked.")
    return 0 if decision.ship else 1


if __name__ == "__main__":
    sys.exit(main())
