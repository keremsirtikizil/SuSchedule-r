"""Aggregate one or more saved online-agent summary files into a comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.common import REPORTS_DIR, write_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+", help="Paths to eval/runs/*/summary.json")
    ap.add_argument("--json", default=str(REPORTS_DIR / "agent_comparison.json"))
    args = ap.parse_args()

    rows = []
    for summary_path in args.summaries:
        payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        rows.append({
            "run_id": payload["run_id"],
            "variant": payload["variant"],
            "model": payload["model"],
            **payload["summary"],
        })
    report = {"suite": "online_agent_comparison", "runs": rows}
    write_json(args.json, report)

    headers = ["variant", "case_count", "mean_latency_s", "mean_react_steps", "mean_tool_calls", "mean_total_tokens", "mean_expected_tool_recall"]
    print(" | ".join(headers))
    print(" | ".join("---" for _ in headers))
    for row in rows:
        print(" | ".join(str(row.get(header, "")) for header in headers))
    print(f"\nReport: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
