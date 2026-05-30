"""Run API-consuming conversational benchmarks and save raw ReAct traces.

This runner is intentionally separate from the offline suite: every prompt may
invoke GPT-4o several times. Start with ``--limit 2`` before running a sweep.

Example:
    python -m eval.run_agent --variant react_full --limit 2
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.common import (
    DATASETS_DIR,
    RUNS_DIR,
    mean,
    read_jsonl,
    resolve_repo_path,
    trace_step_count,
    trace_tool_names,
    usage_delta,
    write_json,
)
from eval.variants import get_variant
from scheduler import llm_client
from scheduler.catalog import Catalog, extract_course_codes
from scheduler.schemas import PlannerRequest
from scheduler.session import PlannerSession


def _request_for_case(case: dict, variant, model: str) -> PlannerRequest:
    fixture = resolve_repo_path(case.get("student_fixture"))
    suffix = fixture.suffix.lower() if fixture else ""
    return PlannerRequest(
        target_term=str(case.get("target_term", "202601")),
        user_request="",
        transcript_html=fixture if fixture and suffix in {".html", ".htm"} else None,
        transcript_json=fixture if fixture and suffix == ".json" else None,
        transcript_pdf=fixture if fixture and suffix not in {".html", ".htm", ".json"} else None,
        planner_model=model,
        intent_model=model,
        include_required=variant.include_required,
        mode=variant.mode,
    )


def _tool_recall(expected: list[str], observed: list[str]) -> float | None:
    if not expected:
        return None
    return round(len(set(expected) & set(observed)) / len(set(expected)), 4)


def _run_case(case: dict, variant, model: str, catalog: Catalog) -> dict:
    session = PlannerSession.from_request(_request_for_case(case, variant, model))
    manual = case.get("manual_context") or {}
    if manual:
        session.update_manual_context(
            program=manual.get("program"),
            admit_term=manual.get("admit_term"),
            completed=set(manual.get("completed", [])),
            in_progress=set(manual.get("in_progress", [])),
        )

    turns_out: list[dict] = []
    for turn_index, message in enumerate(case.get("turns", []), start=1):
        before = llm_client.get_session_usage()
        started = time.perf_counter()
        answer = session.handle_turn(message)
        latency = time.perf_counter() - started
        after = llm_client.get_session_usage()
        trace = list(session.last_raw_trace)
        tools = trace_tool_names(trace)
        codes = extract_course_codes(answer)
        turns_out.append({
            "turn": turn_index,
            "message": message,
            "answer": answer,
            "trace": trace,
            "tool_names": tools,
            "react_steps": trace_step_count(trace),
            "latency_s": round(latency, 4),
            "usage": usage_delta(before, after),
            "answer_course_codes": codes,
            "hallucinated_answer_codes": [code for code in codes if catalog.get(code) is None],
        })

    expected_tools = list(case.get("expected_tools", []))
    observed_tools = [tool for turn in turns_out for tool in turn["tool_names"]]
    plan = session.current_plan.to_dict() if session.current_plan else None
    return {
        "id": case["id"],
        "category": case.get("category"),
        "variant": variant.name,
        "model": model,
        "expected_tools": expected_tools,
        "observed_tools": observed_tools,
        "tool_recall": _tool_recall(expected_tools, observed_tools),
        "profile": session.state_summary(),
        "plan": plan,
        "turns": turns_out,
    }


def _summary(records: list[dict]) -> dict:
    turns = [turn for record in records for turn in record["turns"]]
    hallucinated = [code for turn in turns for code in turn["hallucinated_answer_codes"]]
    return {
        "case_count": len(records),
        "turn_count": len(turns),
        "mean_latency_s": mean(turn["latency_s"] for turn in turns),
        "mean_react_steps": mean(turn["react_steps"] for turn in turns),
        "mean_tool_calls": mean(len(turn["tool_names"]) for turn in turns),
        "mean_total_tokens": mean(turn["usage"]["total_tokens"] for turn in turns),
        "mean_expected_tool_recall": mean(
            record["tool_recall"] for record in records if record["tool_recall"] is not None
        ),
        "hallucinated_answer_codes": sorted(set(hallucinated)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DATASETS_DIR / "prompts.jsonl"))
    ap.add_argument("--variant", default="react_full")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--limit", type=int, default=0, help="Run only the first N cases; 0 means all.")
    ap.add_argument("--category", action="append", help="Only run selected categories; repeat flag as needed.")
    ap.add_argument("--out-dir", default=str(RUNS_DIR))
    args = ap.parse_args()

    variant = get_variant(args.variant)
    cases = read_jsonl(args.dataset)
    if args.category:
        wanted = set(args.category)
        cases = [case for case in cases if case.get("category") in wanted]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No benchmark cases selected.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.out_dir) / f"{run_id}_{variant.name}_{args.model}"
    catalog = Catalog.load()
    records: list[dict] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']}")
        record = _run_case(case, variant, args.model, catalog)
        records.append(record)
        write_json(run_dir / f"{case['id']}.json", record)
        print(
            f"  tools={len(record['observed_tools'])} "
            f"tool_recall={record['tool_recall']} "
            f"tokens={sum(t['usage']['total_tokens'] for t in record['turns'])}"
        )

    report: dict[str, Any] = {
        "suite": "online_agent",
        "run_id": run_id,
        "variant": variant.name,
        "variant_description": variant.description,
        "model": args.model,
        "dataset": args.dataset,
        "summary": _summary(records),
        "records": [{"id": record["id"], "path": f"{record['id']}.json"} for record in records],
    }
    write_json(run_dir / "summary.json", report)
    print(f"\nSummary: {run_dir / 'summary.json'}")
    for key, value in report["summary"].items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
