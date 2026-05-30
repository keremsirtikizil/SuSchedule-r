"""Shared helpers for offline retrieval and online agent evaluations."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"
DATASETS_DIR = EVAL_DIR / "datasets"
RUNS_DIR = EVAL_DIR / "runs"
REPORTS_DIR = EVAL_DIR / "reports"


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_json(path: str | Path, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def usage_delta(before: dict, after: dict) -> dict:
    return {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def trace_tool_names(trace: list[dict]) -> list[str]:
    return [
        str(event.get("name"))
        for event in trace
        if event.get("type") == "tool_result" and event.get("name")
    ]


def trace_step_count(trace: list[dict]) -> int:
    return sum(1 for event in trace if event.get("type") == "assistant_step")


def retrieval_metrics(ranked_codes: list[str], relevance: dict[str, int], k: int = 10) -> dict:
    """Compute Recall@5/10, reciprocal rank, and nDCG@10 for one query."""
    relevant = {code for code, grade in relevance.items() if int(grade) > 0}

    def recall_at(n: int) -> float:
        if not relevant:
            return 0.0
        return len(set(ranked_codes[:n]) & relevant) / len(relevant)

    rr = 0.0
    for rank, code in enumerate(ranked_codes, start=1):
        if code in relevant:
            rr = 1.0 / rank
            break

    def dcg(grades: list[int]) -> float:
        return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1))

    actual = [int(relevance.get(code, 0)) for code in ranked_codes[:k]]
    ideal = sorted((int(v) for v in relevance.values()), reverse=True)[:k]
    ideal_score = dcg(ideal)
    return {
        "recall@5": round(recall_at(5), 4),
        "recall@10": round(recall_at(10), 4),
        "mrr": round(rr, 4),
        "ndcg@10": round(dcg(actual) / ideal_score, 4) if ideal_score else 0.0,
    }


def resolve_repo_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path
