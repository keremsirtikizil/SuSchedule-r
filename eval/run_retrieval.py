"""Measure catalog retrieval quality without invoking the chat agent.

Examples:
    python -m eval.run_retrieval --backends lexical
    python -m eval.run_retrieval --backends lexical faiss faiss_rerank
    python -m eval.run_retrieval --backends faiss_rerank --use-rewrites
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from eval.common import DATASETS_DIR, REPORTS_DIR, mean, read_jsonl, retrieval_metrics, write_json
from scheduler.catalog import Catalog


def _merge_rows(groups: list[list[dict]], k: int) -> list[dict]:
    best: dict[str, dict] = {}
    for rows in groups:
        for row in rows:
            code = row["code"]
            score = float(row.get("reranker_score", row.get("score", 0.0)) or 0.0)
            if code not in best or score > float(best[code].get("_merge_score", -10**9)):
                best[code] = {**row, "_merge_score": score}
    return sorted(best.values(), key=lambda row: (-row["_merge_score"], row["code"]))[:k]


def _lexical_rows(catalog: Catalog, queries: list[str], k: int) -> list[dict]:
    return _merge_rows([catalog.search(query, k=k) for query in queries], k)


def _semantic_rows(retriever, queries: list[str], k: int, rerank: bool) -> list[dict]:
    groups: list[list[dict]] = []
    for query in queries:
        hits = retriever.retrieve(query, k=k, rerank=rerank)
        groups.append([
            {
                "code": hit.code,
                "title": hit.title,
                "score": hit.score,
                "reranker_score": hit.reranker_score,
            }
            for hit in hits
        ])
    return _merge_rows(groups, k)


def _backend_report(
    backend: str,
    cases: list[dict],
    catalog: Catalog,
    k: int,
    use_rewrites: bool,
) -> dict:
    retriever = None
    if backend in {"faiss", "faiss_rerank"}:
        from scheduler.retriever import Retriever
        retriever = Retriever.load(
            device="cpu",
            load_model=True,
            load_reranker=(backend == "faiss_rerank"),
        )
        if retriever._model is None:  # intentional readiness check for the benchmark CLI
            raise RuntimeError(
                "Bi-encoder model is not available in the local Hugging Face cache. "
                "Run python -m scheduler.build_faiss_index once with network access."
            )

    rows_out: list[dict] = []
    for case in cases:
        queries = list(case.get("rewrites") or []) if use_rewrites else []
        queries = queries or [case["query"]]
        if backend == "lexical":
            rows = _lexical_rows(catalog, queries, k)
        elif backend == "faiss":
            rows = _semantic_rows(retriever, queries, k, rerank=False)
        elif backend == "faiss_rerank":
            rows = _semantic_rows(retriever, queries, k, rerank=True)
        else:
            raise ValueError(f"Unknown retrieval backend: {backend}")
        ranked_codes = [row["code"] for row in rows]
        metrics = retrieval_metrics(ranked_codes, case["relevance"], k=k)
        rows_out.append({
            "id": case["id"],
            "query": case["query"],
            "queries_used": queries,
            "ranked_codes": ranked_codes,
            "metrics": metrics,
        })

    return {
        "backend": backend,
        "use_rewrites": use_rewrites,
        "query_count": len(rows_out),
        "metrics": {
            metric: mean(row["metrics"][metric] for row in rows_out)
            for metric in ("recall@5", "recall@10", "mrr", "ndcg@10")
        },
        "queries": rows_out,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default=str(DATASETS_DIR / "retrieval_gold.jsonl"),
        help="Gold retrieval JSONL dataset.",
    )
    ap.add_argument(
        "--backends",
        nargs="+",
        choices=["lexical", "faiss", "faiss_rerank"],
        default=["lexical"],
        help="Retrieval configurations to compare.",
    )
    ap.add_argument("--use-rewrites", action="store_true", help="Use the hand-reviewed query rewrites in the gold dataset.")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument(
        "--json",
        default=str(REPORTS_DIR / "retrieval.json"),
        help="Output report path.",
    )
    args = ap.parse_args()

    cases = read_jsonl(args.dataset)
    catalog = Catalog.load()
    report: dict[str, Any] = {
        "suite": "retrieval",
        "dataset": args.dataset,
        "k": args.k,
        "backends": [],
        "errors": [],
    }
    for backend in args.backends:
        print(f"\n[retrieval] backend={backend} rewrites={args.use_rewrites}")
        try:
            row = _backend_report(backend, cases, catalog, args.k, args.use_rewrites)
            report["backends"].append(row)
            print(
                "  "
                + " ".join(f"{name}={value:.4f}" for name, value in row["metrics"].items())
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            report["errors"].append({"backend": backend, "error": error})
            print(f"  ERROR: {error}")
    write_json(args.json, report)
    print(f"\nReport: {Path(args.json)}")
    return 0 if report["backends"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
