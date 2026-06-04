"""
Retrieval evaluation for SuSchedule-r.

Measures how well the two-stage retriever (BGE bi-encoder -> BGE cross-encoder
reranker) surfaces the relevant course(s) for a labeled set of natural-language
student queries (eval/retrieval_queries.json).

Metrics (averaged over queries):
  - Hit@1, Hit@5  : fraction of queries with >=1 gold course in the top-1 / top-5
  - Recall@5, @10 : fraction of the gold set found in the top-k
  - MRR@10        : mean reciprocal rank of the first gold course
  - nDCG@10       : normalized discounted cumulative gain (binary relevance)

Ablation: every metric is reported twice -- with the cross-encoder reranker ON
(the production setting) and OFF (bi-encoder order only) -- to quantify the
reranker's contribution.

Runs fully locally (no OpenAI). Usage:
    python -m eval.run_retrieval
"""
from __future__ import annotations

import json
import math
import hashlib
from pathlib import Path

from scheduler.retriever import (
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_ID_MAP_PATH,
    DEFAULT_INDEX_PATH,
    DEFAULT_PARQUET_PATH,
    Retriever,
    RetrieverFilter,
)

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"
K = 10  # depth we score to


def first_gold_rank(codes: list[str], gold: set[str]) -> int | None:
    for i, c in enumerate(codes, start=1):
        if c in gold:
            return i
    return None


def ndcg_at_k(codes: list[str], gold: set[str], k: int) -> float:
    dcg = 0.0
    for i, c in enumerate(codes[:k], start=1):
        if c in gold:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def score_query(codes: list[str], gold: set[str]) -> dict:
    top1, top5, top10 = set(codes[:1]), set(codes[:5]), set(codes[:10])
    fr = first_gold_rank(codes, gold)
    return {
        "hit@1": 1.0 if gold & top1 else 0.0,
        "hit@5": 1.0 if gold & top5 else 0.0,
        "recall@5": len(gold & top5) / len(gold),
        "recall@10": len(gold & top10) / len(gold),
        "mrr@10": (1.0 / fr) if (fr is not None and fr <= 10) else 0.0,
        "ndcg@10": ndcg_at_k(codes, gold, 10),
    }


def mean(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def main() -> None:
    dataset_path = EVAL_DIR / "retrieval_queries.json"
    dataset_bytes = dataset_path.read_bytes()
    data = json.loads(dataset_bytes)
    queries = data["queries"]

    required_artifacts = [
        DEFAULT_PARQUET_PATH,
        DEFAULT_ID_MAP_PATH,
        DEFAULT_EMBEDDINGS_PATH,
        DEFAULT_INDEX_PATH,
    ]
    missing_artifacts = [str(path.relative_to(ROOT)) for path in required_artifacts if not path.exists()]
    if missing_artifacts:
        raise SystemExit(
            "Retrieval evaluation requires all local RAG artifacts. Missing: "
            f"{missing_artifacts}. Restore them or run: python -m scheduler.build_faiss_index"
        )

    print(f"Loading retriever (bi-encoder + cross-encoder reranker)...")
    r = Retriever.load(device="cpu")
    initial_backend = r.backend_info()
    if not initial_backend["faiss_index_loaded"]:
        raise SystemExit(
            "FAISS index did not load. Retrieval evaluation requires FAISS so "
            "the reported first-stage metrics match the production design."
        )

    # Catch stale labels before they quietly lower the score.
    all_codes = set(r.filter_only(RetrieverFilter(active_only=False)))
    missing = sorted({g for q in queries for g in q["gold"] if g not in all_codes})
    if missing:
        print(f"[WARN] gold codes not in catalog (will count as unreachable): {missing}")

    metric_keys = ["hit@1", "hit@5", "recall@5", "recall@10", "mrr@10", "ndcg@10"]
    rows = {"rerank": [], "bienc": []}
    per_query = []

    for q in queries:
        gold = set(q["gold"])
        hits_rr = r.retrieve(q["query"], k=K, rerank=True)
        hits_bi = r.retrieve(q["query"], k=K, rerank=False)
        codes_rr = [h.code for h in hits_rr]
        codes_bi = [h.code for h in hits_bi]
        s_rr = score_query(codes_rr, gold)
        s_bi = score_query(codes_bi, gold)
        review_status = q.get("review_status", "existing")
        rows["rerank"].append({**s_rr, "category": q["category"], "review_status": review_status})
        rows["bienc"].append({**s_bi, "category": q["category"], "review_status": review_status})
        per_query.append({
            "id": q["id"], "query": q["query"], "gold": q["gold"],
            "category": q["category"], "review_status": review_status,
            "rerank_top5": codes_rr[:5], "bienc_top5": codes_bi[:5],
            "rerank": s_rr, "bienc": s_bi,
        })

    # Print the headline metrics.
    print(f"\n=== Retrieval metrics over {len(queries)} queries (k={K}) ===")
    header = f"{'metric':<12}{'reranker ON':>14}{'reranker OFF':>14}{'delta':>10}"
    print(header)
    print("-" * len(header))
    summary = {
        "n_queries": len(queries),
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "k": K,
        "rerank": {},
        "bienc": {},
        "delta": {},
    }
    for key in metric_keys:
        a = mean(rows["rerank"], key)
        b = mean(rows["bienc"], key)
        summary["rerank"][key] = round(a, 4)
        summary["bienc"][key] = round(b, 4)
        summary["delta"][key] = round(a - b, 4)
        print(f"{key:<12}{a:>14.3f}{b:>14.3f}{a - b:>+10.3f}")

    # Show where the reranked retriever is strongest and weakest.
    cats = sorted({q["category"] for q in queries})
    print("\n=== recall@10 by category (reranker ON) ===")
    by_cat = {}
    for c in cats:
        sub = [x for x in rows["rerank"] if x["category"] == c]
        v = mean(sub, "recall@10")
        by_cat[c] = round(v, 4)
        print(f"  {c:<16} {v:.3f}  (n={len(sub)})")
    summary["by_category_recall@10"] = by_cat

    print("\n=== recall@10 by label review status (reranker ON) ===")
    by_review = {}
    statuses = sorted({q.get("review_status", "existing") for q in queries})
    for status in statuses:
        sub = [x for x in rows["rerank"] if x["review_status"] == status]
        v = mean(sub, "recall@10")
        by_review[status] = {"recall@10": round(v, 4), "n": len(sub)}
        print(f"  {status:<16} {v:.3f}  (n={len(sub)})")
    summary["by_review_status"] = by_review

    backend = r.backend_info()
    if backend["search_counts"]["faiss"] == 0:
        raise RuntimeError(
            "No retrieval query used FAISS. The benchmark would not measure "
            "the intended FAISS-backed first stage."
        )
    summary["retriever_backend"] = backend
    print("\n=== first-stage backend ===")
    print(
        f"  {backend['index_type']} · {backend['index_ntotal']} vectors · "
        f"FAISS searches={backend['search_counts']['faiss']}"
    )

    out = {"summary": summary, "per_query": per_query, "missing_gold": missing}
    (EVAL_DIR / "results_retrieval.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> eval/results_retrieval.json")


if __name__ == "__main__":
    main()
