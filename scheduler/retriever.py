"""
FAISS-backed semantic retriever + cross-encoder re-ranker for the SU catalog.

Two-stage retrieval pipeline
-----------------------------
Stage 1 — Bi-encoder (FAISS IndexFlatIP)
    ``BAAI/bge-base-en-v1.5`` encodes every course once at build time.
    At query time the query is encoded and scored against the pre-filtered
    candidate set via inner product (≡ cosine, vectors are L2-normalised).
    Fast; retrieves top-``first_stage_k`` candidates (default 4 × k).

Stage 2 — Cross-encoder re-ranker (optional, recommended)
    ``BAAI/bge-reranker-base`` reads the full (query, passage) pair and
    produces a single calibrated relevance logit — far more accurate than
    bi-encoder cosine, at the cost of O(candidates) forward passes.
    At 688 courses the extra latency is ~0.5 s on CPU; negligible on GPU.
    Natural pair: bge-reranker-base ↔ bge-base-en-v1.5 (same BAAI family).

Typical call
------------
    retriever = Retriever.load()          # loads both models + index
    hits = retriever.retrieve(
        query      = "I want a course about operating systems",
        k          = 10,
        filt       = RetrieverFilter(
                         candidate_pool = eligible_set,   # from eligibility.py
                         seasons        = ["Fall"],
                         subj           = ["CS"],
                     ),
        rerank     = True,                # cross-encoder re-ranking (default)
    )

Design notes
------------
* All heavy objects (models, index, DataFrame) are loaded once in
  ``Retriever.load()`` and reused across calls.
* BGE requires a short instruction prefix on *queries only*; passages are
  stored and encoded without it.
* Cross-encoder passage text = same ``embedding_text`` field used by the
  bi-encoder (``"CODE — Title\\ndescription"``).
* Falls back gracefully: no faiss → numpy dot product; no reranker → return
  bi-encoder order; no model → ``retrieve_by_vector`` still works.
* ⚠️  macOS import order: sentence-transformers must be imported BEFORE faiss
  (duplicate OpenBLAS initialisation segfault on Apple Silicon).
* Run ``python -m scheduler.retriever`` for a CLI smoke test.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = ROOT / "embeddings"

DEFAULT_INDEX_PATH = EMBEDDINGS_DIR / "su_courses.index"
DEFAULT_PARQUET_PATH = EMBEDDINGS_DIR / "su_courses.parquet"
DEFAULT_EMBEDDINGS_PATH = EMBEDDINGS_DIR / "su_courses_embeddings.npy"
DEFAULT_ID_MAP_PATH = EMBEDDINGS_DIR / "id_map.json"

MODEL_NAME = "BAAI/bge-base-en-v1.5"
RERANKER_NAME = "BAAI/bge-reranker-base"

# Whether the cross-encoder reranker is on by default.
#
# Kept ON: the full two-stage pipeline is FAISS bi-encoder retrieval ->
# BAAI/bge-reranker-base cross-encoder re-ranking. Note the retrieval evaluation
# (eval/run_retrieval.py) found the reranker slightly *hurt* metrics on these
# short, topical course queries (e.g. Hit@1 0.97 -> 0.91, MRR@10 0.985 -> 0.956)
# and adds ~0.5 s latency; run_retrieval reports both ON and OFF so the trade-off
# stays visible. Set to False to make bi-encoder order the default. If the
# reranker model is unavailable, retrieval transparently falls back to
# bi-encoder order.
RERANK_DEFAULT = True

# BGE query-side instruction prefix (passages do NOT use this).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Default first-stage multiplier: retrieve 4× more than k, then re-rank down to k.
DEFAULT_FIRST_STAGE_MULTIPLIER = 4


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #

@dataclass
class RetrievalResult:
    """One hit returned by :py:meth:`Retriever.retrieve`."""

    rank: int                        # 1-based rank in result list
    score: float                     # bi-encoder cosine similarity ∈ [-1, 1]
    reranker_score: float | None     # cross-encoder logit (higher = more relevant); None if not re-ranked
    code: str                        # "CS 305"
    title: str
    description: str
    subj: str
    num: int
    level: int                       # num // 100  (1-4)
    su_credit: float | None
    ects: float | None
    has_prereq: bool
    prereq_text: str
    programs: list[str]
    sections: list[str]
    seasons_offered: list[str]
    is_active: bool

    def to_dict(self) -> dict:
        d = {
            "rank": self.rank,
            "score": round(self.score, 4),
            "code": self.code,
            "title": self.title,
            "description": self.description[:300] + ("…" if len(self.description) > 300 else ""),
            "su_credit": self.su_credit,
            "has_prereq": self.has_prereq,
            "prereq_text": self.prereq_text,
            "seasons_offered": self.seasons_offered,
            "programs": self.programs,
            "sections": self.sections,
        }
        if self.reranker_score is not None:
            d["reranker_score"] = round(self.reranker_score, 4)
        return d

    def __repr__(self) -> str:
        rerank_part = (
            f", rerank={self.reranker_score:.3f}" if self.reranker_score is not None else ""
        )
        return (
            f"[{self.rank}] {self.code} — {self.title} "
            f"(bienc={self.score:.3f}{rerank_part}, credits={self.su_credit})"
        )


# --------------------------------------------------------------------------- #
# Filter spec
# --------------------------------------------------------------------------- #

@dataclass
class RetrieverFilter:
    """All optional pre-search filters.

    All filters are ANDed together. A ``None`` value means "no restriction".

    Parameters
    ----------
    candidate_pool:
        Restrict to this exact set of course codes (e.g. the eligible set
        produced by ``scheduler.eligibility.eligible_courses``). If *None*,
        the full catalog is searched.
    subj:
        Keep only courses whose ``subj`` is in this list.
        Example: ``["CS", "IE"]``.
    level_min / level_max:
        Filter by course level (``num // 100``).
        E.g. ``level_min=3`` keeps 300-level and above.
    programs:
        Keep only courses listed under at least one of these program codes.
        Example: ``["BSCS-DM"]``.
    sections:
        Keep only courses whose ``sections`` list intersects this set.
        Example: ``["Required", "Core Elective"]``.
    seasons:
        Keep only courses whose ``seasons_offered`` list intersects this set.
        Example: ``["Fall"]``.
    active_only:
        When ``True``, drop courses whose ``is_active`` flag is ``False``.
    has_prereq:
        ``True`` → only courses that have a prereq.
        ``False`` → only courses without a prereq.
        ``None`` → no restriction.
    """

    candidate_pool: set[str] | None = None
    subj: list[str] | None = None
    level_min: int | None = None
    level_max: int | None = None
    programs: list[str] | None = None
    sections: list[str] | None = None
    seasons: list[str] | None = None
    active_only: bool = True
    has_prereq: bool | None = None


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #

class Retriever:
    """Loads the FAISS index once and serves queries.

    Instantiate via :py:meth:`Retriever.load` rather than ``__init__``
    directly so the path defaults are respected.

    Example
    -------
    >>> r = Retriever.load()
    >>> hits = r.retrieve("graph algorithms", k=5)
    >>> for h in hits:
    ...     print(h)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        id_map: dict[int, str],
        model,            # SentenceTransformer bi-encoder (or None)
        index,            # faiss.Index (or None)
        reranker=None,    # CrossEncoder (or None)
    ) -> None:
        self._df = df
        self._embeddings = embeddings  # (N, D) float32, L2-normalised
        self._id_map = id_map          # {row_int: course_code}
        self._code_to_row: dict[str, int] = {v: k for k, v in id_map.items()}
        self._model = model
        # ``_index`` is a faiss.IndexFlatIP over the full corpus (ids 0..N-1 match
        # the embedding/parquet row order). Stage-1 search runs THROUGH it: every
        # query filters to an eligible subset first, and ``_search`` restricts the
        # FAISS search to those row ids via a faiss IDSelector. IndexFlatIP is
        # exact (brute-force inner product), so results equal the numpy dot used
        # as a fallback when faiss/the index is unavailable.
        self._index = index
        self.search_backend = "faiss-flat-ip" if index is not None else "numpy-exact"
        self._reranker = reranker

    # ------------------------------------------------------------------ #
    # Factory
    # ------------------------------------------------------------------ #

    @classmethod
    def load(
        cls,
        index_path: str | Path = DEFAULT_INDEX_PATH,
        parquet_path: str | Path = DEFAULT_PARQUET_PATH,
        embeddings_path: str | Path = DEFAULT_EMBEDDINGS_PATH,
        id_map_path: str | Path = DEFAULT_ID_MAP_PATH,
        load_model: bool = True,
        load_reranker: bool = True,
        device: str = "cpu",
        local_files_only: bool = True,
    ) -> "Retriever":
        """Load all artifacts from disk and return a ready Retriever.

        Parameters
        ----------
        load_model:
            Set to ``False`` to skip loading the bi-encoder (useful in tests
            or when supplying pre-encoded vectors via
            :py:meth:`retrieve_by_vector`).
        load_reranker:
            Set to ``False`` to skip loading the cross-encoder re-ranker.
            ``retrieve()`` will then return bi-encoder order only.
        device:
            ``"cuda"`` or ``"cpu"``.  Applied to both models.
        local_files_only:
            Load Hugging Face models from the local cache without network
            checks. Run ``python -m scheduler.build_faiss_index`` once to
            download/cache the models.
        """
        # DataFrame ----------------------------------------------------------
        df = pd.read_parquet(Path(parquet_path))

        # Normalised embedding matrix ----------------------------------------
        embeddings = np.load(Path(embeddings_path)).astype("float32")

        # id map {int-key str → course code} ---------------------------------
        raw_map = json.loads(Path(id_map_path).read_text(encoding="utf-8"))
        id_map: dict[int, str] = {int(k): v for k, v in raw_map.items()}

        # ⚠️  Import order matters on macOS (Apple Silicon + faiss-cpu):
        # sentence-transformers (PyTorch / Accelerate) must be imported BEFORE
        # faiss or the process segfaults due to a duplicate OpenBLAS initialisation.
        # Always load both ST models first, then open the FAISS index.

        # Bi-encoder ---------------------------------------------------------
        model = None
        if load_model:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                print(f"[Retriever] Loading bi-encoder: {MODEL_NAME}")
                model = SentenceTransformer(
                    MODEL_NAME,
                    device=device,
                    local_files_only=local_files_only,
                )
            except ImportError:
                print(
                    "[Retriever] sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
            except Exception as exc:
                print(
                    f"[Retriever] Could not load bi-encoder ({exc}). "
                    "Run: python -m scheduler.build_faiss_index"
                )

        # Cross-encoder re-ranker --------------------------------------------
        reranker = None
        if load_reranker:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore
                print(f"[Retriever] Loading cross-encoder re-ranker: {RERANKER_NAME}")
                reranker = CrossEncoder(
                    RERANKER_NAME,
                    device=device,
                    local_files_only=local_files_only,
                )
            except ImportError:
                print("[Retriever] sentence-transformers not installed — no re-ranker.")
            except Exception as exc:
                print(f"[Retriever] Could not load re-ranker ({exc}); skipping.")

        # FAISS index (after ST models — see macOS note above) ---------------
        index = None
        try:
            import faiss  # type: ignore
            index = faiss.read_index(str(Path(index_path)))
        except ImportError:
            print(
                "[Retriever] faiss not installed — falling back to numpy dot product. "
                "Install with: pip install faiss-cpu"
            )
        except Exception as exc:
            print(f"[Retriever] Could not load FAISS index ({exc}); using numpy fallback.")

        return cls(
            df=df, embeddings=embeddings, id_map=id_map,
            model=model, index=index, reranker=reranker,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_mask(self, filt: RetrieverFilter) -> np.ndarray:
        """Return a boolean mask (length N) of rows passing all filters."""
        df = self._df
        mask = np.ones(len(df), dtype=bool)

        if filt.active_only:
            mask &= df["is_active"].to_numpy(dtype=bool)

        if filt.candidate_pool is not None:
            pool_mask = df["code"].isin(filt.candidate_pool).to_numpy(dtype=bool)
            mask &= pool_mask

        if filt.subj is not None:
            subj_set = set(filt.subj)
            mask &= df["subj"].isin(subj_set).to_numpy(dtype=bool)

        if filt.level_min is not None:
            mask &= (df["level"].to_numpy() >= filt.level_min)

        if filt.level_max is not None:
            mask &= (df["level"].to_numpy() <= filt.level_max)

        if filt.has_prereq is not None:
            mask &= (df["has_prereq"].to_numpy(dtype=bool) == filt.has_prereq)

        if filt.programs is not None:
            prog_set = set(filt.programs)
            mask &= df["programs"].apply(
                lambda ps: bool(prog_set & set(ps))
            ).to_numpy(dtype=bool)

        if filt.sections is not None:
            sec_set = set(filt.sections)
            mask &= df["sections"].apply(
                lambda ss: bool(sec_set & set(ss))
            ).to_numpy(dtype=bool)

        if filt.seasons is not None:
            sea_set = set(filt.seasons)
            mask &= df["seasons_offered"].apply(
                lambda s: bool(sea_set & set(s))
            ).to_numpy(dtype=bool)

        return mask

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a text query to a normalised float32 vector."""
        if self._model is None:
            raise RuntimeError(
                "No model loaded. Re-create with load_model=True, or call "
                "retrieve_by_vector() with a pre-computed vector."
            )
        vec = self._model.encode(
            [BGE_QUERY_PREFIX + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        return vec  # shape (1, D)

    def _search(self, query_vec: np.ndarray, row_ids: np.ndarray, k: int
                ) -> tuple[np.ndarray, np.ndarray]:
        """Score query_vec against the subset of rows in row_ids.

        Returns ``(scores, local_ranks)`` — both length ``min(k, |row_ids|)``,
        where ``local_ranks`` index into ``row_ids`` (the caller maps them back
        to global rows). The embeddings are L2-normalised, so inner product is
        cosine similarity.

        Primary path: query the FAISS ``IndexFlatIP`` restricted to ``row_ids``
        with a faiss ``IDSelector`` — this honours the eligibility filter while
        running through the index. Because the index is exact (flat), it returns
        the same top-k as the numpy fallback used when faiss or the index is
        unavailable. See ``search_backend``.
        """
        k = min(k, len(row_ids))
        if k == 0:
            return np.array([]), np.array([], dtype=int)

        # Primary path — FAISS, restricted to the eligible row ids.
        if self._index is not None:
            try:
                import faiss  # type: ignore

                q = np.ascontiguousarray(query_vec[:1], dtype="float32")
                ids64 = np.asarray(row_ids, dtype="int64")
                params = faiss.SearchParameters()
                params.sel = faiss.IDSelectorBatch(ids64)
                fscores, fids = self._index.search(q, k, params=params)
                # Map FAISS global ids back to local positions within row_ids.
                pos = {int(g): i for i, g in enumerate(row_ids)}
                out_scores: list[float] = []
                out_local: list[int] = []
                for s, g in zip(fscores[0], fids[0]):
                    if g == -1:
                        continue  # faiss pads with -1 when fewer than k hits
                    out_scores.append(float(s))
                    out_local.append(pos[int(g)])
                return np.asarray(out_scores), np.asarray(out_local, dtype=int)
            except Exception as exc:  # pragma: no cover - defensive fallback
                print(f"[Retriever] FAISS search failed ({exc}); using numpy fallback.")

        # Fallback — exact numpy dot product over the filtered subset.
        sub_emb = self._embeddings[row_ids]          # (M, D)
        scores = (sub_emb @ query_vec[0]).astype(float)  # (M,)
        top_local = np.argsort(scores)[::-1][:k]
        return scores[top_local], top_local

    def _row_to_result(
        self,
        rank: int,
        score: float,
        row: pd.Series,
        reranker_score: float | None = None,
    ) -> RetrievalResult:
        return RetrievalResult(
            rank=rank,
            score=float(score),
            reranker_score=reranker_score,
            code=row["code"],
            title=row["title"],
            description=row["description"],
            subj=row["subj"],
            num=int(row["num"]),
            level=int(row["level"]),
            su_credit=row["su_credit"],
            ects=row["ects"] if not pd.isna(row["ects"]) else None,
            has_prereq=bool(row["has_prereq"]),
            prereq_text=row["prereq_text"],
            programs=list(row["programs"]),
            sections=list(row["sections"]),
            seasons_offered=list(row["seasons_offered"]),
            is_active=bool(row["is_active"]),
        )

    def _rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        k: int,
    ) -> list[RetrievalResult]:
        """Re-rank *candidates* with the cross-encoder and return top-k.

        The cross-encoder receives ``(query, embedding_text)`` pairs and
        produces a raw logit (higher = more relevant). Results are sorted
        descending by logit and ranks are reassigned from 1.

        If the cross-encoder is not loaded, returns the first *k* candidates
        unchanged (bi-encoder order).
        """
        if self._reranker is None or not candidates:
            # No reranker — just truncate to k
            for i, r in enumerate(candidates[:k], start=1):
                r.rank = i
            return candidates[:k]

        # Build (query, passage) pairs using the same embedding_text field.
        codes = [r.code for r in candidates]
        code_to_row = {code: self._df.iloc[self._code_to_row[code]] for code in codes}
        pairs = [(query, code_to_row[r.code]["embedding_text"]) for r in candidates]

        logits: np.ndarray = self._reranker.predict(pairs)  # shape (N,)

        # Attach reranker score and sort descending.
        for result, logit in zip(candidates, logits):
            result.reranker_score = float(logit)

        ranked = sorted(candidates, key=lambda r: r.reranker_score, reverse=True)[:k]
        for i, r in enumerate(ranked, start=1):
            r.rank = i
        return ranked

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        k: int = 10,
        filt: RetrieverFilter | None = None,
        rerank: bool = RERANK_DEFAULT,
        first_stage_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Retrieval: bi-encoder cosine search, optional cross-encoder re-rank.

        Parameters
        ----------
        query:
            Free-text interest description. E.g. "intro to machine learning".
        k:
            Final number of results to return.
        filt:
            Optional :py:class:`RetrieverFilter`. When *None*, only the
            ``active_only=True`` default is applied.
        rerank:
            When ``True``, re-rank the first-stage candidates with the
            cross-encoder. Defaults to ``RERANK_DEFAULT`` (True): the full
            two-stage pipeline is the production default. Automatically falls
            back to bi-encoder order if no reranker was loaded. (The eval
            ablation in ``eval/run_retrieval.py`` still reports both ON and OFF.)
        first_stage_k:
            How many candidates to retrieve in stage 1 before re-ranking.
            Defaults to ``DEFAULT_FIRST_STAGE_MULTIPLIER × k`` (= 4k).
            Ignored when ``rerank=False``.

        Returns
        -------
        List of :py:class:`RetrievalResult`, sorted by descending
        cross-encoder score (or bi-encoder score when reranking is off).
        """
        filt = filt or RetrieverFilter()
        query_vec = self._encode_query(query)
        return self.retrieve_by_vector(
            query_vec, k=k, filt=filt,
            rerank=rerank, first_stage_k=first_stage_k,
            _query_text=query,
        )

    def retrieve_by_vector(
        self,
        query_vec: np.ndarray,
        k: int = 10,
        filt: RetrieverFilter | None = None,
        rerank: bool = RERANK_DEFAULT,
        first_stage_k: int | None = None,
        _query_text: str = "",
    ) -> list[RetrievalResult]:
        """Same as :py:meth:`retrieve` but accepts a pre-encoded vector.

        ``query_vec`` must be shape ``(1, D)`` or ``(D,)`` and L2-normalised.
        ``_query_text`` is required when ``rerank=True`` so the cross-encoder
        has a text query to compare against; pass the original string.
        """
        if query_vec.ndim == 1:
            query_vec = query_vec[np.newaxis, :]  # (1, D)

        filt = filt or RetrieverFilter()
        mask = self._build_mask(filt)
        row_ids = np.where(mask)[0]

        if len(row_ids) == 0:
            return []

        # Stage 1 — bi-encoder: retrieve more candidates than needed so the
        # cross-encoder has a good pool to re-rank.
        stage1_k = (
            first_stage_k
            if first_stage_k is not None
            else min(k * DEFAULT_FIRST_STAGE_MULTIPLIER, len(row_ids))
        ) if rerank else k

        scores, local_idxs = self._search(query_vec, row_ids, stage1_k)

        candidates: list[RetrievalResult] = []
        for rank, (local_idx, score) in enumerate(zip(local_idxs, scores), start=1):
            global_idx = row_ids[local_idx]
            row = self._df.iloc[global_idx]
            candidates.append(self._row_to_result(rank, score, row))

        # Stage 2 — cross-encoder: re-rank and truncate to k.
        if rerank:
            return self._rerank(_query_text, candidates, k)

        return candidates

    def get_course(self, code: str) -> RetrievalResult | None:
        """Fetch a single course record by code, no scoring."""
        if code not in self._code_to_row:
            return None
        idx = self._code_to_row[code]
        row = self._df.iloc[idx]
        return self._row_to_result(rank=1, score=1.0, row=row)

    def filter_only(self, filt: RetrieverFilter) -> list[str]:
        """Return the list of course codes passing *filt* (no semantic scoring).

        Useful for building a ``candidate_pool`` to feed into the eligibility
        module before doing a semantic search.
        """
        mask = self._build_mask(filt)
        return self._df.loc[mask, "code"].tolist()

    # ------------------------------------------------------------------ #
    # Convenience wrappers used by the planner agent
    # ------------------------------------------------------------------ #

    def eligible_and_retrieve(
        self,
        query: str,
        eligible_set: Iterable[str],
        k: int = 10,
        seasons: list[str] | None = None,
        subj: list[str] | None = None,
        sections: list[str] | None = None,
        rerank: bool = RERANK_DEFAULT,
    ) -> list[RetrievalResult]:
        """Retrieve within *eligible_set* — the primary planner entry-point.

        Combines eligibility filtering (from ``scheduler.eligibility``) with
        bi-encoder retrieval and cross-encoder re-ranking in one call.

        Parameters
        ----------
        eligible_set:
            Output of ``scheduler.eligibility.eligible_courses(graph, student)``.
        seasons:
            E.g. ``["Fall"]`` — keeps only courses offered in that season.
        subj, sections:
            Further narrow the candidate space.
        rerank:
            Pass ``False`` to skip cross-encoder re-ranking (faster, less accurate).
        """
        filt = RetrieverFilter(
            candidate_pool=set(eligible_set),
            subj=subj,
            sections=sections,
            seasons=seasons,
            active_only=True,
        )
        return self.retrieve(query, k=k, filt=filt, rerank=rerank)


# --------------------------------------------------------------------------- #
# CLI smoke test
# --------------------------------------------------------------------------- #

def _smoke_test(args: argparse.Namespace) -> None:
    print(f"Loading Retriever (bi-encoder + cross-encoder) from {EMBEDDINGS_DIR} …")
    r = Retriever.load(device="cpu")
    print("Loaded.\n")

    # --- Show bi-encoder vs re-ranked side-by-side for one query ---
    demo_query = "intro to machine learning and neural networks"
    print(f"=== Bi-encoder vs Cross-encoder re-rank: {demo_query!r} ===")
    bi_hits = r.retrieve(demo_query, k=5, rerank=False)
    print("  Bi-encoder order:")
    for h in bi_hits:
        print(f"    {h}")
    reranked = r.retrieve(demo_query, k=5, rerank=True)
    print("  Cross-encoder re-ranked:")
    for h in reranked:
        print(f"    {h}")
    print()

    # --- Standard queries with re-ranking ---
    queries = [
        "operating systems and computer architecture",
        "financial accounting for business",
        "differential equations and linear algebra",
        "algorithms and data structures",
    ]
    for q in queries:
        print(f"Query: {q!r}")
        hits = r.retrieve(q, k=5, rerank=True)
        for h in hits:
            print(f"  {h}")
        print()

    # Filter demo: only CS courses, level >= 3
    print("=== CS 300+ courses matching 'parallel computing' (re-ranked) ===")
    hits = r.retrieve(
        "parallel and distributed computing",
        k=5,
        filt=RetrieverFilter(subj=["CS"], level_min=3),
        rerank=True,
    )
    for h in hits:
        print(f"  {h}")
    print()

    # eligible_and_retrieve demo with a synthetic eligible set
    eligible_fake = {
        "CS 301", "CS 303", "CS 306", "CS 308", "CS 310",
        "MATH 306", "IE 310", "EE 301", "PHYS 301",
    }
    print("=== Planner entry-point: eligible_and_retrieve (re-ranked) ===")
    hits = r.eligible_and_retrieve(
        query="algorithms",
        eligible_set=eligible_fake,
        k=5,
        seasons=["Fall"],
        rerank=True,
    )
    for h in hits:
        print(f"  {h}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="SuSchedule-r FAISS retriever — smoke test / query CLI"
    )
    ap.add_argument("--query", "-q", help="Run a single retrieval query and exit")
    ap.add_argument("--k", type=int, default=5, help="Number of results (default 5)")
    ap.add_argument("--subj", nargs="*", help="Restrict to subject codes")
    ap.add_argument("--seasons", nargs="*", help="Restrict to seasons (Fall/Spring/Summer)")
    ap.add_argument("--level-min", type=int, dest="level_min")
    ap.add_argument("--level-max", type=int, dest="level_max")
    ap.add_argument("--no-active-filter", action="store_true", dest="no_active_filter")
    args = ap.parse_args()

    if args.query:
        r = Retriever.load(device="cpu")
        filt = RetrieverFilter(
            subj=args.subj,
            level_min=args.level_min,
            level_max=args.level_max,
            seasons=args.seasons,
            active_only=not args.no_active_filter,
        )
        hits = r.retrieve(args.query, k=args.k, filt=filt)
        for h in hits:
            print(json.dumps(h.to_dict(), indent=2))
    else:
        _smoke_test(args)


if __name__ == "__main__":
    main()
