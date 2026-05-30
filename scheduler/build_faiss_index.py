"""Build local FAISS artifacts for catalog RAG."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scheduler.retriever import (
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_ID_MAP_PATH,
    DEFAULT_INDEX_PATH,
    DEFAULT_PARQUET_PATH,
    MODEL_NAME,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG_PATH = ROOT / "data" / "SU_full_catalog.json"


def _term_season(label: str) -> str | None:
    label = (label or "").strip()
    if label.startswith("Fall"):
        return "Fall"
    if label.startswith("Spring"):
        return "Spring"
    if label.startswith("Summer"):
        return "Summer"
    return None


def _term_sort_key(label: str) -> tuple[int, int]:
    match = re.search(r"(20\d{2})-(20\d{2})", label or "")
    if not match:
        return (0, 0)
    end = int(match.group(2))
    order = {"Fall": 1, "Spring": 2, "Summer": 3}.get(_term_season(label) or "", 0)
    return (end, order)


def _parse_ects(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*ECTS", str(text or ""))
    return float(match.group(1)) if match else None


def _build_dataframe_from_catalog(catalog_path: Path) -> pd.DataFrame:
    obj = json.loads(catalog_path.read_text(encoding="utf-8"))
    courses = obj.get("courses", obj)
    if isinstance(courses, dict):
        courses = list(courses.values())

    rows = []
    for course in courses:
        code = str(course.get("code", "")).strip()
        if not code:
            continue
        subj = str(course.get("subj") or code.split()[0])
        num_text = str(course.get("num") or re.sub(r"\D", "", code))
        try:
            num = int(num_text)
        except ValueError:
            num = 0
        title = str(course.get("title") or "").strip()
        description = str(course.get("description") or "").strip()
        prereq_text = str(course.get("prereq_text") or "").strip()
        has_prereq = bool(prereq_text and prereq_text != "__")

        program_sections = course.get("program_sections") or []
        programs = sorted({
            str(item.get("program", "")).strip()
            for item in program_sections
            if item.get("program")
        })
        sections = sorted({
            str(item.get("section", "")).strip()
            for item in program_sections
            if item.get("section")
        })

        offered_terms = course.get("offered_terms") or []
        seasons = sorted({
            season for season in (_term_season(item.get("term", "")) for item in offered_terms)
            if season
        })
        last_label = ""
        if offered_terms:
            last_label = max(
                (str(item.get("term", "")) for item in offered_terms),
                key=_term_sort_key,
            )

        rows.append({
            "code": code,
            "subj": subj,
            "num": num,
            "level": num // 100 if num else 0,
            "title": title,
            "description": description,
            "embedding_text": f"{code} - {title}\n{description}",
            "su_credit": course.get("su_credit"),
            "ects": _parse_ects(course.get("ects_text", "")),
            "has_prereq": has_prereq,
            "prereq_text": "" if prereq_text == "__" else prereq_text,
            "programs": programs,
            "sections": sections,
            "seasons_offered": seasons,
            "last_offered_term": last_label,
            "is_active": bool(offered_terms),
        })

    df = pd.DataFrame(rows).sort_values("code").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No courses found in {catalog_path}")
    return df


def _load_passages(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    required = {"code", "embedding_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{parquet_path} is missing required columns: {sorted(missing)}")
    return df.reset_index(drop=True)


def build_index(
    catalog_path: Path | None = DEFAULT_CATALOG_PATH,
    parquet_path: Path = DEFAULT_PARQUET_PATH,
    embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH,
    index_path: Path = DEFAULT_INDEX_PATH,
    id_map_path: Path = DEFAULT_ID_MAP_PATH,
    model_name: str = MODEL_NAME,
    batch_size: int = 32,
    device: str = "cpu",
    local_files_only: bool = True,
) -> None:
    if catalog_path and catalog_path.exists():
        df = _build_dataframe_from_catalog(catalog_path)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        print(f"[FAISS] Refreshed {parquet_path} from {catalog_path} ({len(df)} rows)")
    else:
        df = _load_passages(parquet_path)

    from sentence_transformers import SentenceTransformer  # type: ignore

    print(f"[FAISS] Loading bi-encoder: {model_name} ({device})")
    model = SentenceTransformer(
        model_name,
        device=device,
        local_files_only=local_files_only,
    )
    passages = df["embedding_text"].fillna("").astype(str).tolist()
    print(f"[FAISS] Encoding {len(passages)} course passages")
    embeddings = model.encode(
        passages,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    import faiss  # type: ignore

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    faiss.write_index(index, str(index_path))
    id_map = {str(i): code for i, code in enumerate(df["code"].astype(str))}
    id_map_path.write_text(json.dumps(id_map, indent=2) + "\n", encoding="utf-8")

    print(f"[FAISS] Wrote {embeddings_path} {embeddings.shape}")
    print(f"[FAISS] Wrote {index_path} ({index.ntotal} vectors)")
    print(f"[FAISS] Wrote {id_map_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SuSchedule-r FAISS artifacts")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--no-refresh-parquet", action="store_true")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET_PATH)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--id-map", type=Path, default=DEFAULT_ID_MAP_PATH)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face downloads. By default the builder uses the local model cache only.",
    )
    args = parser.parse_args()

    build_index(
        catalog_path=None if args.no_refresh_parquet else args.catalog,
        parquet_path=args.parquet,
        embeddings_path=args.embeddings,
        index_path=args.index,
        id_map_path=args.id_map,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        local_files_only=not args.allow_download,
    )


if __name__ == "__main__":
    main()
