"""
Stage 2 of the full SU catalog scrape: fetch every course detail page from
data/discovery_manifest.json and assemble data/SU_full_catalog.json.

Course detail URLs are term-agnostic and subject-agnostic — the same
(subj, num) gives the same content regardless of which degree page we found
it under. So we fetch each pair once (cached in courses/<SUBJ>_<NUM>.html)
and attach the union of program_sections discovered in Stage 1.

Run:
  python -m scraper.discover --all          # produces discovery_manifest.json
  python -m scraper.scrape_full             # consumes it, writes SU_full_catalog.json

Resume-safe: existing HTML in courses/ is reused; only missing pairs hit
the network.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scraper.scrape_cs import (
    DATA_DIR,
    fetch_course_page,
    parse_course_page,
)


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run 'python -m scraper.discover --all' first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def scrape_all(manifest_path: Path, out_path: Path, verbose: bool = True) -> dict:
    manifest = load_manifest(manifest_path)
    items = list(manifest["courses"].items())
    total = len(items)
    if verbose:
        print(f"[Stage 2] fetching {total} course detail pages")

    courses: list[dict] = []
    errors: list[dict] = []
    for i, (code, meta) in enumerate(items, start=1):
        subj = meta["subj"]
        num = meta["num"]
        try:
            html = fetch_course_page(subj, num)
            info = parse_course_page(html, subj, num)
        except Exception as exc:
            errors.append({"code": code, "error": str(exc)})
            info = {
                "code": code,
                "subj": subj,
                "num": num,
                "error": str(exc),
            }
        info["program_sections"] = meta.get("program_sections", [])
        courses.append(info)
        if verbose and i % 50 == 0:
            print(f"  .. {i}/{total}  (errors so far: {len(errors)})")

    catalog = {
        "source": "SU_full",
        "degrees": manifest.get("degrees", []),
        "terms": manifest.get("terms", []),
        "course_count": len(courses),
        "error_count": len(errors),
        "errors": errors,
        "courses": courses,
    }
    out_path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if verbose:
        print(f"[Stage 2] wrote {out_path}  ({len(courses)} courses, {len(errors)} errors)")
    return catalog


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default=str(DATA_DIR / "discovery_manifest.json"),
        help="Stage 1 output to consume",
    )
    ap.add_argument(
        "--out",
        default=str(DATA_DIR / "SU_full_catalog.json"),
        help="where to write the unified catalog",
    )
    args = ap.parse_args()
    scrape_all(Path(args.manifest), Path(args.out))


if __name__ == "__main__":
    main()
