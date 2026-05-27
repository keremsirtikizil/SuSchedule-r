"""
Run the prereq parser over a catalog file and report how well it covered the
data.

Buckets each course's prereq_text into one of:
  - empty:    no prereq
  - strict:   parsed cleanly by parse_prereq()
  - lenient:  parse_prereq() raised, but parse_prereq_lenient() produced a
              clean tree (no unparsed leaves)
  - partial:  lenient parser produced a tree containing unparsed leaves
  - failed:   strict raised AND lenient returned only unparsed

Writes data/unparsed_prereqs.txt with one residual span per line so a human
can scan for new patterns worth adding to the parser.

Also runs a regression check: every course in the BSCS-DM catalog must
strict-parse exactly as it does today.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scraper.prereq_parser import (
    has_unparsed,
    parse_prereq,
    parse_prereq_lenient,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def classify(text: str | None) -> tuple[str, dict | None, str | None]:
    """Return (bucket, parsed_tree, residual_text_or_None)."""
    if text is None or not text.strip() or text.strip() in {"__"} or text.strip().startswith("__"):
        return "empty", None, None
    try:
        strict = parse_prereq(text)
    except Exception as exc:
        strict = None
        strict_error = str(exc)
    else:
        return "strict", strict, None

    lenient = parse_prereq_lenient(text)
    if lenient is None:
        return "empty", None, None
    if not has_unparsed(lenient):
        return "lenient", lenient, None
    # Walk leaves to extract the raw residuals for the report.
    residuals: list[str] = []

    def walk(node: dict) -> None:
        if node.get("unparsed"):
            residuals.append(node.get("raw", ""))
            return
        for op in node.get("operands", []):
            walk(op)

    walk(lenient)
    if any(o.get("course") for o in (lenient.get("operands") or [lenient])):
        bucket = "partial"
    else:
        bucket = "failed"
    return bucket, lenient, " | ".join(residuals)


def audit(catalog_path: Path, out_residuals: Path) -> None:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    courses = catalog["courses"]
    counts: Counter[str] = Counter()
    residuals: list[tuple[str, str]] = []
    for cr in courses:
        bucket, _, residual = classify(cr.get("prereq_text"))
        counts[bucket] += 1
        if residual:
            residuals.append((cr["code"], residual))

    print(f"Audit of {catalog_path.name}: {len(courses)} courses")
    for bucket in ("empty", "strict", "lenient", "partial", "failed"):
        print(f"  {bucket:>8}: {counts[bucket]}")

    if residuals:
        with out_residuals.open("w", encoding="utf-8") as f:
            for code, residual in sorted(residuals):
                f.write(f"{code}\t{residual}\n")
        print(f"  wrote {out_residuals}  ({len(residuals)} entries)")
    else:
        if out_residuals.exists():
            out_residuals.unlink()
        print("  no residual unparsed spans — nothing to write")


def regression_check(catalog_path: Path) -> int:
    """Confirm every CS prereq still parses strictly. Returns failure count."""
    if not catalog_path.exists():
        print(f"skip regression: {catalog_path} not found")
        return 0
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    fails = 0
    for cr in catalog["courses"]:
        text = cr.get("prereq_text") or ""
        if not text.strip() or text.strip().startswith("__"):
            continue
        try:
            parse_prereq(text)
        except Exception as exc:
            fails += 1
            print(f"  regression FAIL {cr['code']}: {exc}")
    if fails == 0:
        print(f"regression OK ({catalog_path.name}): all prereq_text strict-parses")
    return fails


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--catalog",
        default=str(DATA_DIR / "SU_full_catalog.json"),
        help="catalog file to audit",
    )
    ap.add_argument(
        "--residuals",
        default=str(DATA_DIR / "unparsed_prereqs.txt"),
        help="where to write residual unparsed text",
    )
    ap.add_argument(
        "--regression",
        default=str(DATA_DIR / "BSCS-DM_202601_catalog.json"),
        help="catalog used as a strict-parse regression baseline",
    )
    args = ap.parse_args()

    audit(Path(args.catalog), Path(args.residuals))
    regression_check(Path(args.regression))


if __name__ == "__main__":
    main()
