r"""
Parse the SU "Basic Science & Engineering" catalog PDF into
``data/eng_sci_credits.json``.

The PDF (``katalog_basic_eng_degerler_<term>_*.pdf``) lists, for every FENS
course, how its ECTS credits are split between **Engineering** and **Basic
Science** content. Sabancı engineering degrees require a minimum number of
Engineering ECTS and Basic-Science ECTS to graduate (the per-degree minimums
appear on the student's Degree Evaluation form, not here). This table is what
lets us compute how much a *candidate* course would contribute toward those
minimums.

Each data row looks like::

    <SUBJ> <NUM> <NOCODE> <TITLE...> FENS <TOTAL> <ENG> <BSCI> <PREV> <T2> <E2> <B2>
                                          \_____ After 2013-2014 _____/      Before 2013-2014

We keep the *After 2013-2014* columns (TOTAL / Engineering / Basic-Science),
which apply to all current cohorts.

Output schema::

    {
      "source": "katalog_basic_eng_degerler_202401_...pdf",
      "term": "202401",
      "course_count": 353,
      "courses": {
        "CS 303": {"ects_total": 7.0, "eng_ects": 6.0, "basic_sci_ects": 1.0},
        ...
      }
    }

Run:
    python -m scraper.build_eng_sci
    python -m scraper.build_eng_sci --pdf path/to/katalog.pdf
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_PATH = DATA_DIR / "eng_sci_credits.json"

DEFAULT_PDF = ROOT / "katalog_basic_eng_degerler_202401_yuklenen_07.05.2025.pdf"

# Row: SUBJ NUM ... FENS <total> <eng> <bsci> ... (we only need the first triple
# of decimal numbers that follow the faculty acronym).
ROW_RE = re.compile(
    r"^(?P<subj>[A-Z]{2,5})\s+(?P<num>\d{2,6}[A-Z]?)\b.*?"
    r"\bFENS\s+"
    r"(?P<total>\d+(?:\.\d+)?)\s+"
    r"(?P<eng>\d+(?:\.\d+)?)\s+"
    r"(?P<bsci>\d+(?:\.\d+)?)\b"
)

TERM_RE = re.compile(r"_(\d{6})_")


def _extract_text(pdf_path: Path) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    return "\n".join(pages)


def parse_pdf(pdf_path: Path) -> dict[str, dict]:
    """Return {course_code: {ects_total, eng_ects, basic_sci_ects}}."""
    text = _extract_text(pdf_path)
    courses: dict[str, dict] = {}
    for line in text.splitlines():
        m = ROW_RE.match(line.strip())
        if not m:
            continue
        code = f"{m.group('subj')} {m.group('num')}"
        total = float(m.group("total"))
        eng = float(m.group("eng"))
        bsci = float(m.group("bsci"))
        # First occurrence wins (the table has no intentional duplicates, but a
        # wrapped title line could in theory re-match).
        courses.setdefault(
            code,
            {"ects_total": total, "eng_ects": eng, "basic_sci_ects": bsci},
        )
    return courses


def build(pdf_path: Path = DEFAULT_PDF, verbose: bool = True) -> dict:
    courses = parse_pdf(pdf_path)
    m = TERM_RE.search(pdf_path.name)
    term = m.group(1) if m else ""
    payload = {
        "source": pdf_path.name,
        "term": term,
        "course_count": len(courses),
        "courses": courses,
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        eng_only = sum(1 for c in courses.values() if c["eng_ects"] > 0)
        sci_only = sum(1 for c in courses.values() if c["basic_sci_ects"] > 0)
        print(f"parsed {len(courses)} courses from {pdf_path.name}")
        print(f"  with Engineering ECTS:   {eng_only}")
        print(f"  with Basic-Science ECTS: {sci_only}")
        print(f"wrote {OUT_PATH}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=str(DEFAULT_PDF), help="path to the catalog PDF")
    args = ap.parse_args()
    build(pdf_path=Path(args.pdf))


if __name__ == "__main__":
    main()
