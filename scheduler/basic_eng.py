"""Basic Science / Engineering ECTS lookup.

The source PDF lists how each FENS course splits into Engineering and Basic
Science ECTS for students whose first admit term is after 2013-2014.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON = ROOT / "data" / "basic_science_engineering_ects.json"


def _num(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "-":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _course_code(subject: str, number: str) -> str:
    return f"{subject.strip().upper()} {number.strip().upper()}"


def parse_basic_eng_pdf(path: str | Path) -> dict[str, Any]:
    import pdfplumber

    path = Path(path)
    courses: list[dict[str, Any]] = []
    seen: set[str] = set()
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                for row in table[2:]:
                    if not row or len(row) < 8:
                        continue
                    subject = str(row[0] or "").strip().upper()
                    number = str(row[1] or "").strip().upper()
                    compact = str(row[2] or "").strip().upper()
                    if not subject or not number:
                        continue
                    if not re.fullmatch(r"[A-Z]{2,5}", subject):
                        continue
                    if not re.fullmatch(r"\d{3,5}[A-Z]?", number):
                        continue
                    code = _course_code(subject, number)
                    if code in seen:
                        continue
                    seen.add(code)
                    courses.append({
                        "code": code,
                        "compact_code": compact,
                        "subject": subject,
                        "number": number,
                        "title": " ".join(str(row[3] or "").split()),
                        "faculty": str(row[4] or "").strip(),
                        "after_2013_2014": {
                            "ects_total": _num(row[5]),
                            "engineering_ects": _num(row[6]),
                            "basic_science_ects": _num(row[7]),
                        },
                        "previous_code": str(row[8] or "").strip() if len(row) > 8 else "",
                        "before_2013_2014": {
                            "ects_total": _num(row[9]) if len(row) > 9 else None,
                            "engineering_ects": _num(row[10]) if len(row) > 10 else None,
                            "basic_science_ects": _num(row[11]) if len(row) > 11 else None,
                        },
                    })

    courses.sort(key=lambda item: (item["subject"], item["number"]))
    return {
        "source": path.name,
        "rule": "first_admit_term_after_2013_2014",
        "courses": courses,
        "by_code": {item["code"]: item for item in courses},
    }


def load_basic_eng(path: str | Path = DEFAULT_JSON) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_basic_eng_split(code: str, path: str | Path = DEFAULT_JSON) -> dict[str, Any] | None:
    data = load_basic_eng(path)
    normalized = re.sub(r"^([A-Z]{2,5})\s*(\d{3,5}[A-Z]?)$", r"\1 \2", code.upper().strip())
    return data.get("by_code", {}).get(normalized)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="Path to Basic Science / Engineering ECTS PDF")
    ap.add_argument("-o", "--out", default=str(DEFAULT_JSON), help="Output JSON path")
    args = ap.parse_args()

    data = parse_basic_eng_pdf(args.pdf)
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(data['courses'])} course splits to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
