"""Parse SUIS Degree Evaluation HTML files.

The degree evaluation page is a better source of student academic state than
the unofficial transcript for planning: it includes the program requirements
term, evaluation term, category-level progress, and the courses SUIS has
already assigned to each requirement bucket.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from scheduler.graph_selector import normalize_program_code
from scheduler.transcript import PROGRAM_NAME_TO_CODE, term_to_code

COURSE_RE = re.compile(r"^[A-Z]{2,5}\s+\d{3,5}[A-Z]?$")
PASSING_GRADES = {
    "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "S", "T",
}
IN_PROGRESS_GRADES = {"I.P.", "IP", "REGISTERED", "REG"}

SECTION_NAMES = {
    "UNIVERSITY COURSES",
    "CORE ELECTIVES",
    "REQUIRED COURSES",
    "AREA ELECTIVES",
    "FREE ELECTIVES",
    "FACULTY COURSES",
    "ENGINEERING",
    "BASIC SCIENCE",
}


def _clean(text: str | None) -> str:
    return " ".join((text or "").replace("\xa0", " ").split())


def _num(text: str | None) -> float | None:
    text = _clean(text)
    if not text or text == "-":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _status_from_grade(grade: str) -> str:
    grade = _clean(grade).upper()
    if grade in IN_PROGRESS_GRADES:
        return "in_progress"
    if grade in PASSING_GRADES:
        return "completed"
    if grade in {"W", "NA", "F", "U"}:
        return "not_counted"
    return "unknown"


def _section_slug(section: str) -> str:
    return section.lower().replace(" ", "_")


def _parse_header(soup: BeautifulSoup) -> dict[str, Any]:
    table = soup.find("table")
    if table is None:
        raise ValueError("Degree evaluation header table not found.")

    out: dict[str, Any] = {"source": "degree_evaluation_html"}
    for tr in table.find_all("tr", recursive=False):
        cells = [_clean(td.get_text(" ", strip=True)) for td in tr.find_all("td", recursive=False)]
        for i in range(0, len(cells) - 1, 2):
            key = cells[i].rstrip(":").strip().lower()
            value = cells[i + 1].strip()
            if key == "student":
                m = re.match(r"(?P<id>\d+)\s*-\s*(?P<name>.+)", value)
                if m:
                    out["student_id"] = m.group("id")
                    out["name"] = m.group("name").strip()
                else:
                    out["name"] = value
            elif key == "program requirements term":
                out["admit_term_label"] = value
                out["program_requirements_term_label"] = value
                out["admit_term"] = term_to_code(value)
            elif key == "program":
                program_text = value
                minor_match = re.search(r"\(Minor-\s*(.+?)\s*\)", program_text, re.I)
                if minor_match:
                    out["minors"] = [minor_match.group(1).strip()]
                    out["intended_minor"] = out["minors"][0]
                    program_text = re.sub(r"\(Minor-.+?\)", "", program_text, flags=re.I).strip()
                else:
                    out["minors"] = []
                    out["intended_minor"] = None
                out["program_name"] = program_text
                mapped = PROGRAM_NAME_TO_CODE.get(program_text)
                out["program"] = normalize_program_code(mapped or program_text)
            elif key == "evaluation term":
                out["evaluation_term_label"] = value
                out["evaluation_term"] = term_to_code(value)
            elif key == "class":
                out["class_level"] = value
            elif key == "status":
                out["academic_status"] = value
            elif key == "result":
                out["degree_result"] = value

    if not out.get("program"):
        raise ValueError("Could not detect program from degree evaluation HTML.")
    if not out.get("admit_term"):
        raise ValueError("Could not detect Program Requirements Term from degree evaluation HTML.")
    return out


def _parse_general_requirements(soup: BeautifulSoup) -> dict[str, Any]:
    for table in soup.find_all("table"):
        first = _clean(table.get_text(" ", strip=True))
        if not first.startswith("GENERAL PROGRAM REQUIREMENTS"):
            continue
        rows = [
            [_clean(td.get_text(" ", strip=True)) for td in tr.find_all("td", recursive=False)]
            for tr in table.find_all("tr", recursive=False)
        ]
        if len(rows) < 4:
            return {}
        minimum = rows[2]
        completed = rows[3]
        return {
            "minimum_required": {
                "ects": _num(minimum[1] if len(minimum) > 1 else None),
                "su": _num(minimum[2] if len(minimum) > 2 else None),
                "program_gpa": _num(minimum[3] if len(minimum) > 3 else None),
                "cumulative_gpa": _num(minimum[4] if len(minimum) > 4 else None),
            },
            "completed": {
                "ects": _num(completed[1] if len(completed) > 1 else None),
                "su": _num(completed[2] if len(completed) > 2 else None),
                "program_gpa": _num(completed[3] if len(completed) > 3 else None),
                "cumulative_gpa": _num(completed[4] if len(completed) > 4 else None),
            },
        }
    return {}


def _is_leaf_section_table(table) -> bool:
    if table.find_all("table"):
        return False
    rows = table.find_all("tr")
    if not rows:
        return False
    header = _clean(rows[0].get_text(" ", strip=True)).upper()
    return header in SECTION_NAMES


def _section_passed(table) -> bool | None:
    img = table.find("img")
    if img is None:
        return None
    src = str(img.get("src", "")).lower()
    if "s_ok" in src:
        return True
    if "s_not" in src:
        return False
    return None


def _parse_section_table(table) -> tuple[str, dict[str, Any]]:
    rows = table.find_all("tr")
    section = _clean(rows[0].get_text(" ", strip=True)).upper()
    section_title = section.title().replace("Ects", "ECTS")
    header = [_clean(td.get_text(" ", strip=True)) for td in rows[1].find_all("td")]
    has_su = any(h.lower().startswith("su credits") for h in header)

    courses: list[dict[str, Any]] = []
    minimum: dict[str, float | None] = {}
    completed: dict[str, float | None] = {}

    for tr in rows[2:]:
        cells = [_clean(td.get_text(" ", strip=True)) for td in tr.find_all("td", recursive=False)]
        if not cells:
            continue
        first = cells[0]
        if COURSE_RE.match(first):
            code = first.upper()
            row = {
                "code": code,
                "grade": cells[1] if len(cells) > 1 else "",
                "ects": _num(cells[2] if len(cells) > 2 else None),
                "term": cells[4] if has_su and len(cells) > 4 else (cells[3] if len(cells) > 3 else ""),
                "status": _status_from_grade(cells[1] if len(cells) > 1 else ""),
            }
            if has_su:
                row["su"] = _num(cells[3] if len(cells) > 3 else None)
            courses.append(row)
            continue

        label = first.lower()
        if label.startswith("minimum required"):
            if has_su:
                minimum = {
                    "ects": _num(cells[1] if len(cells) > 1 else None),
                    "su": _num(cells[2] if len(cells) > 2 else None),
                    "courses": _num(cells[3] if len(cells) > 3 else None),
                }
            else:
                minimum = {
                    "ects": _num(cells[1] if len(cells) > 1 else None),
                    "courses": _num(cells[2] if len(cells) > 2 else None),
                }
        elif label.startswith("completed"):
            if has_su:
                completed = {
                    "ects": _num(cells[1] if len(cells) > 1 else None),
                    "su": _num(cells[2] if len(cells) > 2 else None),
                    "courses": _num(cells[3] if len(cells) > 3 else None),
                }
            else:
                completed = {
                    "ects": _num(cells[1] if len(cells) > 1 else None),
                    "courses": _num(cells[2] if len(cells) > 2 else None),
                }

    return section_title, {
        "passed": _section_passed(table),
        "has_su": has_su,
        "minimum_required": minimum,
        "completed": completed,
        "courses": courses,
    }


def parse_degree_evaluation(path: str | Path) -> dict[str, Any]:
    html = Path(path).read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    profile = _parse_header(soup)
    profile["general_requirements"] = _parse_general_requirements(soup)

    sections: dict[str, dict[str, Any]] = {}
    for table in soup.find_all("table"):
        if not _is_leaf_section_table(table):
            continue
        section, data = _parse_section_table(table)
        sections[section] = data
    profile["sections"] = sections

    completed: set[str] = set()
    in_progress: set[str] = set()
    course_category_map: dict[str, list[str]] = {}
    course_records: dict[str, list[dict[str, Any]]] = {}
    for section, data in sections.items():
        for row in data.get("courses", []):
            code = row["code"]
            course_category_map.setdefault(code, [])
            if section not in course_category_map[code]:
                course_category_map[code].append(section)
            record = dict(row)
            record["section"] = section
            course_records.setdefault(code, []).append(record)
            if row["status"] == "completed":
                completed.add(code)
            elif row["status"] == "in_progress":
                in_progress.add(code)

    profile["completed"] = sorted(completed)
    profile["completed_for_eligibility"] = sorted(completed)
    profile["in_progress"] = sorted(in_progress)
    profile["course_category_map"] = dict(sorted(course_category_map.items()))
    profile["course_records"] = dict(sorted(course_records.items()))
    profile["current_semester"] = 0

    general_completed = profile.get("general_requirements", {}).get("completed", {})
    profile["cumulative_credits"] = general_completed.get("su") or 0.0
    profile["cgpa"] = general_completed.get("cumulative_gpa") or 0.0
    return profile


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("html", help="Path to SUIS Degree Evaluation HTML")
    ap.add_argument("-o", "--out", help="Where to write parsed JSON")
    args = ap.parse_args()

    profile = parse_degree_evaluation(args.html)
    text = json.dumps(profile, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
