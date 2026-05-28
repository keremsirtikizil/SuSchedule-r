"""
Parse a Sabancı University **Degree Evaluation** HTML page into the same
student profile dict the rest of the planner consumes (see
``scheduler.transcript`` for the shared schema), plus the per-degree section
requirements that the transcript PDF does not expose.

Why this exists
---------------
The Degree Evaluation form is richer than the plain transcript: in addition to
completed / in-progress courses it states, per section, the **minimum credits
required** and **completed so far** — including the ``ENGINEERING`` and
``BASIC SCIENCE`` ECTS minimums that every engineering degree must satisfy to
graduate. We surface those so the planner can track science/engineering credit
gaps (combined with ``data/eng_sci_credits.json`` for per-course contributions).

The page is a set of ``<table>`` blocks. Each section table opens with a header
cell (e.g. ``UNIVERSITY COURSES``, ``REQUIRED COURSES``, ``ENGINEERING``,
``BASIC SCIENCE``) followed by course rows (Course / Grade / ECTS / [SU] / Term)
and two summary rows: ``Minimum Required`` and ``Completed``.

Output (superset of the transcript schema)::

    {
      "name": "Çağan Çakır",
      "student_id": "00032254",
      "program": "BSCS-DM",
      "program_name": "Computer Science and Engineering",
      "admit_term": "202201",
      "admit_term_label": "Fall 2022-2023",
      "cgpa": 3.86,
      "cumulative_credits": 122.0,          # SU credits completed
      "cumulative_ects": 230.0,
      "minors": ["Business Analytics", "Finance"],
      "intended_minor": "Business Analytics",
      "completed": [...],
      "in_progress": [...],
      "completed_for_eligibility": [...],
      "source": "degree_evaluation",
      "section_requirements": {
        "ENGINEERING":   {"min_ects": 90.0, "completed_ects": 100.0},
        "BASIC SCIENCE": {"min_ects": 60.0, "completed_ects": 60.0},
        "GENERAL":       {"min_ects": 240.0, "min_su": 125.0,
                          "completed_ects": 230.0, "completed_su": 122.0},
        ...
      }
    }

Run:
    python -m scheduler.degree_eval "DEGREE EVALUATION.html"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

from scheduler.transcript import PASSING_GRADES, PROGRAM_NAME_TO_CODE, term_to_code

# Section headers we recognise inside the evaluation page.
KNOWN_SECTIONS = {
    "UNIVERSITY COURSES",
    "CORE ELECTIVES",
    "REQUIRED COURSES",
    "AREA ELECTIVES",
    "FREE ELECTIVES",
    "FACULTY COURSES",
    "ENGINEERING",
    "BASIC SCIENCE",
}

IN_PROGRESS_GRADES = {"I.P.", "IP", "I.P", "REGISTERED"}

CODE_RE = re.compile(r"^([A-Z]{2,5})\s+(\d{2,6}[A-Z]?)$")
TERM_LABEL_RE = re.compile(r"\b(Fall|Spring|Summer)\s+(\d{4})-(\d{4})\b")


def _clean(text: str) -> str:
    """Collapse whitespace / non-breaking spaces in a cell's text."""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _floats(cells: list[str]) -> list[float]:
    out: list[float] = []
    for c in cells:
        c = c.strip()
        if re.fullmatch(r"\d+(?:\.\d+)?", c):
            out.append(float(c))
    return out


# --------------------------------------------------------------------------- #
# Header parsing
# --------------------------------------------------------------------------- #

def _parse_header(soup: BeautifulSoup, full_text: str) -> dict:
    out: dict = {}

    # Student : "00032254 - Çağan Çakır"
    m = re.search(r"(\d{6,})\s*-\s*([^\n<]+)", full_text)
    if m:
        out["student_id"] = m.group(1).strip()
        out["name"] = _clean(m.group(2))

    # Program Requirements Term -> admit term (first term label in the header)
    m = TERM_LABEL_RE.search(full_text)
    if m:
        label = f"{m.group(1)} {m.group(2)}-{m.group(3)}"
        out["admit_term_label"] = label
        try:
            out["admit_term"] = term_to_code(label)
        except ValueError:
            out["admit_term"] = "202101"

    # Program name + minors:
    #   "Computer Science and Engineering   (Minor- Business Analytics, Finance)"
    for prog_name, code in PROGRAM_NAME_TO_CODE.items():
        if re.search(rf"\b{re.escape(prog_name)}\b", full_text):
            out["program_name"] = prog_name
            out["program"] = code
            break

    minors: list[str] = []
    m = re.search(r"\(\s*Minor\s*-\s*([^)]+)\)", full_text, re.I)
    if m:
        minors = [x.strip() for x in m.group(1).split(",") if x.strip()]
    out["minors"] = minors
    out["intended_minor"] = minors[0] if minors else None

    # CGPA: the last numeric in the GENERAL PROGRAM REQUIREMENTS "Completed" row
    # is captured in section parsing; here we just leave a hook.
    return out


# --------------------------------------------------------------------------- #
# Section-table parsing
# --------------------------------------------------------------------------- #

def _parse_section_table(table) -> tuple[str | None, list[dict], dict]:
    """Return (section_name, courses, summary).

    ``courses`` = [{"code": "CS 302", "grade": "I.P."}, ...]
    ``summary`` = {"Minimum Required": [..floats..], "Completed": [..floats..]}
    """
    rows = table.find_all("tr", recursive=False)
    if not rows:
        rows = table.find_all("tr")

    # Section name lives in the first row's cell text.
    header_text = _clean(rows[0].get_text(" ")) if rows else ""
    section = next((s for s in KNOWN_SECTIONS if s in header_text.upper()), None)
    if section is None:
        return None, [], {}

    courses: list[dict] = []
    summary: dict[str, list[float]] = {}
    for tr in rows[1:]:
        cells = [_clean(td.get_text(" ")) for td in tr.find_all("td")]
        if not cells:
            continue
        first = cells[0]
        low = first.lower()
        if low.startswith("minimum required"):
            summary["Minimum Required"] = _floats(cells[1:])
            continue
        if low.startswith("completed"):
            summary["Completed"] = _floats(cells[1:])
            continue
        # Course row: first cell is a course code, second is the grade.
        cm = CODE_RE.match(first)
        if cm and len(cells) >= 2:
            code = f"{cm.group(1)} {cm.group(2)}"
            grade = cells[1].strip()
            courses.append({"code": code, "grade": grade})
    return section, courses, summary


def _classify(grade: str) -> str:
    g = grade.upper().strip()
    if g in IN_PROGRESS_GRADES:
        return "in_progress"
    if grade.strip() in PASSING_GRADES or g in {x.upper() for x in PASSING_GRADES}:
        return "completed"
    return "other"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def parse_degree_evaluation(html_path: str | Path) -> dict:
    html = Path(html_path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    full_text = _clean(soup.get_text("\n"))

    profile = _parse_header(soup, soup.get_text("\n"))

    # Walk every table; collect sections.
    section_courses: dict[str, list[dict]] = {}
    section_requirements: dict[str, dict] = {}
    for table in soup.find_all("table"):
        name, courses, summary = _parse_section_table(table)
        if not name:
            continue
        section_courses.setdefault(name, []).extend(courses)
        # Map summary numbers to labelled fields. ENGINEERING / BASIC SCIENCE
        # tables are 4-column (ECTS, Courses); the others carry SU credits too.
        req: dict = section_requirements.setdefault(name, {})
        mins = summary.get("Minimum Required") or []
        comp = summary.get("Completed") or []
        if name in {"ENGINEERING", "BASIC SCIENCE"}:
            if mins:
                req["min_ects"] = mins[0]
            if comp:
                req["completed_ects"] = comp[0]
        else:
            # [ECTS, SU, Courses] (some cells may be "-" and dropped by _floats)
            if mins:
                req["min_su"] = mins[-2] if len(mins) >= 2 else mins[0]
            if comp:
                req["completed_su"] = comp[-2] if len(comp) >= 2 else comp[0]

    # GENERAL PROGRAM REQUIREMENTS table (min ECTS/SU + completed ECTS/SU/GPA).
    general = _parse_general(soup)
    if general:
        section_requirements["GENERAL"] = general
        if "cgpa" in general:
            profile["cgpa"] = general["cgpa"]
        if "completed_su" in general:
            profile["cumulative_credits"] = general["completed_su"]
        if "completed_ects" in general:
            profile["cumulative_ects"] = general["completed_ects"]

    # Build completed / in-progress sets (a course appears in several sections).
    status_by_code: dict[str, str] = {}
    for courses in section_courses.values():
        for c in courses:
            cls = _classify(c["grade"])
            if cls == "other":
                continue
            prev = status_by_code.get(c["code"])
            # A real letter/S grade ("completed") always wins over "in_progress".
            if prev == "completed":
                continue
            status_by_code[c["code"]] = cls

    completed = sorted(c for c, s in status_by_code.items() if s == "completed")
    in_progress = sorted(c for c, s in status_by_code.items() if s == "in_progress")

    profile["completed"] = completed
    profile["in_progress"] = in_progress
    profile["completed_for_eligibility"] = completed
    profile["section_requirements"] = section_requirements
    profile["source"] = "degree_evaluation"
    profile.setdefault("program", "BSCS-DM")
    profile.setdefault("admit_term", "202101")
    return profile


def _parse_general(soup: BeautifulSoup) -> dict:
    """Parse the top GENERAL PROGRAM REQUIREMENTS table.

    Layout: header row, then a column-label row, then ``Minimum Required`` and
    ``Completed`` rows each carrying [ECTS, SU, PGPA, CGPA].
    """
    for table in soup.find_all("table"):
        text = _clean(table.get_text(" ")).upper()
        if "GENERAL PROGRAM REQUIREMENTS" not in text:
            continue
        out: dict = {}
        for tr in table.find_all("tr"):
            cells = [_clean(td.get_text(" ")) for td in tr.find_all("td")]
            if not cells:
                continue
            low = cells[0].lower()
            nums = _floats(cells[1:])
            if low.startswith("minimum required") and len(nums) >= 2:
                out["min_ects"] = nums[0]
                out["min_su"] = nums[1]
            elif low.startswith("completed") and len(nums) >= 2:
                out["completed_ects"] = nums[0]
                out["completed_su"] = nums[1]
                if len(nums) >= 4:
                    out["cgpa"] = nums[3]
        return out
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("html", help="Path to the Degree Evaluation HTML file")
    ap.add_argument("-o", "--out", help="Where to write the parsed JSON")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    profile = parse_degree_evaluation(args.html)
    if args.out:
        Path(args.out).write_text(
            json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if not args.quiet:
        print(f"Name:        {profile.get('name')}")
        print(f"Program:     {profile.get('program_name')}  ({profile.get('program')})")
        print(f"Admit term:  {profile.get('admit_term_label')}  ({profile.get('admit_term')})")
        print(f"CGPA:        {profile.get('cgpa')}")
        print(f"Minors:      {profile.get('minors')}")
        print(f"Completed   ({len(profile['completed'])}): {profile['completed']}")
        print(f"In progress ({len(profile['in_progress'])}): {profile['in_progress']}")
        print("Section requirements:")
        for name, req in profile["section_requirements"].items():
            print(f"  {name:<32} {req}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
