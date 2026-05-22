"""
Parse a Sabancı University "Academic Records Summary" PDF transcript
into a structured Student profile.

Output schema (also accepted by ``scheduler.eligibility.Student.from_dict``):

    {
      "name": str,
      "student_id": str,
      "level": "Undergraduate",
      "program": "BSCS-DM",                  # mapped from degree name
      "program_name": "Computer Science and Engineering",
      "admit_term": "202201",                # 6-digit SUIS-style code
      "admit_term_label": "Fall 2022-2023",
      "current_semester": int,
      "cgpa": float,
      "cumulative_credits": float,           # earned SU credits
      "cumulative_ects": float,
      "minors": [str, ...],
      "intended_minor": str | None,          # picked = first minor by default
      "advisor": str,
      "completed": [str, ...],               # course codes with passing/S grade
      "in_progress": [str, ...],             # "Registered"
      "withdrawn": [str, ...],
      "excluded": [str, ...],
      "transfer": [str, ...],
      "terms": [
        {
          "term": "Fall 2023-2024",
          "term_code": "202301",
          "level": "Undergraduate",
          "program": "...",
          "is_transfer": bool,
          "transferred_from": str | None,
          "courses": [
            {"code": "CS 201", "title": "Programming Fundamentals",
             "level": "UG", "grade": "A", "credit": 3.0, "ects": 6.0,
             "status": "completed"},
            ...
          ]
        }, ...
      ]
    }

Run:
    python -m scheduler.transcript /path/to/Academic\\ Records\\ Summary.pdf
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pdfplumber

# --------------------------------------------------------------------------- #
# Lookup tables
# --------------------------------------------------------------------------- #

# Map the degree program name found in the transcript header to the SUIS
# program code used by our catalog scraper. Extend as we add programs.
# Known faculty / school prefixes that appear before a minor's name on the
# header line. We strip these before extracting the minor itself.
SCHOOL_PREFIXES: tuple[str, ...] = (
    "Sabancı Business School",
    "Sabanci Business School",
    "Faculty of Engineering and Natural Sciences",
    "Faculty of Arts and Social Sciences",
    "School of Management",
    "School of Languages",
)

PROGRAM_NAME_TO_CODE: dict[str, str] = {
    "Computer Science and Engineering": "BSCS-DM",
    "Data Science and Analytics": "BSDSA-DM",
    "Economics": "BAECON-DM",
    "Electronics Engineering": "BSEE-DM",
    "Industrial Engineering": "BSIE-DM",
    "International Studies": "BAIS-DM",
    "Management": "BAMAN-DM",
    "Materials Science and Nano Engineering": "BSMAT-DM",
    "Mechatronics Engineering": "BSME-DM",
    "Molecular Biology, Genetics and Bioengineering": "BSBIO-DM",
    "Political Science": "BAPOLS-DM",
    "Political Science and International Relations": "BAPSIR-DM",
    "Psychology": "BAPSY-DM",
    "Visual Arts and Visual Communications Design": "BAVACD-DM",
}

PASSING_GRADES: set[str] = {
    "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D",
    "S", "EL", "T",
}
FAILING_GRADES: set[str] = {"F", "U", "UL", "NA"}
IN_PROGRESS_TOKEN = "Registered"
WITHDRAWN_GRADES: set[str] = {"W"}
INCOMPLETE_GRADES: set[str] = {"I", "P"}

# All recognised level tokens that sit between the title and the grade.
LEVEL_TOKENS: set[str] = {"UG", "FDY", "MA", "DR", "SP"}

# --------------------------------------------------------------------------- #
# Regex helpers
# --------------------------------------------------------------------------- #

CODE_RE = re.compile(r"^([A-Z]{2,5})\s+(\d{2,5}[A-Z]?)\b")
# Match a course-row's tail: ... LEVEL GRADE CREDIT.xx ECTS.xx [STATUS]
ROW_TAIL_RE = re.compile(
    r"\s+(?P<level>UG|FDY|MA|DR|SP)\s+"
    r"(?P<grade>[A-Z][+\-]?|Registered|NA|EL|UL|SL|S|F|P|I|T|W|U)\s+"
    r"(?P<credit>\d+\.\d{2})\s+"
    r"(?P<ects>\d+\.\d{2})"
    r"(?:\s+(?P<status>\w[\w\- ]*))?\s*$"
)
TERM_LABEL_RE = re.compile(r"^(Fall|Spring|Summer)\s+(\d{4})-(\d{4})\s*$")
STATUS_LINE_RE = re.compile(r"^Status\s*:\s*(?P<status>.+?)\s*/\s*Level\s*:\s*(?P<level>.+?)\s*$")
PROGRAM_LINE_RE = re.compile(r"^Program\s*:\s*(?P<program>.+?)\s*$")
TRANSFER_FROM_RE = re.compile(r"^Transferred from:\s*(?P<src>.+?)\s*$")
TOTALS_RE = re.compile(
    r"^(?P<su>\d+\.\d{2})\s+(?P<ects>\d+\.\d{2})\s+(?P<cgpa>\d+\.\d{2})\s*$"
)


def term_to_code(label: str) -> str:
    """Convert ``Fall 2022-2023`` -> ``202201`` (matches SUIS term codes)."""
    m = TERM_LABEL_RE.match(label.strip())
    if not m:
        raise ValueError(f"Unrecognised term label: {label!r}")
    season, y1, _ = m.group(1), m.group(2), m.group(3)
    suffix = {"Fall": "01", "Spring": "02", "Summer": "03"}[season]
    return f"{y1}{suffix}"


def classify_grade(grade: str) -> str:
    if grade == IN_PROGRESS_TOKEN:
        return "in_progress"
    if grade in PASSING_GRADES:
        return "completed"
    if grade in FAILING_GRADES:
        return "failed"
    if grade in WITHDRAWN_GRADES:
        return "withdrawn"
    if grade in INCOMPLETE_GRADES:
        return "incomplete"
    return "unknown"


# --------------------------------------------------------------------------- #
# Header (page 1) parsing
# --------------------------------------------------------------------------- #


def _parse_header(text: str) -> dict:
    """Extract the name, student id, admit term, program, minors, CGPA, totals."""
    out: dict = {}
    # Name + semester:   "Name Çağan Çakır Semester 7"
    m = re.search(r"Name\s+(.+?)\s+Semester\s+(\d+)", text)
    if m:
        out["name"] = m.group(1).strip()
        out["current_semester"] = int(m.group(2))
    # Student number:    "Student 00032254 First Admit Fall 2022-2023 / Standard - OSYM"
    m = re.search(r"Student\s+(\d+)\s+First Admit\s+(Fall|Spring|Summer)\s+(\d{4}-\d{4})", text)
    if m:
        out["student_id"] = m.group(1)
        admit_label = f"{m.group(2)} {m.group(3)}"
        out["admit_term_label"] = admit_label
        out["admit_term"] = term_to_code(admit_label)
    # Level
    m = re.search(r"Level\s+(Undergraduate|Graduate|Foundation Development Year)", text)
    if m:
        out["level"] = m.group(1)
    # Totals + CGPA: a single line with three decimals.
    for line in text.splitlines():
        m = TOTALS_RE.match(line.strip())
        if m:
            out["cumulative_credits"] = float(m.group("su"))
            out["cumulative_ects"] = float(m.group("ects"))
            out["cgpa"] = float(m.group("cgpa"))
            break
    # Faculty / Program / Degree rows. There may be a primary program + minors.
    programs: list[str] = []
    minors: list[str] = []
    for line in text.splitlines():
        if "(Minor)" in line:
            # "Sabancı Business School Business Analytics (Minor)"
            stripped = line.strip()
            for prefix in SCHOOL_PREFIXES:
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix):].strip()
                    break
            m = re.match(r"(.+?)\s+\(Minor\)\s*$", stripped)
            if m:
                minors.append(m.group(1).strip())
        else:
            for prog_name in PROGRAM_NAME_TO_CODE:
                # We require word boundaries to avoid matching "Engineering"
                # in "Faculty of Engineering and Natural Sciences".
                if re.search(rf"\b{re.escape(prog_name)}\b", line):
                    programs.append(prog_name)
                    break
    if programs:
        out["program_name"] = programs[0]
        out["program"] = PROGRAM_NAME_TO_CODE[programs[0]]
    out["minors"] = minors
    out["intended_minor"] = minors[0] if minors else None
    # Advisor
    m = re.search(r"academic advisor,\s*(.+?),", text)
    if m:
        out["advisor"] = m.group(1).strip()
    return out


# --------------------------------------------------------------------------- #
# Term-block parsing
# --------------------------------------------------------------------------- #


def _iter_lines(pdf_text_pages: list[str]):
    """Yield (page_index, raw_line) for every non-empty line, skipping
    PDF browser-print headers/footers."""
    skip_substrings = (
        "Academic Records Summary",
        "apps.sabanciuniv.edu/acad/transcript",
        "1/8", "2/8", "3/8", "4/8", "5/8", "6/8", "7/8", "8/8",
    )
    for idx, page in enumerate(pdf_text_pages):
        for raw in page.splitlines():
            line = raw.strip()
            if not line:
                continue
            if any(s in line for s in skip_substrings):
                continue
            yield idx, line


def _parse_courses(pdf_text_pages: list[str]) -> tuple[list[dict], dict]:
    """Return (term_blocks, bucket_index).

    bucket_index = {"completed": [...], "in_progress": [...], ...}
    """
    terms: list[dict] = []
    current_term: dict | None = None
    pending_status: dict | None = None
    pending_transfer_from: str | None = None
    pending_is_transfer = False
    last_course: dict | None = None

    buckets: dict[str, list[str]] = {
        "completed": [],
        "in_progress": [],
        "failed": [],
        "withdrawn": [],
        "incomplete": [],
        "excluded": [],
        "transfer": [],
    }

    def close_term():
        nonlocal current_term
        if current_term is not None:
            terms.append(current_term)
        current_term = None

    for _, line in _iter_lines(pdf_text_pages):
        # Status line opens a new term block (Status + Level are on the same line).
        m_status = STATUS_LINE_RE.match(line)
        if m_status:
            pending_status = {
                "status": m_status.group("status").strip(),
                "level": m_status.group("level").strip(),
            }
            last_course = None
            continue
        # Term label
        m_term = TERM_LABEL_RE.match(line)
        if m_term:
            term_label = line.strip()
            term_code = term_to_code(term_label)
            # If a PDF page break split a term in two, the same term label
            # appears twice with a Status line in between. Reattach instead of
            # opening a fresh term block.
            same_as_open = (
                current_term is not None
                and current_term["term_code"] == term_code
            )
            same_as_last_closed = (
                terms and current_term is None and terms[-1]["term_code"] == term_code
            )
            if same_as_open:
                pending_status = None
                last_course = None
                continue
            if same_as_last_closed:
                current_term = terms.pop()
                pending_status = None
                last_course = None
                continue
            close_term()
            current_term = {
                "term": term_label,
                "term_code": term_code,
                "level": (pending_status or {}).get("level"),
                "status": (pending_status or {}).get("status"),
                "program": None,
                "is_transfer": False,
                "transferred_from": None,
                "courses": [],
            }
            pending_status = None
            last_course = None
            continue
        # Program inside a term block
        m_prog = PROGRAM_LINE_RE.match(line)
        if m_prog and current_term is not None:
            current_term["program"] = m_prog.group("program").strip()
            continue
        # Transfer-courses marker
        if line.startswith("Transfer Courses") and current_term is not None:
            current_term["is_transfer"] = True
            pending_is_transfer = True
            continue
        m_tf = TRANSFER_FROM_RE.match(line)
        if m_tf and current_term is not None:
            current_term["transferred_from"] = m_tf.group("src").strip()
            pending_transfer_from = current_term["transferred_from"]
            continue
        # Standing / Dean's List / Term GPA lines (skip)
        if line.startswith(("Standing:", "Dean's List:", "GPA", "Term", "Cumulative")):
            last_course = None
            continue
        # Column header (skip)
        if line.startswith("COURSE CODE"):
            last_course = None
            continue

        # Course row?
        m_code = CODE_RE.match(line)
        if m_code and current_term is not None:
            m_tail = ROW_TAIL_RE.search(line)
            if not m_tail:
                # Looks like a code but no trailing grade/credit -- treat as continuation
                last_course = None
                continue
            subj = m_code.group(1)
            num = m_code.group(2)
            code = f"{subj} {num}"
            # Title sits between the code and the level token at m_tail.start
            title = line[m_code.end():m_tail.start()].strip()
            grade = m_tail.group("grade")
            credit = float(m_tail.group("credit"))
            ects = float(m_tail.group("ects"))
            status_token = m_tail.group("status") or ""
            status_token = status_token.strip()

            if status_token == "Excluded":
                bucket = "excluded"
            elif current_term["is_transfer"]:
                bucket = "transfer"
            else:
                bucket = classify_grade(grade)

            row = {
                "code": code,
                "title": title,
                "level": m_tail.group("level"),
                "grade": grade,
                "credit": credit,
                "ects": ects,
                "raw_status": status_token,
                "status": bucket,
            }
            current_term["courses"].append(row)
            last_course = row
            buckets.setdefault(bucket, []).append(code)
            continue

        # Otherwise treat as a wrapped continuation of the previous course's title.
        if last_course is not None and not line[0].isupper() or (
            last_course is not None and not CODE_RE.match(line)
        ):
            # Avoid hijacking known structural lines (already handled above).
            last_course["title"] = (last_course["title"] + " " + line).strip()
            continue

    close_term()

    # Dedup buckets while preserving order
    for k, v in list(buckets.items()):
        seen: set[str] = set()
        deduped: list[str] = []
        for c in v:
            if c in seen:
                continue
            seen.add(c)
            deduped.append(c)
        buckets[k] = deduped

    return terms, buckets


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def parse_transcript(pdf_path: str | Path) -> dict:
    pdf_path = Path(pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    full_text = "\n".join(pages)
    header = _parse_header(full_text)
    terms, buckets = _parse_courses(pages)

    profile: dict = {**header}
    profile["terms"] = terms
    profile["completed"] = buckets["completed"]
    profile["in_progress"] = buckets["in_progress"]
    profile["failed"] = buckets["failed"]
    profile["withdrawn"] = buckets["withdrawn"]
    profile["excluded"] = buckets["excluded"]
    profile["transfer"] = buckets["transfer"]

    # Convenience field for the eligibility module: completed + transfer.
    # Transfer rows include both SU equivalents (e.g. CS 307) and the host
    # university's codes (e.g. CORE 100). Foreign codes are inert because no
    # SU course lists them as a prereq, so we just merge everything.
    seen: set[str] = set(buckets["completed"])
    eligibility_completed = list(buckets["completed"])
    for code in buckets["transfer"]:
        if code in seen:
            continue
        seen.add(code)
        eligibility_completed.append(code)
    profile["completed_for_eligibility"] = eligibility_completed
    return profile


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="Path to Academic Records Summary PDF")
    ap.add_argument("-o", "--out", help="Where to write the parsed JSON")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    profile = parse_transcript(args.pdf)

    if args.out:
        Path(args.out).write_text(
            json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if not args.quiet:
        # Print a compact human-readable summary instead of the whole JSON.
        print(f"Name:          {profile.get('name')}")
        print(f"Student #:     {profile.get('student_id')}")
        print(f"Program:       {profile.get('program_name')}  ({profile.get('program')})")
        print(f"Admit term:    {profile.get('admit_term_label')}  ({profile.get('admit_term')})")
        print(f"Semester:      {profile.get('current_semester')}")
        print(f"CGPA:          {profile.get('cgpa')}")
        print(f"SU credits:    {profile.get('cumulative_credits')}")
        print(f"ECTS credits:  {profile.get('cumulative_ects')}")
        print(f"Minors:        {profile.get('minors')}")
        print(f"Advisor:       {profile.get('advisor')}")
        print()
        print(f"Completed   ({len(profile['completed'])}): {profile['completed']}")
        print(f"In progress ({len(profile['in_progress'])}): {profile['in_progress']}")
        print(f"Withdrawn   ({len(profile['withdrawn'])}): {profile['withdrawn']}")
        print(f"Excluded    ({len(profile['excluded'])}): {profile['excluded']}")
        print(f"Transfer    ({len(profile['transfer'])}): {profile['transfer']}")
        print()
        print("Terms:")
        for t in profile["terms"]:
            print(f"  {t['term']} ({t['term_code']}) [{t['level']}] -- {len(t['courses'])} courses"
                  f"{'  (transfer)' if t['is_transfer'] else ''}")
            for c in t["courses"]:
                print(f"    {c['code']:<10} {c['grade']:<10} {c['credit']:>5}  {c['status']:<11}  {c['title']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
