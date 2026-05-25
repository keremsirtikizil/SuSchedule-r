"""
Stage 1 of the full SU catalog scrape: course discovery.

Two passes produce a single deduped set of (SUBJ, NUM) pairs to fetch in
Stage 2:

  1a. Degree-driven: for each (DEGREE_CODE, TERM) pair, fetch the degree
      page and any linked elective pool pages. Collect every course code
      found, tagged with the (program, term, section) it appeared under.

  1b. Subject brute-force: collect every distinct subject code seen in the
      degree/pool HTML, then for each subject try crse_numb in
      {100..499} via the course-detail endpoint. Drop subjects that yield
      nothing in their first 50 attempts; early-stop a subject after 20
      consecutive misses.

Run:
  python -m scraper.discover --degrees                # Stage 1a only
  python -m scraper.discover --degrees --subjects     # both passes
  python -m scraper.discover --all                    # alias for above

Output:
  data/discovery_manifest.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from scraper.scrape_cs import (
    BASE,
    COURSE_HREF_RE,
    DATA_DIR,
    REQUEST_DELAY_S,
    SNAP_CRS,
    SNAP_DEG,
    SNAP_POOL,
    fetch_degree_page,
    fetch_pool_page,
    parse_degree_page,
    parse_pool_page,
    section_labels_for,
    session,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Degree codes derived from scheduler/transcript.py PROGRAM_NAME_TO_CODE.
# Bad codes 404 cleanly during fetch; the loop logs and continues.
DEGREE_CODES: tuple[str, ...] = (
    "BSCS-DM",   # Computer Science and Engineering
    "BSDSA-DM",  # Data Science and Analytics
    "BAECON-DM", # Economics
    "BSEE-DM",   # Electronics Engineering
    "BSIE-DM",   # Industrial Engineering
    "BAIS-DM",   # International Studies
    "BAMAN-DM",  # Management
    "BSMAT-DM",  # Materials Science and Nano Engineering
    "BSME-DM",   # Mechatronics Engineering
    "BSBIO-DM",  # Molecular Biology, Genetics and Bioengineering
    "BAPOLS-DM", # Political Science
    "BAPSIR-DM", # Political Science and International Relations
    "BAPSY-DM",  # Psychology
    "BAVACD-DM", # Visual Arts and Visual Communications Design
)

TERMS: tuple[str, ...] = (
    "202501",  # Fall 25-26
    "202502",  # Spring 25-26
    "202601",  # Fall 26-27
)

BRUTE_NUM_RANGE = range(100, 500)
BRUTE_PROBE_THRESHOLD = 50   # if zero hits in first N probes, skip subject
BRUTE_MISS_STREAK_STOP = 20  # stop a subject after N consecutive misses


# ---------------------------------------------------------------------------
# Stage 1a: degree-driven discovery
# ---------------------------------------------------------------------------


def discover_from_degrees(
    degrees: tuple[str, ...] = DEGREE_CODES,
    terms: tuple[str, ...] = TERMS,
    verbose: bool = True,
) -> tuple[dict, set[str], list[dict]]:
    """Run Stage 1a.

    Returns:
      courses:  {"SUBJ NUM": {"subj","num","program_sections":[...]}, ...}
      subjects: set of subject codes seen anywhere in scraped HTML
      missing_degrees: [{"program","term","error"}] for degrees that 404'd
    """
    courses: dict[str, dict] = {}
    subjects: set[str] = set()
    missing: list[dict] = []

    def add_course(subj: str, num: str, program: str, term: str, section: str) -> None:
        code = f"{subj} {num}"
        entry = courses.setdefault(
            code,
            {"subj": subj, "num": num, "program_sections": []},
        )
        tag = {"program": program, "term": term, "section": section}
        if tag not in entry["program_sections"]:
            entry["program_sections"].append(tag)

    for program in degrees:
        labels = section_labels_for(program)
        for term in terms:
            if verbose:
                print(f"[1a] {program} / {term}")
            try:
                deg_html = fetch_degree_page(term, program=program)
            except requests.HTTPError as exc:
                if verbose:
                    print(f"     ! degree page error: {exc}")
                missing.append({"program": program, "term": term, "error": str(exc)})
                continue
            except Exception as exc:  # network etc.
                if verbose:
                    print(f"     ! degree page error: {exc}")
                missing.append({"program": program, "term": term, "error": str(exc)})
                continue

            # Track all subject codes that show up in the raw HTML.
            for m in COURSE_HREF_RE.finditer(deg_html):
                subjects.add(m.group(1).upper())

            inline, pool_links = parse_degree_page(deg_html, section_labels=labels)
            for c in inline:
                add_course(c["subj"], c["num"], program, term, c["section"])

            for pl in pool_links:
                try:
                    pool_html = fetch_pool_page(term, pl["area_key"], pl["href"])
                except Exception as exc:
                    if verbose:
                        print(f"     ! pool {pl['area_key']}: {exc}")
                    continue
                for m in COURSE_HREF_RE.finditer(pool_html):
                    subjects.add(m.group(1).upper())
                for subj, num in parse_pool_page(pool_html):
                    add_course(subj, num, program, term, pl["section"])

            if verbose:
                print(f"     courses so far: {len(courses)}  subjects: {len(subjects)}")

    return courses, subjects, missing


# ---------------------------------------------------------------------------
# Stage 1b: subject brute-force
# ---------------------------------------------------------------------------


_TITLE_RE = re.compile(r"<th[^>]*>\s*([A-Z]{2,5})\s+(\d{2,5}[A-Z]?)\s", re.IGNORECASE)


def _course_page_exists(html: str, subj: str, num: str) -> bool:
    """Heuristic: a real course page has '<subj> <num>' inside a <th>."""
    if not html or len(html) < 200:
        return False
    if "Course not found" in html or "no course" in html.lower():
        return False
    m = _TITLE_RE.search(html)
    if not m:
        return False
    return m.group(1).upper() == subj.upper() and m.group(2).upper() == num.upper()


def _fetch_course_html(subj: str, num: str) -> str | None:
    """Return HTML for the course detail page, using the local cache when present."""
    snap = SNAP_CRS / f"{subj}_{num}.html"
    if snap.exists():
        return snap.read_text(encoding="utf-8", errors="replace")
    url = (
        f"{BASE}sabanci_www.p_get_courses"
        f"?levl_code=UG&subj_code={subj}&crse_numb={num}&lang=eng"
    )
    try:
        r = session.get(url, timeout=30)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    html = r.text
    time.sleep(REQUEST_DELAY_S)
    if _course_page_exists(html, subj, num):
        snap.write_text(html, encoding="utf-8")
        return html
    # Don't cache miss responses on disk; they'd hide future fixes.
    return html


def brute_force_subjects(
    subjects: set[str],
    courses: dict,
    verbose: bool = True,
) -> dict:
    """Run Stage 1b. Mutates `courses` in-place and returns it."""
    for subj in sorted(subjects):
        if verbose:
            print(f"[1b] subject {subj}")
        hits_this_subject = 0
        miss_streak = 0
        for i, num_int in enumerate(BRUTE_NUM_RANGE):
            num = str(num_int)
            code = f"{subj} {num}"
            if code in courses:
                # Already known from Stage 1a — verify cached HTML existence.
                hits_this_subject += 1
                miss_streak = 0
                continue
            html = _fetch_course_html(subj, num)
            if html and _course_page_exists(html, subj, num):
                hits_this_subject += 1
                miss_streak = 0
                courses.setdefault(
                    code,
                    {"subj": subj, "num": num, "program_sections": []},
                )
            else:
                miss_streak += 1
            if (
                i + 1 >= BRUTE_PROBE_THRESHOLD
                and hits_this_subject == 0
            ):
                if verbose:
                    print(f"     {subj}: no hits in {BRUTE_PROBE_THRESHOLD} probes, skipping")
                break
            if miss_streak >= BRUTE_MISS_STREAK_STOP and hits_this_subject > 0:
                if verbose:
                    print(f"     {subj}: {miss_streak} consecutive misses, early stop")
                break
        if verbose:
            print(f"     {subj}: {hits_this_subject} courses confirmed")
    return courses


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def write_manifest(
    courses: dict,
    subjects: set[str],
    missing: list[dict],
    out_path: Path,
) -> None:
    payload = {
        "discovered_at": datetime.now(timezone.utc).isoformat(),
        "degrees": list(DEGREE_CODES),
        "terms": list(TERMS),
        "subjects_discovered": sorted(subjects),
        "missing_degrees": missing,
        "course_count": len(courses),
        "courses": dict(sorted(courses.items())),
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {out_path}  ({len(courses)} courses, {len(subjects)} subjects)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--degrees", action="store_true", help="run Stage 1a")
    ap.add_argument("--subjects", action="store_true", help="run Stage 1b")
    ap.add_argument("--all", action="store_true", help="shortcut for --degrees --subjects")
    ap.add_argument(
        "--out",
        default=str(DATA_DIR / "discovery_manifest.json"),
        help="output manifest path",
    )
    args = ap.parse_args()

    if args.all:
        args.degrees = True
        args.subjects = True
    if not (args.degrees or args.subjects):
        ap.error("specify at least one of --degrees, --subjects, --all")

    courses: dict = {}
    subjects: set[str] = set()
    missing: list[dict] = []

    if args.degrees:
        courses, subjects, missing = discover_from_degrees()

    if args.subjects:
        # If --subjects alone, prime the subject set from cached degree HTML.
        if not args.degrees and not subjects:
            for cache_dir in (SNAP_POOL, SNAP_DEG):
                for html_path in cache_dir.glob("*.html"):
                    html = html_path.read_text(encoding="utf-8", errors="replace")
                    for m in COURSE_HREF_RE.finditer(html):
                        subjects.add(m.group(1).upper())
        brute_force_subjects(subjects, courses)

    write_manifest(courses, subjects, missing, Path(args.out))


if __name__ == "__main__":
    main()
