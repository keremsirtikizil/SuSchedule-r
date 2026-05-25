"""
Scrape Sabancı University Computer Science (BSCS-DM) undergraduate program
into a structured JSON catalog.

Pipeline:
  1. Fetch the degree-requirements page for a target admit term.
     -> list of (course_code, program_section) pairs.
  2. For each unique course, fetch its course-detail page.
     -> title, su_credit, ects, description, prereq_text, coreq_text,
        offered_terms[].
  3. Write everything to data/cs_catalog.json and snapshot raw HTML to
     snapshots/ so we are insulated from upstream changes.

Run:
  python scrape_cs.py            # default admit term = 202601 (Fall 2026-2027)
  python scrape_cs.py 202501     # any 6-digit term code
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://suis.sabanciuniv.edu/prod/"
PROGRAM = "BSCS-DM"
DEFAULT_TERM = "202601"
REQUEST_DELAY_S = 0.4  # be polite

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SNAP_DEG = ROOT / "degrees"
SNAP_CRS = ROOT / "courses"
SNAP_POOL = ROOT / "pools"
for d in (DATA_DIR, SNAP_DEG, SNAP_CRS, SNAP_POOL):
    d.mkdir(parents=True, exist_ok=True)


# Faculty-wide section anchors are program-independent; program-specific ones
# (Required / Core / Area / Free) carry the program code as a prefix.
GLOBAL_SECTION_LABELS = {
    "UC_FENS": "University Courses",
    "UC_FASS": "University Courses",
    "UC_SOM": "University Courses",
    "FC_FENS": "Faculty Courses",
    "FC_FASS": "Faculty Courses",
    "FC_SOM": "Faculty Courses",
    "FC_DSA": "Faculty Courses",
    "ENG_SCIE": "Engineering",
    "BASIC_SCIE": "Basic Science",
}


# SU's degree pages use four naming conventions for the per-program section
# anchors. Mapping every observed code to a canonical label here:
#   R           -> Required
#   C, C1..C4   -> Core Elective   (some programs split Core into groups)
#   A           -> Area Elective
#   F           -> Free Elective
#   M           -> Math Requirement (BAECON-DM)
#   PH          -> Pre-PhD Track (BAPSY-DM)
_SECTION_CODE_TO_LABEL: dict[str, str] = {
    "R": "Required",
    "C": "Core Elective",
    "C1": "Core Elective",
    "C2": "Core Elective",
    "C3": "Core Elective",
    "C4": "Core Elective",
    "A": "Area Elective",
    "F": "Free Elective",
    "M": "Math Requirement",
    "PH": "Pre-PhD Track",
}

# One-off typos / non-canonical prefixes seen on real SU pages.
_EXTRA_PROGRAM_PREFIXES: dict[str, tuple[str, ...]] = {
    "BAVACD-DM": ("BAVACDD",),  # the C1/C2 anchors on this page drop a 'D'
}


def section_labels_for(program: str) -> dict[str, str]:
    """Return the {anchor_name: section_label} map for a given program.

    Generates all observed naming variants:
      - canonical with dash:    BSCS-DM_R, BSCS-DM_C, BSCS-DM_C1
      - no-dash variant:        BAECONDM_R, BAECONDM_C
      - no-dash no-underscore:  BAPOLSDMC1, BAPSIRDMC2
      - any program-specific typo overrides from _EXTRA_PROGRAM_PREFIXES
    """
    labels: dict[str, str] = dict(GLOBAL_SECTION_LABELS)
    prefixes = [program, program.replace("-", ""), *_EXTRA_PROGRAM_PREFIXES.get(program, ())]
    for prefix in prefixes:
        for code, label in _SECTION_CODE_TO_LABEL.items():
            labels[f"{prefix}_{code}"] = label  # with underscore
            labels[f"{prefix}{code}"] = label   # without underscore (BAPOLSDMC1 style)
    return labels


SECTION_LABELS = section_labels_for(PROGRAM)

session = requests.Session()
session.headers.update({"User-Agent": "sucheduler-scraper/0.1 (course catalog research)"})


def fetch(url: str, snapshot_path: Path | None = None) -> str:
    if snapshot_path and snapshot_path.exists():
        return snapshot_path.read_text(encoding="utf-8", errors="replace")
    r = session.get(url, timeout=30)
    r.raise_for_status()
    html = r.text
    if snapshot_path:
        snapshot_path.write_text(html, encoding="utf-8")
    time.sleep(REQUEST_DELAY_S)
    return html


def fetch_degree_page(term: str, program: str = PROGRAM) -> str:
    url = (
        f"{BASE}SU_DEGREE.p_degree_detail"
        f"?P_PROGRAM={program}&P_LANG=EN&P_LEVEL=UG&P_TERM={term}&P_SUBMIT=Select"
    )
    return fetch(url, SNAP_DEG / f"{program}_{term}.html")


COURSE_HREF_RE = re.compile(
    r"sabanci_www\.p_get_courses\?levl_code=UG&subj_code=([A-Z]+)&crse_numb=([A-Z0-9]+)",
    re.IGNORECASE,
)

# Elective pool page link (Core / Area / Free / Faculty)
POOL_HREF_RE = re.compile(
    r"SU_DEGREE\.p_list_courses\?[^\"']*P_AREA=([A-Z0-9_\-]+)[^\"']*",
    re.IGNORECASE,
)


def fetch_pool_page(term: str, area_key: str, query_string: str) -> str:
    """area_key is e.g. BSCS-DM_C; query_string is the raw href as found."""
    url = urljoin(BASE, query_string)
    snap = SNAP_POOL / f"{area_key}_{term}.html"
    return fetch(url, snap)


def parse_pool_page(html: str) -> list[tuple[str, str]]:
    """Return list of (subj, num) found on a pool page."""
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for m in COURSE_HREF_RE.finditer(html):
        key = (m.group(1).upper(), m.group(2).upper())
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def parse_degree_page(
    html: str,
    section_labels: dict[str, str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Parse the degree page.

    Returns (inline_courses, pool_links):
      inline_courses: courses listed directly on the degree page
                      (Required, University, etc.).
      pool_links:     {area_key, section, href} pairs for each elective pool
                      page that must be followed.
    """
    labels = section_labels if section_labels is not None else SECTION_LABELS
    soup = BeautifulSoup(html, "html.parser")
    inline: list[dict] = []
    pools: list[dict] = []
    seen: set[tuple[str, str]] = set()
    seen_pools: set[str] = set()
    current_section_key: str | None = None
    current_section_label: str | None = None

    for el in soup.find_all("a"):
        name = el.get("name")
        if name and name in labels:
            current_section_key = name
            current_section_label = labels[name]
            continue
        href = el.get("href") or ""
        m_course = COURSE_HREF_RE.search(href)
        if m_course and current_section_label:
            subj, num = m_course.group(1).upper(), m_course.group(2).upper()
            code = f"{subj} {num}"
            key = (code, current_section_label)
            if key not in seen:
                seen.add(key)
                inline.append(
                    {
                        "code": code,
                        "subj": subj,
                        "num": num,
                        "section": current_section_label,
                        "section_key": current_section_key,
                    }
                )
            continue
        m_pool = POOL_HREF_RE.search(href)
        if m_pool and current_section_label:
            area_key = m_pool.group(1)
            if area_key in seen_pools:
                continue
            seen_pools.add(area_key)
            pools.append(
                {
                    "area_key": area_key,
                    "section": current_section_label,
                    "section_key": current_section_key,
                    "href": href,
                }
            )
    return inline, pools


def fetch_course_page(subj: str, num: str) -> str:
    url = (
        f"{BASE}sabanci_www.p_get_courses"
        f"?levl_code=UG&subj_code={subj}&crse_numb={num}&lang=eng"
    )
    return fetch(url, SNAP_CRS / f"{subj}_{num}.html")


PREREQ_LABEL = re.compile(r"Prerequisite\s*:", re.IGNORECASE)
COREQ_LABEL = re.compile(r"Corequisite\s*:", re.IGNORECASE)
ECTS_LABEL = re.compile(r"ECTS\s*Credit\s*:", re.IGNORECASE)
GENREQ_LABEL = re.compile(r"General\s*Requirements\s*:", re.IGNORECASE)


def _text(node) -> str:
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()


def parse_course_page(html: str, subj: str, num: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    # Title + SU credit are in the first <th> pair.
    th_cells = soup.find_all("th")
    title = ""
    su_credit_raw = ""
    if len(th_cells) >= 2:
        head = _text(th_cells[0])
        # head looks like: "CS 201 Programming Fundamentals"
        prefix = f"{subj} {num}".upper()
        if head.upper().startswith(prefix):
            title = head[len(prefix):].strip()
        else:
            title = head
        su_credit_raw = _text(th_cells[1])

    su_credit = None
    m = re.search(r"(\d+(?:\.\d+)?)\s*Credits?", su_credit_raw)
    if m:
        try:
            su_credit = float(m.group(1))
        except ValueError:
            pass

    # Walk every <td colspan=2> row in document order so we can accumulate
    # continuation rows under their owning label. Prereqs that span multiple
    # rows for AND/OR clauses live in unlabelled rows following "Prerequisite:".
    label_patterns = (
        ("prereq", PREREQ_LABEL),
        ("coreq", COREQ_LABEL),
        ("ects", ECTS_LABEL),
        ("genreq", GENREQ_LABEL),
    )
    rows: list[tuple[str, str]] = []
    for td in soup.find_all("td"):
        if td.get("colspan") not in ("2", 2):
            continue
        if td.find("table") is not None:
            rows.append(("table", ""))
            continue
        txt = _text(td)
        if not txt:
            rows.append(("empty", ""))
            continue
        matched_kind = None
        matched_text = None
        for kind, pat in label_patterns:
            if pat.search(txt):
                matched_kind = kind
                matched_text = pat.sub("", txt, count=1).strip()
                break
        if matched_kind:
            rows.append((matched_kind, matched_text or ""))
        else:
            rows.append(("cont", txt))

    description = ""
    for kind, txt in rows:
        if kind == "cont":
            description = txt
            break
        if kind in {"empty", "table"}:
            continue
        break  # first content row is already labelled, no description

    fields: dict[str, list[str]] = {"prereq": [], "coreq": [], "ects": [], "genreq": []}
    current: str | None = None
    desc_consumed = False
    for kind, txt in rows:
        if kind in {"empty", "table"}:
            current = None
            continue
        if kind == "cont":
            if not desc_consumed and current is None:
                desc_consumed = True  # description row, already captured
                continue
            if current is not None:
                fields[current].append(txt)
            continue
        desc_consumed = True
        current = kind
        if txt:
            fields[kind].append(txt)

    prereq_text = " ".join(fields["prereq"]).strip()
    coreq_text = " ".join(fields["coreq"]).strip()
    ects_text = " ".join(fields["ects"]).strip()

    # Offered terms: inner table with rows [term, name, su_credit].
    offered: list[dict] = []
    inner_tables = soup.find_all("table")
    for tbl in inner_tables[1:]:  # skip outer wrapper
        header = tbl.find("tr")
        if not header:
            continue
        header_text = _text(header)
        if "Last Offered Terms" not in header_text:
            continue
        for tr in tbl.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if len(tds) < 3:
                continue
            offered.append(
                {
                    "term": _text(tds[0]),
                    "name": _text(tds[1]),
                    "su_credit": _text(tds[2]),
                }
            )
        break

    return {
        "code": f"{subj} {num}",
        "subj": subj,
        "num": num,
        "title": title,
        "su_credit": su_credit,
        "su_credit_raw": su_credit_raw,
        "ects_text": ects_text,
        "description": description,
        "prereq_text": prereq_text,
        "coreq_text": coreq_text,
        "offered_terms": offered,
    }


def scrape(term: str = DEFAULT_TERM) -> dict:
    print(f"[1/4] Fetching CS degree page for admit term {term}...")
    deg_html = fetch_degree_page(term)
    inline_courses, pool_links = parse_degree_page(deg_html)
    print(f"      inline courses: {len(inline_courses)}, elective pools: {len(pool_links)}")

    print(f"[2/4] Fetching {len(pool_links)} elective pool pages...")
    program_courses: list[dict] = list(inline_courses)
    for pl in pool_links:
        try:
            pool_html = fetch_pool_page(term, pl["area_key"], pl["href"])
        except Exception as exc:
            print(f"  ! pool {pl['area_key']}: {exc}")
            continue
        pairs = parse_pool_page(pool_html)
        print(f"      {pl['area_key']:>16} ({pl['section']}): {len(pairs)} courses")
        for subj, num in pairs:
            program_courses.append(
                {
                    "code": f"{subj} {num}",
                    "subj": subj,
                    "num": num,
                    "section": pl["section"],
                    "section_key": pl["area_key"],
                }
            )

    unique_codes: dict[str, dict] = {}
    sections_by_code: dict[str, list[str]] = {}
    for pc in program_courses:
        if pc["section"] not in sections_by_code.setdefault(pc["code"], []):
            sections_by_code[pc["code"]].append(pc["section"])
        unique_codes.setdefault(pc["code"], pc)

    print(f"[3/4] Fetching details for {len(unique_codes)} unique courses...")
    courses: list[dict] = []
    for i, pc in enumerate(unique_codes.values(), start=1):
        try:
            html = fetch_course_page(pc["subj"], pc["num"])
            info = parse_course_page(html, pc["subj"], pc["num"])
        except Exception as exc:
            print(f"  ! {pc['code']}: {exc}")
            info = {
                "code": pc["code"],
                "subj": pc["subj"],
                "num": pc["num"],
                "error": str(exc),
            }
        info["program_sections"] = sections_by_code[pc["code"]]
        courses.append(info)
        if i % 10 == 0:
            print(f"  .. {i}/{len(unique_codes)}")

    catalog = {
        "program": PROGRAM,
        "admit_term": term,
        "course_count": len(courses),
        "courses": courses,
    }
    out_path = DATA_DIR / f"{PROGRAM}_{term}_catalog.json"
    out_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[4/4] Wrote {out_path}")
    return catalog


if __name__ == "__main__":
    term_arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TERM
    scrape(term_arg)
