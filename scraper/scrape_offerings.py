"""
Scrape course offerings (sections actually opened) for a window of recent
terms from the Sabancı University Banner schedule search.

For each term we POST one search per subject (or, optionally, every subject
in a single multi-select POST) to ``bwckschd.p_get_crse_unsec`` and parse the
``<th class="ddlabel">`` rows of the result.

Output:
    data/offerings_<TERM>.json
    data/offerings.json   (merged map: term -> code -> [sections])

Run:
    python -m scraper.scrape_offerings              # last 6 terms, all subjects
    python -m scraper.scrape_offerings --terms 202401 202402

The user explicitly asked us not to go back further than 2 years, so the
default window is the last 6 terms (last two academic years).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SNAP_DIR = ROOT / "snapshots" / "offerings"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SNAP_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://suis.sabanciuniv.edu/prod/"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
)

# Default window: last 6 terms (last two academic years). Anything older was
# explicitly out of scope.
DEFAULT_TERMS = ["202401", "202402", "202403", "202501", "202502", "202503"]

# Every subject ever returned by the dynamic-schedule form.
ALL_SUBJECTS = (
    "ACC AL ANTH ARA BAN BIO CHEM CIP CONF CS CULT DA DS DSA ECON EE ENG ENRG "
    "ENS ENT ES ETM FIN GEN GR HART HIST HUM IE IF IR IT LAW LIT MART MAT MATH "
    "ME MFE MFG MFIN MGMT MIM MKTG MRES NS OPIM ORG PERS PHIL PHYS POLS PROJ "
    "PSIR PSY QL SEC SOC SPS TLL TS TUR VA XM"
).split()


# A section row looks like:
#   <a ...>Programming Fundamentals - 10190 - CS 201 - A</a>
# Capture: title, crn, subj+num, section letter/code.
SECTION_RE = re.compile(
    r'class="ddlabel"[^>]*>'                                   # row marker
    r'\s*<a[^>]*href="/prod/bwckschd\.p_disp_detail_sched\?'
    r'term_in=\d+&(?:amp;)?crn_in=(?P<crn>\d+)"[^>]*>'
    r'(?P<title>[^<]*?)\s+-\s+(?P=crn)\s+-\s+'
    r'(?P<subj>[A-Z]{2,5})\s+(?P<num>\d{2,5}[A-Z]?)\s+-\s+'
    r'(?P<section>[^<]+?)</a>',
    re.IGNORECASE | re.DOTALL,
)

# After each section header comes a "Scheduled Meeting Times" table. We look
# for rows of seven <td CLASS="dddefault"> cells: Type | Time | Days | Where |
# Date Range | Schedule Type | Instructors.
MEETING_TABLE_RE = re.compile(
    r'<caption class="captiontext">Scheduled Meeting Times</caption>'
    r'(?P<body>.*?)</table>',
    re.IGNORECASE | re.DOTALL,
)
MEETING_ROW_RE = re.compile(
    r'<tr>\s*'
    r'<td CLASS="dddefault">(?P<type>[^<]*)</td>\s*'
    r'<td CLASS="dddefault">(?P<time>[^<]*)</td>\s*'
    r'<td CLASS="dddefault">(?P<days>[^<]*)</td>\s*'
    r'<td CLASS="dddefault">(?P<where>[^<]*)</td>\s*'
    r'<td CLASS="dddefault">(?P<date_range>[^<]*)</td>\s*'
    r'<td CLASS="dddefault">(?P<sched_type>[^<]*)</td>\s*'
    r'<td CLASS="dddefault">(?P<instructors>.*?)</td>\s*'
    r'</tr>',
    re.IGNORECASE | re.DOTALL,
)
TIME_RANGE_RE = re.compile(
    r'(?P<sh>\d{1,2}):(?P<sm>\d{2})\s*(?P<sap>am|pm)\s*-\s*'
    r'(?P<eh>\d{1,2}):(?P<em>\d{2})\s*(?P<eap>am|pm)',
    re.IGNORECASE,
)
DAY_INDEX = {"M": 0, "T": 1, "W": 2, "R": 3, "F": 4, "S": 5, "U": 6}


def _to_minutes(h: int, m: int, ap: str) -> int:
    """Convert a 12-hour clock value to minutes since midnight."""
    ap = ap.lower()
    if ap == "pm" and h != 12:
        h += 12
    if ap == "am" and h == 12:
        h = 0
    return h * 60 + m


def parse_meeting(time_s: str, days_s: str, where_s: str,
                  instr_s: str, type_s: str) -> dict | None:
    """Return a normalised meeting dict, or None if the row has no real time."""
    days_s = days_s.strip()
    time_s = time_s.strip()
    if not days_s or not time_s or time_s.upper() == "TBA":
        return None
    m = TIME_RANGE_RE.search(time_s)
    if not m:
        return None
    start = _to_minutes(int(m["sh"]), int(m["sm"]), m["sap"])
    end = _to_minutes(int(m["eh"]), int(m["em"]), m["eap"])
    day_codes = [d for d in days_s if d in DAY_INDEX]
    days = [DAY_INDEX[d] for d in day_codes]
    # Strip the (P) / (S) primary-secondary markers from instructors
    instr_clean = re.sub(r"<[^>]+>", "", instr_s).strip()
    instr_clean = re.sub(r"\s*\([PS]\)\s*", "", instr_clean)
    instr_clean = re.sub(r"\s+", " ", instr_clean).strip(" ,")
    return {
        "type": type_s.strip() or None,
        "days_raw": days_s,
        "days": days,
        "start": start,                # minutes since midnight
        "end": end,
        "where": where_s.strip(),
        "instructor": instr_clean,
    }


def new_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def pick_term(session: requests.Session, term: str) -> None:
    """Hit the entry page + select-term page so the session has the right state."""
    session.get(BASE + "bwckschd.p_disp_dyn_sched")
    session.post(
        BASE + "bwckgens.p_proc_term_date",
        data={"p_calling_proc": "bwckschd.p_disp_dyn_sched", "p_term": term},
        headers={"Referer": BASE + "bwckschd.p_disp_dyn_sched"},
    )


def fetch_offerings(session: requests.Session, term: str, subjects: list[str]) -> str:
    """POST the schedule-search and return the HTML."""
    data: list[tuple[str, str]] = [
        ("term_in", term),
        # Banner expects the dummy sentinel BEFORE the real values.
        ("sel_subj", "dummy"),
        ("sel_day", "dummy"),
        ("sel_schd", "dummy"),
        ("sel_insm", "dummy"),
        ("sel_camp", "dummy"),
        ("sel_levl", "dummy"),
        ("sel_sess", "dummy"),
        ("sel_instr", "dummy"),
        ("sel_ptrm", "dummy"),
        ("sel_attr", "dummy"),
    ]
    for subj in subjects:
        data.append(("sel_subj", subj))
    data.extend([
        ("sel_crse", ""),
        ("sel_title", ""),
        ("sel_schd", "%"),
        ("sel_from_cred", ""),
        ("sel_to_cred", ""),
        ("sel_camp", "%"),
        ("sel_levl", "%"),
        ("sel_ptrm", "%"),
        ("sel_instr", "%"),
        ("sel_attr", "%"),
        ("begin_hh", "0"),
        ("begin_mi", "0"),
        ("begin_ap", "a"),
        ("end_hh", "0"),
        ("end_mi", "0"),
        ("end_ap", "a"),
    ])
    r = session.post(
        BASE + "bwckschd.p_get_crse_unsec",
        data=data,
        headers={"Referer": BASE + "bwckgens.p_proc_term_date"},
        timeout=120,
    )
    r.raise_for_status()
    return r.text


def parse_offerings(html: str) -> dict[str, list[dict]]:
    """Return {course_code: [section_dict, ...]} from one page.

    Each section dict carries: crn, section, title, meetings[] (with day
    indices + start/end minutes), instructors[], locations[].
    """
    out: dict[str, list[dict]] = defaultdict(list)
    section_matches = list(SECTION_RE.finditer(html))
    for i, m in enumerate(section_matches):
        section_end = section_matches[i + 1].start() if i + 1 < len(section_matches) else len(html)
        block = html[m.end():section_end]

        meetings: list[dict] = []
        instructors: set[str] = set()
        locations: set[str] = set()
        m_table = MEETING_TABLE_RE.search(block)
        if m_table:
            for row in MEETING_ROW_RE.finditer(m_table.group("body")):
                meeting = parse_meeting(
                    row.group("time"),
                    row.group("days"),
                    row.group("where"),
                    row.group("instructors"),
                    row.group("type"),
                )
                if meeting is None:
                    continue
                meetings.append(meeting)
                if meeting.get("instructor"):
                    instructors.add(meeting["instructor"])
                if meeting.get("where"):
                    locations.add(meeting["where"])

        code = f"{m.group('subj').upper()} {m.group('num').upper()}"
        out[code].append(
            {
                "crn": m.group("crn"),
                "section": m.group("section").strip(),
                "title": m.group("title").strip(),
                "meetings": meetings,
                "instructors": sorted(instructors),
                "locations": sorted(locations),
            }
        )
    return dict(out)


def scrape_term(term: str, subjects: list[str], snapshot: bool = True) -> dict[str, list[dict]]:
    """Scrape every subject for one term in a single POST."""
    session = new_session()
    pick_term(session, term)
    print(f"  term {term}: searching {len(subjects)} subjects...")
    html = fetch_offerings(session, term, subjects)
    if snapshot:
        snap_path = SNAP_DIR / f"offerings_{term}.html"
        snap_path.write_text(html, encoding="utf-8")
    parsed = parse_offerings(html)
    print(f"  term {term}: {sum(len(v) for v in parsed.values())} sections, "
          f"{len(parsed)} distinct courses")
    return parsed


def merge_offerings(per_term: dict[str, dict[str, list[dict]]]) -> dict:
    """Reshape per-term results into the merged on-disk format."""
    return {
        term: {
            code: sections
            for code, sections in sorted(per_term[term].items())
        }
        for term in sorted(per_term)
    }


def scrape(terms: list[str], subjects: list[str]) -> dict:
    per_term: dict[str, dict[str, list[dict]]] = {}
    for term in terms:
        try:
            per_term[term] = scrape_term(term, subjects)
        except Exception as e:
            print(f"  ! {term} failed: {e}")
            per_term[term] = {}
        time.sleep(0.4)
        out_term = DATA_DIR / f"offerings_{term}.json"
        out_term.write_text(
            json.dumps(per_term[term], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    merged = merge_offerings(per_term)
    out_merged = DATA_DIR / "offerings.json"
    out_merged.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out_merged}")
    return merged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms", nargs="+", default=DEFAULT_TERMS,
                    help="6-digit term codes (default: last 6 terms)")
    ap.add_argument("--subjects", nargs="+", default=ALL_SUBJECTS,
                    help="Subject prefixes to include (default: all)")
    args = ap.parse_args()
    scrape(args.terms, args.subjects)
    return 0


if __name__ == "__main__":
    sys.exit(main())
