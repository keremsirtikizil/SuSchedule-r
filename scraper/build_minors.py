"""
Scrape SU undergraduate MINOR programs into data/minors.json.

Reuses the same SUIS endpoints as scraper.scrape_cs, but for minor programs.
Minors list their requirement courses inline (no elective pool pages) under
section anchors of the form ``<PREFIX>_<SUFFIX>``:

  _REQ              -> Required
  _CEL / _CORE / _ELEC -> Core Elective
  _ARE / _AEL       -> Area Elective

Course *details* (title, credits, prereqs) are NOT refetched here — they are
already in data/SU_full_catalog.json. This module only captures the thin
``minor -> {section -> [course codes]}`` mapping, which is all the advisory
tool needs.

Run:
  python -m scraper.build_minors
  python -m scraper.build_minors --term 202601
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

BASE = "https://suis.sabanciuniv.edu/prod/"
DEFAULT_TERM = "202601"
REQUEST_DELAY_S = 0.4  # be polite

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_PATH = DATA_DIR / "minors.json"

MINOR_LIST_URL = BASE + "SU_DEGREE.p_list_degree?P_LEVEL=UG&P_LANG=EN&P_PRG_TYPE=MINOR"

# Minor section-anchor suffix -> canonical label.
MINOR_SECTION_SUFFIX: dict[str, str] = {
    "REQ": "Required",
    "CEL": "Core Elective",
    "CORE": "Core Elective",
    "ELEC": "Core Elective",
    "ARE": "Area Elective",
    "AEL": "Area Elective",
}

COURSE_HREF_RE = re.compile(
    r"sabanci_www\.p_get_courses\?levl_code=UG&subj_code=([A-Z]+)&crse_numb=([A-Z0-9]+)",
    re.IGNORECASE,
)

session = requests.Session()
session.headers.update({"User-Agent": "sucheduler-scraper/0.1 (course catalog research)"})


def _get(url: str) -> str:
    r = session.get(url, timeout=30)
    r.raise_for_status()
    time.sleep(REQUEST_DELAY_S)
    return r.text


def fetch_minor_list() -> dict[str, str]:
    """Return {minor_code: display_name} from the UG minor list page."""
    html = _get(MINOR_LIST_URL)
    soup = BeautifulSoup(html, "html.parser")
    out: dict[str, str] = {}
    for a in soup.find_all("a", href=True):
        m = re.search(r"P_PROGRAM=([A-Z0-9_\-]+)", a["href"], re.I)
        if m:
            name = re.sub(r"\s+", " ", a.get_text(" ", strip=True)).strip()
            out[m.group(1).upper()] = name
    return out


def fetch_minor_detail(code: str, term: str) -> str:
    url = (
        f"{BASE}SU_DEGREE.p_degree_detail"
        f"?P_PROGRAM={code}&P_LANG=EN&P_LEVEL=UG&P_TERM={term}&P_SUBMIT=Select"
    )
    return _get(url)


def _section_for_anchor(anchor: str) -> str | None:
    """Map a minor section anchor (e.g. 'FIN_CEL') to a canonical label."""
    if "_" not in anchor:
        return None
    suffix = anchor.rsplit("_", 1)[1].upper()
    return MINOR_SECTION_SUFFIX.get(suffix)


def parse_minor_detail(html: str) -> dict[str, list[str]]:
    """Return {section_label: [course codes]} for one minor detail page."""
    soup = BeautifulSoup(html, "html.parser")
    sections: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    current: str | None = None
    for el in soup.find_all("a"):
        name = el.get("name")
        if name:
            sec = _section_for_anchor(name)
            if sec:
                current = sec
                sections.setdefault(current, [])
            continue
        href = el.get("href") or ""
        m = COURSE_HREF_RE.search(href)
        if m and current:
            code = f"{m.group(1).upper()} {m.group(2).upper()}"
            key = (code, current)
            if key not in seen:
                seen.add(key)
                sections[current].append(code)
    return sections


def _num(cell: str) -> int | None:
    """Parse a summary-table credit/course cell ('-' or '' -> None)."""
    cell = cell.strip()
    if not cell or cell == "-":
        return None
    try:
        return int(float(cell))
    except ValueError:
        return None


def parse_minor_requirements(html: str) -> dict[str, dict]:
    """Parse the "SUMMARY OF DEGREE REQUIREMENTS" table.

    Returns ``{section_label: {min_courses, min_su, min_ects}}`` with an extra
    ``"total"`` entry. Section rows link to their anchor (e.g. ``#FIN_REQ``),
    which is mapped to a canonical label; the ``Total`` row has no anchor. Some
    minors specify only credits (no course count), so ``min_courses`` may be
    ``None``.
    """
    soup = BeautifulSoup(html, "html.parser")
    out: dict[str, dict] = {}
    for table in soup.find_all("table"):
        htxt = table.get_text(" ", strip=True)
        if "Min. Courses" not in htxt or "Course Category" not in htxt:
            continue
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all("td")]
            if len(cells) != 4:
                continue
            anchor = tr.find("a", href=True)
            if anchor and anchor["href"].startswith("#"):
                label = _section_for_anchor(anchor["href"][1:])
            elif cells[0].strip().lower() == "total":
                label = "total"
            else:
                label = None
            if not label:
                continue
            out[label] = {
                "min_ects": _num(cells[1]),
                "min_su": _num(cells[2]),
                "min_courses": _num(cells[3]),
            }
        break
    return out


def parse_minor_choice_groups(html: str, listed_codes: set[str]) -> list[dict]:
    """Extract either/or choice groups from a minor detail page's footnotes.

    Uses the same footnote parser as the degree extractor
    (``scraper.extract_choice_groups``). A group is kept only if anchored to a
    course actually listed in this minor, which filters out unrelated "X or Y"
    prose. Returns groups in the config form consumed by ``scheduler.minors``;
    an empty list when none are found or parsing fails.
    """
    try:
        from scraper.extract_choice_groups import (
            canonical,
            emit_group,
            groups_from_html,
        )

        listed = {c.upper().strip() for c in listed_codes}
        deduped: dict = {}
        for alts in groups_from_html(html):
            all_codes = {c for alt in alts for c in alt}
            if all_codes & listed:
                deduped[canonical(alts)] = alts
        return [emit_group(a) for a in deduped.values()]
    except Exception:
        return []


def build(term: str = DEFAULT_TERM, verbose: bool = True) -> dict:
    names = fetch_minor_list()
    if verbose:
        print(f"found {len(names)} minors")
    minors: dict[str, dict] = {}
    for code, name in sorted(names.items()):
        try:
            html = fetch_minor_detail(code, term)
            sections = parse_minor_detail(html)
            requirements = parse_minor_requirements(html)
        except Exception as exc:
            if verbose:
                print(f"  ! {code}: {exc}")
            continue
        total = sum(len(v) for v in sections.values())
        if total == 0:
            if verbose:
                print(f"  - {code}: no courses (skipped)")
            continue
        listed_codes = {c for codes in sections.values() for c in codes}
        choice_groups = parse_minor_choice_groups(html, listed_codes)
        minors[code] = {
            "code": code,
            "name": name,
            "term": term,
            "sections": sections,
            "requirements": requirements,
            "course_count": total,
        }
        if choice_groups:
            minors[code]["choice_groups"] = choice_groups
        if verbose:
            secsum = ", ".join(f"{k}:{len(v)}" for k, v in sections.items())
            extra = f"  [+{len(choice_groups)} choice group(s)]" if choice_groups else ""
            print(f"  {code:14s} {total:>3} courses  ({secsum}){extra}")

    payload = {
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "term": term,
        "minor_count": len(minors),
        "minors": minors,
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(f"wrote {OUT_PATH} ({len(minors)} minors)")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--term", default=DEFAULT_TERM, help="cohort term to scrape")
    args = ap.parse_args()
    build(term=args.term)


if __name__ == "__main__":
    main()
