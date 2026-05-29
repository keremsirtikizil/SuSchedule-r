"""
Time-conflict checker and feasible-timetable builder.

Given a target term and a list of course codes the student wants to take,
this module figures out whether there is a combination of CRN sections (one
per course, plus required coreqs) whose weekly meeting times do not overlap.

We work off ``data/offerings.json`` (produced by ``scraper.scrape_offerings``)
which now carries per-section meeting times with normalised
``days`` (0=Mon..6=Sun) and ``start`` / ``end`` in minutes-since-midnight.

Public surface:

  * ``meetings_overlap(a, b)``         -- pairwise meeting check
  * ``sections_conflict(s1, s2)``      -- pairwise section check
  * ``conflict_report(sections)``      -- list every conflicting pair
  * ``build_timetable(plan, term, ...)``-- backtracking search for a feasible
                                          assignment; returns a list of picks
                                          or a structured failure report.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OFFERINGS = ROOT / "data" / "offerings.json"


DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _fmt_clock(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    return f"{h:02d}:{m:02d}"


# --------------------------------------------------------------------------- #
# Conflict checks
# --------------------------------------------------------------------------- #


def meetings_overlap(a: dict, b: dict) -> bool:
    """Return True if two meetings share at least one day AND time overlaps."""
    days_a, days_b = set(a.get("days") or []), set(b.get("days") or [])
    if not days_a or not days_b:
        return False
    if days_a.isdisjoint(days_b):
        return False
    return a["start"] < b["end"] and b["start"] < a["end"]


def sections_conflict(s1: dict, s2: dict) -> bool:
    for m1 in s1.get("meetings", []):
        for m2 in s2.get("meetings", []):
            if meetings_overlap(m1, m2):
                return True
    return False


def conflict_report(picks: list[dict]) -> list[tuple[str, str, dict, dict]]:
    """For every conflicting pair of picks, yield (label_i, label_j, m_i, m_j)."""
    conflicts = []
    for i in range(len(picks)):
        for j in range(i + 1, len(picks)):
            a = picks[i]
            b = picks[j]
            for ma in a.get("meetings", []):
                for mb in b.get("meetings", []):
                    if meetings_overlap(ma, mb):
                        conflicts.append((a["label"], b["label"], ma, mb))
    return conflicts


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #


def load_offerings(path: str | Path = DEFAULT_OFFERINGS) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sections_for(off: dict, term: str, code: str) -> list[dict]:
    """Return every section listed for ``code`` in ``term``."""
    return list(off.get(term, {}).get(code, []))


def resolve_schedule_term(offerings: dict, target_term: str) -> tuple[str, bool]:
    """Pick the term whose published meeting times we actually schedule against.

    Future planning terms (e.g. 202601) usually have no offerings yet. When the
    target term is missing we fall back to the most recent term of the *same
    season* and flag it as a proxy. Returns ``(schedule_term, is_proxy)``;
    ``schedule_term`` is empty if no same-season data exists at all.
    """
    if target_term in offerings:
        return target_term, False
    from scheduler.offerings import terms_of_season, season_of

    same_season = terms_of_season(offerings, season_of(target_term))
    if same_season:
        return same_season[0], True
    return "", False


def annotate_meeting(m: dict) -> dict:
    """Add human-readable day/clock labels to a meeting dict (in place)."""
    m["start_label"] = _fmt_clock(m["start"]) if "start" in m else ""
    m["end_label"] = _fmt_clock(m["end"]) if "end" in m else ""
    m["day_labels"] = [DAY_LABELS[d] for d in m.get("days", []) if 0 <= d < 7]
    return m


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


@dataclass
class TimetableResult:
    ok: bool
    term: str
    picks: list[dict] = field(default_factory=list)        # chosen sections
    missing: list[str] = field(default_factory=list)       # courses with no section
    no_meetings: list[str] = field(default_factory=list)   # offered but TBA
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "term": self.term,
            "missing": self.missing,
            "no_meetings": self.no_meetings,
            "reason": self.reason,
            "picks": [
                {
                    "course": p["code"],
                    "crn": p["crn"],
                    "section": p["section"],
                    "title": p["title"],
                    "instructors": p.get("instructors", []),
                    "meetings": p["meetings"],
                }
                for p in self.picks
            ],
        }


def _section_label(code: str, sec: dict) -> str:
    return f"{code} ({sec['section']}, CRN {sec['crn']})"


def build_timetable(
    plan: list[str],
    term: str,
    offerings: dict | None = None,
    skip_no_meeting: bool = True,
) -> TimetableResult:
    """Search for a feasible (conflict-free) section assignment.

    Returns ``TimetableResult.ok == True`` iff one was found. If a course
    has no sections in ``term`` we record it in ``missing``. If every section
    of a course has empty meeting times (TBA / arranged), we either skip the
    course (``skip_no_meeting=True``) or include the first such section.
    """
    off = offerings if offerings is not None else load_offerings()

    # Collect candidate sections per course
    candidates: list[tuple[str, list[dict]]] = []
    missing: list[str] = []
    no_meetings: list[str] = []
    for code in plan:
        sects = sections_for(off, term, code)
        if not sects:
            missing.append(code)
            continue
        with_times = [s for s in sects if s.get("meetings")]
        if not with_times:
            no_meetings.append(code)
            if skip_no_meeting:
                continue
            with_times = sects[:1]
        # Normalise: attach the course code + a label so the search can reason
        for s in with_times:
            s.setdefault("code", code)
            s.setdefault("label", _section_label(code, s))
        candidates.append((code, with_times))

    if missing and not candidates:
        return TimetableResult(False, term, missing=missing, no_meetings=no_meetings,
                               reason="no scheduleable courses")

    # Heuristic: try courses with fewer sections first (more constrained).
    candidates.sort(key=lambda kv: len(kv[1]))

    picks: list[dict] = []

    def backtrack(idx: int) -> bool:
        if idx == len(candidates):
            return True
        _, sects = candidates[idx]
        for s in sects:
            if any(sections_conflict(s, p) for p in picks):
                continue
            picks.append(s)
            if backtrack(idx + 1):
                return True
            picks.pop()
        return False

    found = backtrack(0)
    if not found:
        return TimetableResult(
            False, term,
            missing=missing,
            no_meetings=no_meetings,
            reason="no conflict-free combination exists",
        )
    return TimetableResult(
        True, term, picks=list(picks),
        missing=missing, no_meetings=no_meetings,
    )


# --------------------------------------------------------------------------- #
# Pretty-print
# --------------------------------------------------------------------------- #


def format_timetable(result: TimetableResult) -> str:
    if not result.ok:
        lines = [f"No feasible timetable for term {result.term}: {result.reason}"]
        if result.missing:
            lines.append(f"  missing in term:    {result.missing}")
        if result.no_meetings:
            lines.append(f"  TBA / no meetings:  {result.no_meetings}")
        return "\n".join(lines)

    # Build a Mon-Fri grid sorted by start time per day.
    grid: dict[int, list[tuple[int, int, str]]] = {d: [] for d in range(7)}
    for p in result.picks:
        for m in p["meetings"]:
            for day in m["days"]:
                grid[day].append((m["start"], m["end"], p["label"], m.get("where", "")))
    lines = [f"Feasible timetable for {result.term}:"]
    for p in result.picks:
        meets = "; ".join(
            f"{m['days_raw']} {_fmt_clock(m['start'])}-{_fmt_clock(m['end'])}"
            for m in p["meetings"]
        )
        lines.append(f"  {p['label']:<28}  {meets}")
    if result.no_meetings:
        lines.append(f"  (no scheduled meetings: {result.no_meetings})")
    if result.missing:
        lines.append(f"  (not offered this term: {result.missing})")
    lines.append("")
    lines.append("Weekly grid:")
    for d in range(5):  # Mon-Fri
        items = sorted(grid[d])
        if not items:
            continue
        lines.append(f"  {DAY_LABELS[d]}:")
        for start, end, label, where in items:
            lines.append(f"    {_fmt_clock(start)}-{_fmt_clock(end)}  {label}  ({where})")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--term", required=True, help="Target term code, e.g. 202501")
    ap.add_argument("--plan", nargs="+", required=True, help="Course codes to schedule")
    args = ap.parse_args()
    result = build_timetable(args.plan, args.term)
    print(format_timetable(result))


if __name__ == "__main__":
    _cli()
