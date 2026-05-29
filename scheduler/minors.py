"""Minor requirement lookup and per-student progress (advisory only).

Reads ``data/minors.json`` (produced by ``scraper.build_minors``) and reports
what a student has done toward a given minor. Course titles/credits come from
``scheduler.catalog``. This module is advisory: it does NOT enforce minor
completion in term-plan validation.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MINORS_PATH = ROOT / "data" / "minors.json"

SECTION_ORDER = ["Required", "Core Elective", "Area Elective"]


def _normalize_groups(raw: list | None) -> list[list[list[str]]]:
    """Normalize a minor's ``choice_groups`` config into lists of alternatives.

    Accepts both ``{"choose": 1, "options": [...]}`` and
    ``{"satisfied_by_any_of": [[...], [...]]}`` (see scheduler.requirements for
    the satisfaction semantics). Each alternative is a list of codes that must
    ALL be completed to satisfy the group.
    """
    out: list[list[list[str]]] = []
    for g in raw or []:
        if not isinstance(g, dict):
            continue
        if g.get("satisfied_by_any_of"):
            alts = [list(a) for a in g["satisfied_by_any_of"] if a]
        elif g.get("options"):
            if int(g.get("choose", 1)) != 1:
                continue
            alts = [[opt] for opt in g["options"]]
        else:
            continue
        if alts:
            out.append(alts)
    return out


def _suppressed_codes(
    groups: list[list[list[str]]], done: set[str], wip: set[str]
) -> set[str]:
    """Codes to hide from a minor's 'remaining' list because of either/or groups.

    For a satisfied or on-track group, all its members are suppressed. For an
    unmet group, only the redundant alternatives are suppressed — one concrete
    path is kept so the student still sees what to take.
    """
    done_or_wip = done | wip
    suppress: set[str] = set()
    for alts in groups:
        members = {c for alt in alts for c in alt}
        if any(all(c in done for c in alt) for alt in alts):
            suppress |= members
            continue
        if any(all(c in done_or_wip for c in alt) for alt in alts):
            suppress |= members
            continue

        def _untaken(alt: list[str]) -> list[str]:
            return [c for c in alt if c not in done_or_wip]

        best = min(alts, key=lambda a: (len(_untaken(a)), -sum(c in done for c in a)))
        suppress |= members - set(_untaken(best))
    return suppress


class MinorCatalog:
    def __init__(self, data: dict) -> None:
        self._minors: dict[str, dict] = data.get("minors", {})
        self.term: str = data.get("term", "")

    @classmethod
    def load(cls, path: str | Path = MINORS_PATH) -> "MinorCatalog":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(raw)

    # -- listing / resolution ------------------------------------------------

    def list_minors(self) -> list[dict]:
        return [
            {"code": code, "name": m.get("name", code), "course_count": m.get("course_count", 0)}
            for code, m in sorted(self._minors.items())
        ]

    def get(self, code: str) -> dict | None:
        return self._minors.get(code.upper().strip())

    def resolve(self, query: str) -> tuple[str | None, list[str]]:
        """Resolve a free-text minor reference to a code.

        Returns (code, candidates). ``code`` is set when the match is
        unambiguous; otherwise ``candidates`` holds the possible codes so the
        caller can disambiguate.
        """
        if not query or not query.strip():
            return None, []
        q = query.strip().upper()
        # Direct code forms: "FIN-MINOR", "FIN", "MATH MINOR"
        if q in self._minors:
            return q, [q]
        stem = q.replace("MINOR", "").replace("-", " ").strip().replace(" ", "")
        if f"{stem}-MINOR" in self._minors:
            return f"{stem}-MINOR", [f"{stem}-MINOR"]
        # Name substring match (e.g. "finance" -> "Finance Minor")
        ql = query.strip().lower().replace("minor", "").strip()
        if ql:
            matches = [c for c, m in self._minors.items() if ql in m.get("name", "").lower()]
            if len(matches) == 1:
                return matches[0], matches
            if matches:
                return None, sorted(matches)
        return None, []

    # -- progress ------------------------------------------------------------

    def progress(
        self,
        code: str,
        completed: set[str],
        in_progress: set[str],
        catalog=None,
    ) -> dict:
        """Per-section breakdown of done / in-progress / remaining courses."""
        minor = self.get(code)
        if minor is None:
            return {"error": f"unknown minor: {code}"}

        done = {c.upper().strip() for c in completed}
        wip = {c.upper().strip() for c in in_progress}

        # Either/or footnote groups (e.g. "take IR 391 or IR 394"): hide unused
        # alternatives from 'remaining' so completing one isn't reported as
        # leaving the others outstanding.
        groups = _normalize_groups(minor.get("choice_groups"))
        suppress = _suppressed_codes(groups, done, wip)

        def _title(c: str) -> str:
            if catalog is not None:
                course = catalog.get(c)
                if course:
                    return course.title
            return ""

        sections_out: dict[str, dict] = {}
        total_done = 0
        total_in_section = 0
        for section in SECTION_ORDER:
            codes = minor.get("sections", {}).get(section)
            if not codes:
                continue
            sec_done = [c for c in codes if c in done]
            sec_wip = [c for c in codes if c in wip and c not in done]
            sec_left = [
                c for c in codes
                if c not in done and c not in wip and c not in suppress
            ]
            total_done += len(sec_done)
            total_in_section += len(codes)
            sections_out[section] = {
                "total": len(codes),
                "completed": sorted(sec_done),
                "in_progress": sorted(sec_wip),
                "remaining": [
                    {"code": c, "title": _title(c)} for c in sec_left
                ],
            }

        return {
            "code": minor["code"],
            "name": minor.get("name", minor["code"]),
            "term": minor.get("term", self.term),
            "completed_count": total_done,
            "total_listed": total_in_section,
            "note": (
                "Advisory only. Elective sections usually require choosing a "
                "subset, not all listed courses; consult the official minor "
                "page for exact counts."
            ),
            "sections": sections_out,
        }
