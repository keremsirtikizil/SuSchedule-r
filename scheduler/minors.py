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
        """Per-section breakdown of done / in-progress / remaining courses.

        Uses the scraped per-section requirements (min courses / min SU
        credits) so that "how many more do I need" answers reflect the
        *required subset* of each elective pool, not the full list.
        """
        minor = self.get(code)
        if minor is None:
            return {"error": f"unknown minor: {code}"}

        done = {c.upper().strip() for c in completed}
        wip = {c.upper().strip() for c in in_progress}
        reqs = minor.get("requirements", {})

        def _course(c: str):
            return catalog.get(c) if catalog is not None else None

        def _title(c: str) -> str:
            course = _course(c)
            return course.title if course else ""

        def _su(c: str) -> float:
            course = _course(c)
            return float(course.su_credit) if course and course.su_credit else 0.0

        sections_out: dict[str, dict] = {}
        agg = {
            "courses_done": 0,
            "courses_remaining": 0,
            "su_done": 0.0,
            "su_remaining": 0.0,
            "has_course_req": False,
            "has_su_req": False,
        }
        for section in SECTION_ORDER:
            codes = minor.get("sections", {}).get(section)
            req = reqs.get(section)
            if not codes and not req:
                continue
            codes = codes or []
            sec_done = [c for c in codes if c in done]
            sec_wip = [c for c in codes if c in wip and c not in done]
            sec_left = [c for c in codes if c not in done and c not in wip]

            min_courses = (req or {}).get("min_courses")
            min_su = (req or {}).get("min_su")
            su_done = sum(_su(c) for c in sec_done)

            # Courses still required from this section (counting in-progress as
            # on track toward the minimum).
            courses_remaining = None
            if min_courses is not None:
                courses_remaining = max(0, min_courses - len(sec_done) - len(sec_wip))
            su_remaining = None
            if min_su is not None:
                su_in_flight = su_done + sum(_su(c) for c in sec_wip)
                su_remaining = max(0.0, min_su - su_in_flight)

            agg["courses_done"] += len(sec_done)
            agg["su_done"] += su_done
            if courses_remaining is not None:
                agg["courses_remaining"] += courses_remaining
                agg["has_course_req"] = True
            if su_remaining is not None:
                agg["su_remaining"] += su_remaining
                agg["has_su_req"] = True

            sections_out[section] = {
                "min_courses": min_courses,
                "min_su_credits": min_su,
                "completed": sorted(sec_done),
                "in_progress": sorted(sec_wip),
                "courses_remaining": courses_remaining,
                "su_credits_remaining": su_remaining,
                "options": [{"code": c, "title": _title(c)} for c in sec_left],
            }

        total = reqs.get("total", {})
        result = {
            "code": minor["code"],
            "name": minor.get("name", minor["code"]),
            "term": minor.get("term", self.term),
            "credit_unit": "SU credit",
            "required_total": {
                "min_courses": total.get("min_courses"),
                "min_su_credits": total.get("min_su"),
            },
            "completed_courses": agg["courses_done"],
            "completed_su_credits": agg["su_done"],
            "courses_remaining": agg["courses_remaining"] if agg["has_course_req"] else None,
            "su_credits_remaining": agg["su_remaining"] if agg["has_su_req"] else None,
            "note": (
                "Advisory only. All credit figures are SU credits. 'options' "
                "lists the pool for each elective section; you choose the "
                "required number from it. Extra core electives can often count "
                "toward area electives — consult the official minor page for "
                "exact rules."
            ),
            "sections": sections_out,
        }
        # ECTS is published by SUIS only as a whole-minor grand total, with no
        # per-section or per-course breakdown — so completed/remaining ECTS
        # cannot be derived. Expose it as a clearly-scoped FYI only.
        total_ects = total.get("min_ects")
        if total_ects is not None:
            result["total_ects_whole_minor"] = total_ects
            result["ects_note"] = (
                "Whole-minor ECTS total only; no per-course ECTS data, so "
                "completed/remaining ECTS is unknown. Do not estimate it."
            )
        return result
