"""Science & Engineering credit tracking (degree-evaluation feature).

Sabancı engineering degrees require a minimum number of **Engineering** ECTS
and **Basic-Science** ECTS to graduate. Two data sources feed this:

* ``data/eng_sci_credits.json`` — per-course Engineering / Basic-Science ECTS
  split (built by ``scraper.build_eng_sci`` from the catalog PDF).
* the student's Degree Evaluation (``scheduler.degree_eval``) — the per-degree
  ``ENGINEERING`` / ``BASIC SCIENCE`` minimum + completed ECTS.

This module combines them so the planner can:
  1. report how many Engineering / Basic-Science ECTS are still needed, and
  2. tell how much a candidate course (or a whole term plan) would contribute.

Advisory: if a degree evaluation was not uploaded we fall back to summing the
per-course table over completed courses, which is approximate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENG_SCI_PATH = ROOT / "data" / "eng_sci_credits.json"


@dataclass
class CourseEngSci:
    code: str
    ects_total: float
    eng_ects: float
    basic_sci_ects: float


class EngSciCredits:
    """Per-course Engineering / Basic-Science ECTS lookup."""

    def __init__(self, data: dict) -> None:
        self._courses: dict[str, dict] = data.get("courses", {})
        self.term: str = data.get("term", "")
        self.source: str = data.get("source", "")

    @classmethod
    def load(cls, path: str | Path = ENG_SCI_PATH) -> "EngSciCredits":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(raw)

    def get(self, code: str) -> CourseEngSci | None:
        rec = self._courses.get(code.upper().strip())
        if rec is None:
            return None
        return CourseEngSci(
            code=code.upper().strip(),
            ects_total=float(rec.get("ects_total", 0.0)),
            eng_ects=float(rec.get("eng_ects", 0.0)),
            basic_sci_ects=float(rec.get("basic_sci_ects", 0.0)),
        )

    def sum_over(self, codes) -> tuple[float, float]:
        """Return (engineering_ects, basic_science_ects) summed over ``codes``."""
        eng = sci = 0.0
        for c in codes:
            rec = self.get(c)
            if rec:
                eng += rec.eng_ects
                sci += rec.basic_sci_ects
        return eng, sci

    def contributions(self, codes) -> list[dict]:
        """Per-course Engineering / Basic-Science contribution for ``codes``.

        Only includes courses that carry any Engineering or Basic-Science ECTS.
        """
        out: list[dict] = []
        for c in codes:
            rec = self.get(c)
            if rec and (rec.eng_ects or rec.basic_sci_ects):
                out.append({
                    "code": rec.code,
                    "eng_ects": rec.eng_ects,
                    "basic_sci_ects": rec.basic_sci_ects,
                })
        return out


def progress(
    section_requirements: dict | None,
    completed: set[str],
    in_progress: set[str],
    credits: EngSciCredits | None = None,
) -> dict:
    """Engineering / Basic-Science credit progress and remaining gap.

    Prefers the authoritative minimum + completed ECTS from the degree
    evaluation's ``section_requirements``. If those are missing, estimates
    completed ECTS from the per-course table.
    """
    if credits is None:
        credits = EngSciCredits.load()

    section_requirements = section_requirements or {}

    def _section(name: str) -> dict:
        req = section_requirements.get(name, {}) or {}
        min_ects = float(req.get("min_ects", 0.0)) or None
        completed_ects = req.get("completed_ects")
        return {"min_ects": min_ects, "completed_ects": completed_ects}

    eng = _section("ENGINEERING")
    sci = _section("BASIC SCIENCE")

    # Estimate from the per-course table when the evaluation didn't provide it.
    est_eng_done, est_sci_done = credits.sum_over(completed)
    est_eng_wip, est_sci_wip = credits.sum_over(in_progress)

    def _pack(section: dict, est_done: float, est_wip: float) -> dict:
        min_ects = section["min_ects"]
        completed_ects = section["completed_ects"]
        source = "degree_evaluation"
        if completed_ects is None:
            completed_ects = round(est_done, 1)
            source = "estimated_from_catalog"
        remaining = None
        if min_ects is not None:
            remaining = round(max(0.0, min_ects - float(completed_ects)), 1)
        return {
            "min_ects": min_ects,
            "completed_ects": float(completed_ects),
            "in_progress_ects": round(est_wip, 1),
            "remaining_ects": remaining,
            "met": (remaining == 0.0) if remaining is not None else None,
            "source": source,
        }

    return {
        "engineering": _pack(eng, est_eng_done, est_eng_wip),
        "basic_science": _pack(sci, est_sci_done, est_sci_wip),
        "note": (
            "Engineering and Basic-Science ECTS minimums must both be met to "
            "graduate. Per-course splits come from the catalog; the degree "
            "evaluation is authoritative for totals."
        ),
    }


def _smoke_test() -> None:
    from scheduler.degree_eval import parse_degree_evaluation

    prof = parse_degree_evaluation(ROOT / "DEGREE EVALUATION.html")
    rep = progress(
        prof.get("section_requirements"),
        set(prof["completed"]),
        set(prof["in_progress"]),
    )
    import pprint
    pprint.pp(rep)
    creds = EngSciCredits.load()
    print("\nCandidate contributions (CS 306, MATH 202, PHYS 301):")
    pprint.pp(creds.contributions(["CS 306", "MATH 202", "PHYS 301"]))


if __name__ == "__main__":
    _smoke_test()
