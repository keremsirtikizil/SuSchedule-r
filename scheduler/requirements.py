"""
Graduation requirements tracker.

Reads the per-degree, per-section DAGs produced by
``scraper.build_degree_graphs`` and computes what a student still needs
to complete for each section of their degree.

Public surface
--------------
``compute_remaining(program, student, data_dir)``
    Returns a ``RequirementsReport`` with four lists:
    required_left, core_left, area_left, free_left — each a list of
    course codes the student has NOT yet completed or enrolled in.

The agent uses this to:
  1. Tell the LLM what graduation requirements remain (context).
  2. Auto-inject still-needed required courses into the candidate pool
     so the LLM is never tempted to ignore them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from scheduler.graph_selector import DEGREE_GRAPHS_DIR, cohort_for_admit

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = DEGREE_GRAPHS_DIR

SECTION_FILES = {
    "required":       "Required.json",
    "core_elective":  "Core_Elective.json",
    "area_elective":  "Area_Elective.json",
    "free_elective":  "Free_Elective.json",
}


# --------------------------------------------------------------------------- #
# Data class
# --------------------------------------------------------------------------- #

@dataclass
class RequirementsReport:
    program: str
    cohort_term: str = ""

    # Codes still needed (not completed, not in-progress)
    required_left:      list[str] = field(default_factory=list)
    core_left:          list[str] = field(default_factory=list)
    area_left:          list[str] = field(default_factory=list)
    free_left:          list[str] = field(default_factory=list)

    # Codes the student is currently enrolled in (good — counts toward graduation)
    required_in_progress:     list[str] = field(default_factory=list)
    core_in_progress:         list[str] = field(default_factory=list)

    # Credit totals for each section (0-credit courses like ENS 491 included)
    required_credits_left:    float = 0.0

    def all_left(self) -> list[str]:
        """All still-needed codes across every section."""
        return (
            self.required_left
            + self.core_left
            + self.area_left
            + self.free_left
        )

    def summary_lines(self) -> list[str]:
        """Human-readable lines for the planner prompt."""
        lines = []
        def _fmt(label, left, in_prog):
            parts = [f"{label} ({len(left)} still needed)"]
            if left:
                parts.append("  Need: " + ", ".join(left))
            if in_prog:
                parts.append("  In progress: " + ", ".join(in_prog))
            return "\n".join(parts)

        lines.append(_fmt("Required", self.required_left, self.required_in_progress))
        lines.append(_fmt("Core Electives", self.core_left, self.core_in_progress))
        if self.area_left:
            lines.append(f"Area Electives ({len(self.area_left)} still needed): "
                         + ", ".join(self.area_left[:10])
                         + ("…" if len(self.area_left) > 10 else ""))
        if self.free_left:
            lines.append(f"Free Electives ({len(self.free_left)} still needed)")
        return lines

    def to_dict(self) -> dict:
        return {
            "program": self.program,
            "cohort_term": self.cohort_term,
            "required_left": self.required_left,
            "core_left": self.core_left,
            "area_left": self.area_left,
            "free_left": self.free_left,
            "required_in_progress": self.required_in_progress,
            "required_credits_left": self.required_credits_left,
        }


# --------------------------------------------------------------------------- #
# Loader helpers
# --------------------------------------------------------------------------- #

def _load_in_slice_codes(json_path: Path) -> dict[str, float]:
    """Return {code: su_credit} for all in-slice nodes in a degree graph."""
    if not json_path.exists():
        return {}
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return {
        n["code"]: float(n.get("su_credit") or 0)
        for n in data.get("nodes", [])
        if n.get("in_slice")
    }


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def compute_remaining(
    program: str,
    completed_for_eligibility: set[str],
    in_progress: set[str],
    admit_term: str | None = None,
    data_dir: Path = DATA_DIR,
) -> RequirementsReport:
    """Compute what the student still needs for each degree section.

    Parameters
    ----------
    program:
        Degree code, e.g. ``"BSCS-DM"``.
    completed_for_eligibility:
        ``Student.completed_for_eligibility`` — includes transfers, passed
        courses, and courses that count for graduation.
    in_progress:
        ``Student.in_progress`` — currently enrolled.
    admit_term:
        Student admit term, mapped to the closest available cohort directory.
    data_dir:
        Directory containing ``<PROGRAM>/<COHORT>/Required.json`` etc.
        Defaults to ``data/degree_graphs/``.
    """
    cohort = cohort_for_admit(program, admit_term, data_dir)
    prog_dir = data_dir / program / cohort
    report = RequirementsReport(program=program, cohort_term=cohort)

    done = set(completed_for_eligibility)
    wip = set(in_progress)
    all_done_or_wip = done | wip

    # Required ----------------------------------------------------------------
    req_codes = _load_in_slice_codes(prog_dir / SECTION_FILES["required"])
    for code, credits in req_codes.items():
        if code in done:
            continue
        if code in wip:
            report.required_in_progress.append(code)
        else:
            report.required_left.append(code)
            report.required_credits_left += credits

    # Core elective -----------------------------------------------------------
    core_codes = _load_in_slice_codes(prog_dir / SECTION_FILES["core_elective"])
    for code in core_codes:
        if code in done:
            continue
        if code in wip:
            report.core_in_progress.append(code)
        else:
            report.core_left.append(code)

    # Area elective -----------------------------------------------------------
    area_codes = _load_in_slice_codes(prog_dir / SECTION_FILES["area_elective"])
    for code in area_codes:
        if code not in all_done_or_wip:
            report.area_left.append(code)

    # Free elective -----------------------------------------------------------
    free_codes = _load_in_slice_codes(prog_dir / SECTION_FILES["free_elective"])
    for code in free_codes:
        if code not in all_done_or_wip:
            report.free_left.append(code)

    # Sort for deterministic output
    report.required_left.sort()
    report.core_left.sort()
    report.area_left.sort()
    report.free_left.sort()

    return report


# --------------------------------------------------------------------------- #
# CLI smoke test
# --------------------------------------------------------------------------- #

def _smoke_test() -> None:
    import json as _json
    from pathlib import Path as _Path

    transcript = _json.loads(
        (_Path(__file__).resolve().parent.parent / "data" / "transcript_cagan.json")
        .read_text(encoding="utf-8")
    )
    completed = set(transcript.get("completed_for_eligibility", []))
    in_progress = set(transcript.get("in_progress", []))
    program = transcript.get("program", "BSCS-DM")

    report = compute_remaining(
        program,
        completed,
        in_progress,
        admit_term=transcript.get("admit_term"),
    )
    print(f"Requirements report for {program}\n" + "=" * 50)
    for line in report.summary_lines():
        print(line)
        print()
    print(f"Total required credits still needed: {report.required_credits_left}")


if __name__ == "__main__":
    _smoke_test()
