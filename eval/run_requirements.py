"""
Degree-requirement evaluation for SuSchedule-r.

Ground truth = the student's OFFICIAL Banner "Degree Evaluation" HTML, which
reports, per requirement section, the minimum SU credits required and the SU
credits already completed. We compare the official REQUIRED COURSES credit gap
against the graph-based requirement engine `compute_remaining`, which the agent
uses to decide which mandatory courses are still missing.

The audit's GENERAL row is the authoritative overall program total. Section
rows overlap with each other and must not be summed together.

Runs locally (no OpenAI). Usage:
    python -m eval.run_requirements
"""
from __future__ import annotations

import json
from pathlib import Path

from scheduler.degree_eval import parse_degree_evaluation
from scheduler.requirements import compute_remaining

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"
DEGREE_HTML = ROOT / "DEGREE EVALUATION.html"

# These five sections add up to the student's 125-SU program total. Faculty,
# engineering, and science rows classify courses that are already counted here.
PRIMARY_SU_SECTIONS = {
    "UNIVERSITY COURSES",
    "CORE ELECTIVES",
    "REQUIRED COURSES",
    "AREA ELECTIVES",
    "FREE ELECTIVES",
}

SECTION_ORDER = [
    "UNIVERSITY COURSES",
    "CORE ELECTIVES",
    "REQUIRED COURSES",
    "AREA ELECTIVES",
    "FREE ELECTIVES",
    "FACULTY COURSES",
    "ENGINEERING",
    "BASIC SCIENCE",
    "GENERAL",
]


def main() -> None:
    raw = parse_degree_evaluation(DEGREE_HTML)
    sr = raw["section_requirements"]

    # Read the remaining SU credits shown in each audit section.
    sections = []
    for name, d in sr.items():
        if not isinstance(d, dict):
            continue
        mn = float(d.get("min_su", 0) or 0)
        dn = float(d.get("completed_su", 0) or 0)
        rem = max(mn - dn, 0.0)
        sections.append({"section": name, "min_su": mn, "completed_su": dn, "remaining_su": rem})
    section_rank = {name: i for i, name in enumerate(SECTION_ORDER)}
    sections.sort(key=lambda s: (section_rank.get(s["section"], len(SECTION_ORDER)), s["section"]))

    primary = [s for s in sections if s["section"] in PRIMARY_SU_SECTIONS]
    primary_min = sum(s["min_su"] for s in primary)
    primary_done = sum(s["completed_su"] for s in primary)

    general = next((s for s in sections if s["section"] == "GENERAL"), None)
    if general is not None:
        total_source = "GENERAL"
        total_min = general["min_su"]
        total_done = general["completed_su"]
    else:
        total_source = "sum_of_primary_non_overlapping_sections"
        total_min = primary_min
        total_done = primary_done
    overall_remaining = max(total_min - total_done, 0.0)
    completion_pct = (total_done / total_min * 100) if total_min else 0.0
    general_primary_discrepancy = (
        round((general["remaining_su"] - max(primary_min - primary_done, 0.0)), 2)
        if general is not None else None
    )

    # What the graph-based requirement engine sees.
    rep = compute_remaining(
        program=raw["program"],
        completed_for_eligibility=set(raw["completed_for_eligibility"]),
        in_progress=set(raw.get("in_progress", [])),
        admit_term=raw["admit_term"],
    )
    engine = {
        "required_left": sorted(rep.required_left),
        "required_credits_left": rep.required_credits_left,
        "core_left_count": len(rep.core_left),
        "area_left_count": len(rep.area_left),
    }

    # Required courses are the one section both sources can compare directly.
    off_required = next((s for s in sections if s["section"] == "REQUIRED COURSES"), None)
    required_discrepancy = None
    if off_required is not None:
        required_discrepancy = round(rep.required_credits_left - off_required["remaining_su"], 2)

    print("=== Official Degree Evaluation: per-section SU (ground truth) ===")
    print(f"{'section':<22}{'min':>7}{'done':>8}{'remaining':>11}")
    for s in sections:
        print(f"{s['section']:<22}{s['min_su']:>7.1f}{s['completed_su']:>8.1f}{s['remaining_su']:>11.1f}")
    print(f"\nAuthoritative overall total source: {total_source}")
    print(f"Overall completion (SU): {total_done:.1f}/{total_min:.1f} = {completion_pct:.1f}%")
    print(
        "Primary non-overlapping section sum: "
        f"{primary_done:.1f}/{primary_min:.1f} SU "
        f"(remaining {max(primary_min - primary_done, 0.0):.1f})"
    )

    print("\n=== Engine (compute_remaining) view ===")
    print(f"  required_left           : {engine['required_left']}")
    print(f"  required_credits_left   : {engine['required_credits_left']}")
    print(f"  core electives left     : {engine['core_left_count']} courses")
    print(f"  area electives left     : {engine['area_left_count']} courses")

    if off_required is not None:
        print("\n=== REQUIRED bucket agreement ===")
        print(f"  official remaining SU : {off_required['remaining_su']:.1f}")
        print(f"  engine remaining SU   : {engine['required_credits_left']:.1f}")
        print(f"  discrepancy           : {required_discrepancy:+.1f} SU "
              + ("(MATCH)" if required_discrepancy == 0 else "(DIVERGES — see error analysis)"))

    out = {
        "official_sections": sections,
        "official_total": {
            "source": total_source,
            "min_su": total_min,
            "completed_su": total_done,
            "remaining_su": overall_remaining,
            "completion_pct": round(completion_pct, 1),
        },
        "primary_section_sum": {
            "sections": sorted(PRIMARY_SU_SECTIONS),
            "min_su": primary_min,
            "completed_su": primary_done,
            "remaining_su": max(primary_min - primary_done, 0.0),
            "general_remaining_discrepancy_su": general_primary_discrepancy,
        },
        "engine": engine,
        "required_bucket_discrepancy_su": required_discrepancy,
        "notes": [
            "Section rows overlap and are not summed into the overall total.",
            "The required-bucket comparison is a credit-gap diagnostic, not a precision/recall score.",
        ],
    }
    (EVAL_DIR / "results_requirements.json").write_text(json.dumps(out, indent=2))
    print("\nSaved -> eval/results_requirements.json")


if __name__ == "__main__":
    main()
