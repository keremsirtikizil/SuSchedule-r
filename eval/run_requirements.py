"""
Degree-requirement evaluation for SuSchedule-r.

Ground truth = the student's OFFICIAL Banner "Degree Evaluation" HTML, which
reports, per requirement section, the minimum SU credits required and the SU
credits already completed. We compare that against:

  (a) the HTML parser's own extraction (sanity: does parse_degree_evaluation
      recover the section credit table the agent relies on?), and
  (b) the graph-based requirement engine `compute_remaining`, which the agent
      uses to decide what is still needed.

We report per-section remaining SU from the official audit and the engine's
required-bucket remaining credits, and surface any disagreement honestly.

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


def main() -> None:
    raw = parse_degree_evaluation(DEGREE_HTML)
    sr = raw["section_requirements"]

    # Official per-section remaining SU.
    sections = []
    tot_min = tot_done = 0.0
    for name, d in sr.items():
        if not isinstance(d, dict):
            continue
        mn = float(d.get("min_su", 0) or 0)
        dn = float(d.get("completed_su", 0) or 0)
        rem = max(mn - dn, 0.0)
        tot_min += mn
        tot_done += dn
        sections.append({"section": name, "min_su": mn, "completed_su": dn, "remaining_su": rem})

    overall_remaining = max(tot_min - tot_done, 0.0)
    completion_pct = (tot_done / tot_min * 100) if tot_min else 0.0

    # Engine view.
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

    # Headline comparison on the cleanest-mapping bucket.
    off_required = next((s for s in sections if s["section"] == "REQUIRED COURSES"), None)
    required_discrepancy = None
    if off_required is not None:
        required_discrepancy = round(rep.required_credits_left - off_required["remaining_su"], 2)

    print("=== Official Degree Evaluation: per-section SU (ground truth) ===")
    print(f"{'section':<22}{'min':>7}{'done':>8}{'remaining':>11}")
    for s in sections:
        print(f"{s['section']:<22}{s['min_su']:>7.1f}{s['completed_su']:>8.1f}{s['remaining_su']:>11.1f}")
    print(f"{'TOTAL':<22}{tot_min:>7.1f}{tot_done:>8.1f}{overall_remaining:>11.1f}")
    print(f"\nOverall completion (SU): {tot_done:.1f}/{tot_min:.1f} = {completion_pct:.1f}%")

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
        "official_total": {"min_su": tot_min, "completed_su": tot_done,
                           "remaining_su": overall_remaining,
                           "completion_pct": round(completion_pct, 1)},
        "engine": engine,
        "required_bucket_discrepancy_su": required_discrepancy,
    }
    (EVAL_DIR / "results_requirements.json").write_text(json.dumps(out, indent=2))
    print("\nSaved -> eval/results_requirements.json")


if __name__ == "__main__":
    main()
