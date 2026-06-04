"""
Time-conflict detection evaluation for SuSchedule-r.

Two parts:

1. LABELED PAIRS -- a hand-built set of meeting pairs with a known ground-truth
   answer (overlap / no overlap), covering the tricky cases: same-day overlap,
   back-to-back (touching but not overlapping), different days, partial overlap,
   multi-day meetings. We measure accuracy / precision / recall of
   `meetings_overlap`.

2. REAL COURSE BUNDLES -- add each lecture's required recitations/labs, run
   `build_timetable` for a real term, and verify the returned schedule has no
   overlaps. The cases include bundles we know should fit and bundles we know
   should not.

Runs fully locally (no OpenAI). Usage:
    python -m eval.run_conflicts
"""
from __future__ import annotations

import json
from pathlib import Path

from scheduler.catalog import Catalog, corequisite_codes
from scheduler.timetable import (
    meetings_overlap, conflict_report, build_timetable,
    load_offerings, resolve_schedule_term,
)

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"


def M(days, start, end):
    """Helper: build a meeting dict (minutes since midnight)."""
    return {"days": days, "start": start, "end": end}


# (name, meeting_a, meeting_b, expected_overlap)
LABELED_PAIRS = [
    ("same slot exactly",          M([0], 520, 570), M([0], 520, 570), True),
    ("partial overlap same day",   M([0], 520, 620), M([0], 580, 680), True),
    ("contained within",           M([2], 600, 800), M([2], 640, 700), True),
    ("back-to-back touching",      M([0], 520, 580), M([0], 580, 640), False),
    ("same time different day",    M([0], 520, 570), M([1], 520, 570), False),
    ("overlap on one shared day",  M([0, 2], 520, 570), M([2, 4], 540, 600), True),
    ("shared day no time overlap", M([0, 2], 520, 570), M([2], 600, 650), False),
    ("disjoint days multi",        M([0, 2], 520, 570), M([1, 3], 520, 570), False),
    ("one minute overlap",         M([3], 600, 660), M([3], 659, 720), True),
    ("adjacent one minute gap",    M([3], 600, 659), M([3], 660, 720), False),
    ("evening overlap",            M([4], 1120, 1170), M([4], 1100, 1130), True),
    ("empty days never overlaps",  M([], 520, 570), M([0], 520, 570), False),
]


def eval_labeled_pairs() -> dict:
    tp = fp = tn = fn = 0
    wrong = []
    for name, a, b, expected in LABELED_PAIRS:
        got = meetings_overlap(a, b)
        if got and expected:
            tp += 1
        elif got and not expected:
            fp += 1; wrong.append(name)
        elif not got and not expected:
            tn += 1
        else:
            fn += 1; wrong.append(name)
    n = len(LABELED_PAIRS)
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    return {
        "n_pairs": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": round(acc, 4), "precision": round(prec, 4), "recall": round(rec, 4),
        "wrong": wrong,
    }


# These start as lecture choices. The test adds their required labs and
# recitations before asking the timetable builder to place them.
REAL_BUNDLES = [
    {"name": "intro courses", "plan": ["CS 201", "MATH 101"], "expected_feasible": True},
    {"name": "known constrained bundle", "plan": ["CS 300", "CS 301", "MATH 201"], "expected_feasible": False},
    {"name": "machine learning bundle", "plan": ["CS 201", "CS 204", "CS 412"], "expected_feasible": True},
    {
        "name": "software systems lectures fit but required sessions do not",
        "plan": ["CS 306", "CS 307", "CS 308"],
        "expected_feasible": False,
    },
    {
        "name": "lecture fits but recitation conflicts",
        "plan": ["CS 412", "CS 302"],
        "expected_feasible": False,
    },
    {
        "name": "multiple required labs and recitations fit",
        "plan": ["CS 302", "CS 308"],
        "expected_feasible": True,
    },
]


def expand_with_corequisites(catalog: Catalog, plan: list[str]) -> tuple[list[str], list[str]]:
    expanded: list[str] = []
    added: list[str] = []
    for code in plan:
        if code not in expanded:
            expanded.append(code)
        for coreq in corequisite_codes(catalog, code):
            if coreq not in expanded:
                expanded.append(coreq)
                added.append(coreq)
    return expanded, added


def eval_soundness() -> dict:
    offerings = load_offerings()
    catalog = Catalog.load()
    term, is_proxy = resolve_schedule_term(offerings, "202601")
    results = []
    unsound = 0
    wrong_feasibility = 0
    incomplete_coreq_picks = 0
    for bundle in REAL_BUNDLES:
        plan = bundle["plan"]
        expanded, added_coreqs = expand_with_corequisites(catalog, plan)
        res = build_timetable(expanded, term, offerings)
        picks = []
        for p in res.picks:
            picks.append({"label": p.get("label", p.get("code", "?")), "meetings": p.get("meetings", [])})
        clashes = conflict_report(picks)
        sound = len(clashes) == 0  # a feasible result must be conflict-free
        if res.ok and not sound:
            unsound += 1
        feasibility_correct = res.ok == bundle["expected_feasible"]
        if not feasibility_correct:
            wrong_feasibility += 1
        picked_codes = {p.get("code") for p in res.picks}
        coreqs_present_when_feasible = not res.ok or set(added_coreqs).issubset(picked_codes)
        if not coreqs_present_when_feasible:
            incomplete_coreq_picks += 1
        results.append({
            "name": bundle["name"],
            "plan": plan,
            "expanded_plan": expanded,
            "auto_added_corequisites": added_coreqs,
            "schedule_term": term,
            "expected_feasible": bundle["expected_feasible"],
            "feasible": res.ok,
            "feasibility_correct": feasibility_correct,
            "corequisites_present_when_feasible": coreqs_present_when_feasible,
            "picks": [p["label"] for p in picks],
            "residual_conflicts": len(clashes),
            "missing": res.missing,
        })
    return {
        "schedule_term": term,
        "is_proxy": is_proxy,
        "n_bundles": len(REAL_BUNDLES),
        "unsound_results": unsound,
        "wrong_feasibility_results": wrong_feasibility,
        "incomplete_corequisite_picks": incomplete_coreq_picks,
        "feasibility_accuracy": round((len(REAL_BUNDLES) - wrong_feasibility) / len(REAL_BUNDLES), 4),
        "bundles": results,
    }


def main() -> None:
    pairs = eval_labeled_pairs()
    print("=== Conflict detector on labeled meeting pairs ===")
    print(f"  pairs={pairs['n_pairs']}  accuracy={pairs['accuracy']:.3f}  "
          f"precision={pairs['precision']:.3f}  recall={pairs['recall']:.3f}")
    if pairs["wrong"]:
        print(f"  MISCLASSIFIED: {pairs['wrong']}")
    else:
        print("  all pairs classified correctly")

    sound = eval_soundness()
    print("\n=== build_timetable soundness on real bundles ===")
    print(f"  schedule_term={sound['schedule_term']} (proxy={sound['is_proxy']})  "
          f"bundles={sound['n_bundles']}  unsound={sound['unsound_results']}  "
          f"feasibility_accuracy={sound['feasibility_accuracy']:.3f}  "
          f"incomplete_coreqs={sound['incomplete_corequisite_picks']}")
    for b in sound["bundles"]:
        flag = "OK " if (
            b["feasibility_correct"]
            and b["corequisites_present_when_feasible"]
            and (not b["feasible"] or b["residual_conflicts"] == 0)
        ) else "BAD"
        print(f"  [{flag}] {b['name']}: {b['plan']} + {b['auto_added_corequisites']} -> feasible={b['feasible']} "
              f"picks={len(b['picks'])} residual_conflicts={b['residual_conflicts']}"
              + (f" missing={b['missing']}" if b['missing'] else ""))

    out = {"labeled_pairs": pairs, "soundness": sound}
    (EVAL_DIR / "results_conflicts.json").write_text(json.dumps(out, indent=2))
    print("\nSaved -> eval/results_conflicts.json")


if __name__ == "__main__":
    main()
