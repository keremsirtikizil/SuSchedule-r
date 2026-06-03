"""
Time-conflict detection evaluation for SuSchedule-r.

Two parts:

1. LABELED PAIRS -- a hand-built set of meeting pairs with a known ground-truth
   answer (overlap / no overlap), covering the tricky cases: same-day overlap,
   back-to-back (touching but not overlapping), different days, partial overlap,
   multi-day meetings. We measure accuracy / precision / recall of
   `meetings_overlap`.

2. SOUNDNESS ON REAL DATA -- run `build_timetable` on several real course
   bundles for an actual term and assert the returned assignment is genuinely
   conflict-free (`conflict_report` on the chosen picks must be empty). A
   scheduler that returns overlapping picks would be unsound.

Runs fully locally (no OpenAI). Usage:
    python -m eval.run_conflicts
"""
from __future__ import annotations

import json
from pathlib import Path

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


# Real bundles to schedule (a feasible one + a known clashing pair).
REAL_BUNDLES = [
    ["CS 201", "MATH 101"],
    ["CS 300", "CS 301", "MATH 201"],
    ["CS 201", "CS 204", "CS 412"],
    ["CS 306", "CS 307", "CS 308"],
]


def eval_soundness() -> dict:
    offerings = load_offerings()
    term, is_proxy = resolve_schedule_term(offerings, "202601")
    results = []
    unsound = 0
    for plan in REAL_BUNDLES:
        res = build_timetable(plan, term, offerings)
        picks = []
        for p in res.picks:
            picks.append({"label": p.get("label", p.get("code", "?")), "meetings": p.get("meetings", [])})
        clashes = conflict_report(picks)
        sound = len(clashes) == 0  # a feasible result must be conflict-free
        if res.ok and not sound:
            unsound += 1
        results.append({
            "plan": plan, "schedule_term": term, "feasible": res.ok,
            "picks": [p["label"] for p in picks],
            "residual_conflicts": len(clashes),
            "missing": res.missing,
        })
    return {"schedule_term": term, "is_proxy": is_proxy,
            "n_bundles": len(REAL_BUNDLES), "unsound_results": unsound,
            "bundles": results}


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
          f"bundles={sound['n_bundles']}  unsound={sound['unsound_results']}")
    for b in sound["bundles"]:
        flag = "OK " if (not b["feasible"] or b["residual_conflicts"] == 0) else "BAD"
        print(f"  [{flag}] {b['plan']} -> feasible={b['feasible']} "
              f"picks={len(b['picks'])} residual_conflicts={b['residual_conflicts']}"
              + (f" missing={b['missing']}" if b['missing'] else ""))

    out = {"labeled_pairs": pairs, "soundness": sound}
    (EVAL_DIR / "results_conflicts.json").write_text(json.dumps(out, indent=2))
    print("\nSaved -> eval/results_conflicts.json")


if __name__ == "__main__":
    main()
