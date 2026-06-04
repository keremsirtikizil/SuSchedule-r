"""
Eligibility / prerequisite-engine evaluation for SuSchedule-r.

Adds regression coverage for a bug the other evals did not exercise: a course
whose *hard* (prior-term) prerequisite is currently IN PROGRESS must count as
eligible for the future target term, because the in-progress course will have
finished by then. Before the fix, in-progress courses only satisfied prereqs
explicitly tagged "(can be taken concurrently)", so e.g. a student taking the
prereq this term was wrongly told the next course was still locked.

Method: pull from the real degree graph every course whose prerequisite is a
single mandatory (non-concurrent) course, then assert three states per course:

  - nothing done            -> NOT eligible      (control: the gate works)
  - prereq completed         -> eligible          (control)
  - prereq IN PROGRESS        -> eligible          (the regression case)

It also checks validate_plan agrees: a one-course plan whose only prereq is in
progress raises no 'prereq_unsatisfied' violation.

Runs fully locally (no OpenAI). Usage:
    python -m eval.run_eligibility
"""
from __future__ import annotations

import json
from pathlib import Path

from scheduler.eligibility import (
    Student,
    eligible_courses,
    load_graph,
    validate_plan,
)

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"
MAX_CASES = 12


def bare_hard_prereq(expr) -> str | None:
    """Return the course if *expr* is a single mandatory (non-concurrent) atom."""
    if isinstance(expr, dict) and "course" in expr and not expr.get("concurrent"):
        return expr["course"]
    return None


def is_eligible(graph, course: str, *, completed=None, in_progress=None) -> bool:
    student = Student(completed=set(completed or []), in_progress=set(in_progress or []))
    return course in eligible_courses(graph, student, candidate_pool=[course])


def has_prereq_violation(graph, course: str, *, in_progress=None) -> bool:
    student = Student(in_progress=set(in_progress or []))
    report = validate_plan(graph, student, [course])
    return any(v.kind == "prereq_unsatisfied" for v in report.violations)


def main() -> None:
    graph = load_graph(program="BSCS-DM", admit_term="202601")

    cases = []
    for node, data in graph.nodes(data=True):
        if not data.get("in_catalog"):
            continue
        prereq = bare_hard_prereq(data.get("prereq_expr"))
        if prereq and graph.has_node(prereq) and prereq != node:
            cases.append((node, prereq))
        if len(cases) >= MAX_CASES:
            break

    rows = []
    passed = 0
    for course, prereq in cases:
        none_elig = is_eligible(graph, course)
        done_elig = is_eligible(graph, course, completed={prereq})
        wip_elig = is_eligible(graph, course, in_progress={prereq})
        wip_no_violation = not has_prereq_violation(graph, course, in_progress={prereq})

        # Expected: gate closed when nothing done; open when prereq done OR in progress.
        ok = (not none_elig) and done_elig and wip_elig and wip_no_violation
        passed += int(ok)
        rows.append({
            "course": course,
            "prereq": prereq,
            "eligible_when_nothing_done": none_elig,     # expect False
            "eligible_when_prereq_completed": done_elig,  # expect True
            "eligible_when_prereq_in_progress": wip_elig,  # expect True (regression)
            "no_prereq_violation_when_in_progress": wip_no_violation,  # expect True
            "pass": ok,
        })

    n = len(rows)
    print(f"=== In-progress prereq eligibility ({n} graph-derived cases) ===")
    print(f"{'course':<12}{'prereq':<12}{'none':>6}{'done':>6}{'wip':>6}{'plan_ok':>9}{'pass':>6}")
    for r in rows:
        print(f"{r['course']:<12}{r['prereq']:<12}"
              f"{str(r['eligible_when_nothing_done']):>6}"
              f"{str(r['eligible_when_prereq_completed']):>6}"
              f"{str(r['eligible_when_prereq_in_progress']):>6}"
              f"{str(r['no_prereq_violation_when_in_progress']):>9}"
              f"{('PASS' if r['pass'] else 'FAIL'):>6}")
    accuracy = (passed / n) if n else 0.0
    print(f"\npassed {passed}/{n}  (accuracy={accuracy:.3f})")
    if passed != n:
        print("  REGRESSION: in-progress courses are not satisfying hard prereqs.")

    out = {
        "n_cases": n,
        "passed": passed,
        "accuracy": round(accuracy, 4),
        "cases": rows,
    }
    (EVAL_DIR / "results_eligibility.json").write_text(json.dumps(out, indent=2))
    print("\nSaved -> eval/results_eligibility.json")


if __name__ == "__main__":
    main()
