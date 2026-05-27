"""
Hard-constraint / eligibility module.

This is the symbolic core of SUcheduler. It owns every rule we are NOT
willing to trust the LLM with:

  * Prerequisites (the boolean expression tree on each course).
  * Already-completed-course de-duplication.
  * Credit-load limit per term.
  * Co-requisite pairing (e.g. CS 201 must be paired with CS 201R).
  * Target-term offering window (handled separately by the offerings filter,
    but consulted here so the validator can flag it).

The module exposes two surfaces:

  1. ``eligible_courses(graph, student, options)``
        -> the set of course codes the student is allowed to take
           in the target term.

  2. ``validate_plan(graph, student, plan, options)``
        -> a structured report with every violation found in a proposed
           term plan.

There is no LLM, no embedding, no probability anywhere here. Both functions
are deterministic and trivially testable.

Run ``python -m scheduler.eligibility`` for a smoke test against a synthetic
CS-junior profile.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


# --------------------------------------------------------------------------- #
# Student / plan data classes
# --------------------------------------------------------------------------- #


@dataclass
class Student:
    """The bare minimum state a hard-constraint check needs."""

    program: str = "BSCS-DM"
    admit_term: str = "202101"
    completed: set[str] = field(default_factory=set)
    in_progress: set[str] = field(default_factory=set)
    cumulative_credits: float = 0.0
    intended_minor: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Student":
        return cls(
            program=d.get("program", "BSCS-DM"),
            admit_term=str(d.get("admit_term", "202101")),
            completed=set(d.get("completed", [])),
            in_progress=set(d.get("in_progress", [])),
            cumulative_credits=float(d.get("cumulative_credits", 0.0)),
            intended_minor=d.get("intended_minor"),
        )


@dataclass
class EligibilityOptions:
    """Tunable hard-rule knobs."""

    max_term_credits: float = 21.0          # SU undergrad standard ceiling
    min_term_credits: float = 0.0           # 0 = no floor (let the planner pick)
    # If True, an "(can be taken concurrently)" prereq is considered satisfied
    # when the prereq course is included in the *current* plan or is already
    # in ``Student.in_progress``.
    allow_concurrent_prereqs: bool = True


# --------------------------------------------------------------------------- #
# Prereq evaluation
# --------------------------------------------------------------------------- #


def _expr_satisfied(expr: dict | None, completed: set[str], concurrent_ok: set[str]) -> bool:
    """Walk a parsed prereq expression tree and decide if it is satisfied.

    ``completed`` is the strict set of finished courses. ``concurrent_ok`` is
    a wider set we allow for atoms tagged ``concurrent=True``.
    """
    if expr is None:
        return True
    if "course" in expr:
        if expr.get("concurrent"):
            return expr["course"] in completed or expr["course"] in concurrent_ok
        return expr["course"] in completed
    op = expr["op"]
    operands = expr["operands"]
    if op == "and":
        return all(_expr_satisfied(o, completed, concurrent_ok) for o in operands)
    if op == "or":
        return any(_expr_satisfied(o, completed, concurrent_ok) for o in operands)
    raise ValueError(f"Unknown op {op!r}")


def prereqs_satisfied(graph: nx.DiGraph, course: str, completed: set[str],
                      concurrent_ok: set[str] | None = None) -> bool:
    """Public helper: does the student satisfy ``course``'s prereq expression?"""
    if not graph.has_node(course):
        return False
    expr = graph.nodes[course].get("prereq_expr")
    return _expr_satisfied(expr, completed, concurrent_ok or set())


# --------------------------------------------------------------------------- #
# Eligibility set
# --------------------------------------------------------------------------- #


def _credits_of(graph: nx.DiGraph, course: str) -> float:
    """Return SU credits for a course, defaulting to 0.0 for unknown nodes."""
    su = graph.nodes.get(course, {}).get("su_credit")
    try:
        return float(su) if su is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _coreq_codes(graph: nx.DiGraph, course: str) -> list[str]:
    """Extract bare course codes from the raw ``coreq_text`` string.

    The course-detail pages list co-reqs as comma- or space-separated codes,
    e.g. ``"CS 201R"`` or ``"CS 301R, CS 301L"``. We just pull anything that
    looks like ``<SUBJ> <NUM>``.
    """
    import re
    txt = graph.nodes.get(course, {}).get("coreq_text", "") or ""
    return [
        f"{m.group(1).upper()} {m.group(2).upper()}"
        for m in re.finditer(r"\b([A-Z]{2,5})\s*(\d{2,5}[A-Z]?)\b", txt)
    ]


def eligible_courses(
    graph: nx.DiGraph,
    student: Student,
    options: EligibilityOptions | None = None,
    candidate_pool: Iterable[str] | None = None,
) -> set[str]:
    """Return every course the student is allowed to take next term.

    ``candidate_pool`` lets the caller restrict the universe (e.g. only the
    courses being offered next term). When omitted we consider every node in
    the catalog.
    """
    options = options or EligibilityOptions()
    completed = set(student.completed)
    in_progress = set(student.in_progress)

    if candidate_pool is None:
        candidates = [n for n, d in graph.nodes(data=True) if d.get("in_catalog")]
    else:
        candidates = [c for c in candidate_pool if graph.has_node(c)]

    out: set[str] = set()
    for code in candidates:
        if code in completed:
            continue
        # Concurrent-prereq satisfaction looks at the in-progress set only;
        # for "plan-level" concurrency the caller uses validate_plan().
        concurrent_ok = in_progress if options.allow_concurrent_prereqs else set()
        if prereqs_satisfied(graph, code, completed, concurrent_ok):
            out.add(code)
    return out


# --------------------------------------------------------------------------- #
# Plan validator
# --------------------------------------------------------------------------- #


@dataclass
class Violation:
    code: str | None      # course code the violation is about (None = whole-plan)
    kind: str             # short tag, e.g. "prereq_unsatisfied"
    message: str

    def to_dict(self) -> dict:
        return {"code": self.code, "kind": self.kind, "message": self.message}


@dataclass
class PlanReport:
    ok: bool
    total_credits: float
    violations: list[Violation]
    auto_added_coreqs: list[str]

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "total_credits": self.total_credits,
            "violations": [v.to_dict() for v in self.violations],
            "auto_added_coreqs": self.auto_added_coreqs,
        }


def validate_plan(
    graph: nx.DiGraph,
    student: Student,
    plan: Iterable[str],
    options: EligibilityOptions | None = None,
) -> PlanReport:
    """Validate a proposed term plan.

    Rules enforced:
      * No duplicates.
      * No already-completed courses.
      * Every course exists in the catalog (warn if external).
      * Prereqs satisfied, including same-term concurrent prereqs when the
        plan itself contains the concurrent course.
      * Co-requisites auto-added; reported in ``auto_added_coreqs``.
      * ``options.min_term_credits`` <= sum of SU credits <= ``options.max_term_credits``.
    """
    options = options or EligibilityOptions()
    violations: list[Violation] = []
    plan_list = list(plan)

    # Duplicate detection -----------------------------------------------------
    dupes = {c for c in plan_list if plan_list.count(c) > 1}
    for c in sorted(dupes):
        violations.append(Violation(c, "duplicate", f"{c} listed more than once"))

    plan_set = set(plan_list)

    # Already-completed ------------------------------------------------------
    for c in sorted(plan_set & student.completed):
        violations.append(Violation(c, "already_completed", f"{c} already in transcript"))

    # Unknown courses --------------------------------------------------------
    for c in sorted(plan_set):
        if not graph.has_node(c):
            violations.append(Violation(c, "unknown_course", f"{c} not in catalog graph"))
            continue
        if not graph.nodes[c].get("in_catalog"):
            violations.append(
                Violation(c, "external_course",
                          f"{c} is referenced only as an external prereq")
            )

    # Co-requisite auto-add --------------------------------------------------
    auto_added: list[str] = []
    extended = set(plan_set)
    for c in plan_set:
        for coreq in _coreq_codes(graph, c):
            if coreq not in extended and coreq not in student.completed:
                extended.add(coreq)
                auto_added.append(coreq)

    # Prerequisite check (allow concurrent within the extended plan) ---------
    completed = set(student.completed)
    concurrent_ok = extended | student.in_progress
    for c in sorted(plan_set):
        if not graph.has_node(c):
            continue
        expr = graph.nodes[c].get("prereq_expr")
        if not _expr_satisfied(expr, completed, concurrent_ok):
            violations.append(
                Violation(
                    c,
                    "prereq_unsatisfied",
                    f"{c}: prereq {graph.nodes[c].get('prereq_text', '')!r} not satisfied",
                )
            )

    # Credit-load check ------------------------------------------------------
    total = sum(_credits_of(graph, c) for c in extended)
    if total > options.max_term_credits:
        violations.append(
            Violation(
                None,
                "credit_limit_exceeded",
                f"Total credits {total} exceeds max {options.max_term_credits}",
            )
        )
    if total < options.min_term_credits:
        violations.append(
            Violation(
                None,
                "credit_limit_below_min",
                f"Total credits {total} below min {options.min_term_credits}",
            )
        )

    return PlanReport(
        ok=not violations,
        total_credits=total,
        violations=violations,
        auto_added_coreqs=sorted(auto_added),
    )


# --------------------------------------------------------------------------- #
# Convenience loader + smoke test
# --------------------------------------------------------------------------- #


_COHORT_TERMS = ("202201", "202301", "202401", "202601")

def _cohort_for_admit(admit_term: str) -> str:
    """Map any admit term to the nearest fall cohort catalog.

    Spring admits (XX02) follow the preceding fall's catalog (XX01).
    Returns the closest available cohort term on or before admit_term.
    """
    # If it's a spring term, use the same year's fall.
    if admit_term.endswith("02"):
        admit_term = admit_term[:-2] + "01"
    # Find the latest cohort term that is <= admit_term.
    best = _COHORT_TERMS[0]
    for ct in _COHORT_TERMS:
        if ct <= admit_term:
            best = ct
    return best


def load_section_graph(
    program: str,
    cohort_term: str,
    section: str,
) -> nx.DiGraph:
    """Load one (program, cohort_term, section) graph from degree_graphs/."""
    slug = section.replace(" ", "_")
    path = DATA_DIR / "degree_graphs" / program / cohort_term / f"{slug}.gpickle"
    if not path.exists():
        raise SystemExit(
            f"Missing {path} -- run 'python -m scraper.build_degree_graphs' first."
        )
    with path.open("rb") as f:
        return pickle.load(f)


def load_graph(
    program: str = "BSCS-DM",
    admit_term: str = "202601",
    sections: tuple[str, ...] = (
        "Required", "Core Elective", "Area Elective", "Free Elective"
    ),
) -> nx.DiGraph:
    """Load and merge all section graphs for a given program and admit term.

    The admit_term is mapped to the nearest cohort catalog term so that
    each student sees the degree requirements that applied when they enrolled.
    """
    cohort = _cohort_for_admit(admit_term)
    merged: nx.DiGraph | None = None
    for section in sections:
        try:
            G = load_section_graph(program, cohort, section)
        except SystemExit:
            continue
        if merged is None:
            merged = G
        else:
            for node, attrs in G.nodes(data=True):
                if not merged.has_node(node):
                    merged.add_node(node, **attrs)
                else:
                    # Promote in_slice if the node is in-slice in any section.
                    if attrs.get("in_slice"):
                        merged.nodes[node]["in_slice"] = True
            for u, v, attrs in G.edges(data=True):
                if not merged.has_edge(u, v):
                    merged.add_edge(u, v, **attrs)
    if merged is None:
        raise SystemExit(
            f"No graphs found for {program}/{cohort} -- "
            "run 'python -m scraper.build_degree_graphs' first."
        )
    return merged


def _smoke_test() -> None:
    """Quick sanity check against a synthetic CS-junior profile."""
    G = load_graph(program="BSCS-DM", admit_term="202601")
    # A student who has finished freshman + sophomore CS core + math.
    student = Student(
        completed={
            # University courses
            "IF 100", "MATH 101", "MATH 102", "NS 101", "NS 102",
            "SPS 101", "SPS 102", "AL 102", "TLL 101", "TLL 102",
            "HIST 191", "HIST 192", "HUM 201",
            # Sophomore math + science
            "MATH 203", "MATH 204", "MATH 212", "PHYS 113",
            # Required CS so far
            "CS 201", "CS 204", "CS 300",
        },
        cumulative_credits=64.0,
    )
    elig = eligible_courses(G, student)
    print(f"Eligible (out of catalog): {len(elig)} courses")
    cs_only = sorted(c for c in elig if c.startswith("CS "))
    print(f"CS-prefixed eligible:      {cs_only}")
    print()

    # Validate a sensible plan.
    plan = ["CS 301", "CS 306", "CS 308", "MATH 306", "HUM 202"]
    report = validate_plan(G, student, plan)
    print(f"Plan {plan}")
    print(f"  ok = {report.ok}, total_credits = {report.total_credits}")
    if report.auto_added_coreqs:
        print(f"  auto-added co-reqs: {report.auto_added_coreqs}")
    for v in report.violations:
        print(f"  - [{v.kind}] {v.message}")
    print()

    # Validate a broken plan: includes a course whose prereq is missing
    # (CS 401 needs CS 303, which the student hasn't taken).
    bad = ["CS 401", "CS 408", "CS 999"]
    rep = validate_plan(G, student, bad)
    print(f"Broken plan {bad}")
    print(f"  ok = {rep.ok}, total_credits = {rep.total_credits}")
    for v in rep.violations:
        print(f"  - [{v.kind}] {v.message}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", default="BSCS-DM")
    ap.add_argument("--term", default="202601")
    ap.add_argument("--student", help="Path to a student profile JSON")
    ap.add_argument("--plan", nargs="*", help="Course codes to validate")
    ap.add_argument("--smoke", action="store_true", help="Run built-in synthetic example")
    args = ap.parse_args()

    if args.smoke or not args.student:
        _smoke_test()
        return

    G = load_graph(args.program, args.term)
    student = Student.from_dict(json.loads(Path(args.student).read_text()))
    if args.plan:
        report = validate_plan(G, student, args.plan)
        print(json.dumps(report.to_dict(), indent=2))
    else:
        elig = sorted(eligible_courses(G, student))
        print(json.dumps({"eligible_count": len(elig), "eligible": elig}, indent=2))


if __name__ == "__main__":
    main()
