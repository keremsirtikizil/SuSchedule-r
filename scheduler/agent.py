"""
SuSchedule-r — Phase A planner agent.

Entry point: ``plan(request: PlannerRequest) -> TermPlan``

Pipeline (Stages 1–8):
  1  Profile        transcript JSON/PDF → Student
  2  Requirements   degree graphs → what's still needed
  3  Candidate pool eligibility + offerings filter
  4  RAG            intent parse → retrieval → cross-encoder re-rank
  5  LLM propose    gpt-4o → PlannerOutput (strict JSON)
  6  Validate       eligibility.validate_plan
  7  Repair loop    (if violations) gpt-4o → RepairOutput → re-validate
  8  Timetable      timetable.find_feasible_schedule (best-effort)

CLI:
  python -m scheduler.agent \\
      --transcript data/transcript_cagan.json \\
      --request "I want ML and a database course" \\
      --term 202601
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

from scheduler.eligibility import Student, EligibilityOptions, eligible_courses, validate_plan
from scheduler.offerings import load_offerings, likely_offered_in
from scheduler.requirements import compute_remaining
from scheduler.retriever import Retriever, RetrieverFilter
from scheduler.schemas import PlannerRequest, TermPlan, IntentQueries, PlannerOutput, RepairOutput
from scheduler.prompts import (
    INTENT_PARSE_SYSTEM, format_intent_user,
    PLANNER_SYSTEM, format_planner_user,
    REPAIR_SYSTEM, format_repair_user,
    term_label,
)
from scheduler import llm_client


# --------------------------------------------------------------------------- #
# Lazy singletons — loaded once per process
# --------------------------------------------------------------------------- #

_retriever: Retriever | None = None
_graph_cache: dict[str, nx.DiGraph] = {}


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        print("[Agent] Loading retriever (bi-encoder + cross-encoder)…")
        _retriever = Retriever.load(device="cpu")
    return _retriever


def _get_graph(program: str = "BSCS-DM") -> nx.DiGraph:
    """Load the full SU prereq graph (preferred) or fall back to program-specific."""
    if program not in _graph_cache:
        full_path = DATA_DIR / "SU_full_graph.gpickle"
        prog_path = DATA_DIR / f"{program}_202601_graph.gpickle"
        path = full_path if full_path.exists() else prog_path
        if not path.exists():
            raise FileNotFoundError(
                f"Prereq graph not found at {path}. "
                "Run scraper/build_graph.py --full first."
            )
        print(f"[Agent] Loading graph from {path.name}…")
        with path.open("rb") as f:
            _graph_cache[program] = pickle.load(f)
    return _graph_cache[program]


# --------------------------------------------------------------------------- #
# Stage helpers
# --------------------------------------------------------------------------- #

def _load_student(request: PlannerRequest) -> tuple[Student, dict]:
    """Stage 1: Load and return (Student dataclass, raw transcript dict)."""
    if request.transcript_json:
        raw = json.loads(Path(request.transcript_json).read_text(encoding="utf-8"))
    elif request.transcript_pdf:
        from scheduler.transcript import parse_transcript
        raw = parse_transcript(str(request.transcript_pdf))
    else:
        raise ValueError("PlannerRequest must set transcript_json or transcript_pdf.")

    student = Student(
        program=raw.get("program", "BSCS-DM"),
        admit_term=str(raw.get("admit_term", "202101")),
        completed=set(raw.get("completed_for_eligibility", raw.get("completed", []))),
        in_progress=set(raw.get("in_progress", [])),
        cumulative_credits=float(raw.get("cumulative_credits", 0.0)),
    )
    return student, raw


def _build_candidate_pool(
    graph: nx.DiGraph,
    student: Student,
    request: PlannerRequest,
    required_left: list[str],
    offerings: dict,
) -> tuple[set[str], set[str]]:
    """Stage 3: Returns (eligible_pool, required_injected_set).

    eligible_pool  = courses the student CAN take AND are likely offered.
    required_injected = subset of eligible_pool that are still-needed required
                        courses (so the LLM prompt can flag them specially).
    """
    offered = likely_offered_in(offerings, request.target_term, same_season_only=True)
    eligible = eligible_courses(
        graph, student,
        options=EligibilityOptions(),
        candidate_pool=offered,
    )

    # Always inject required-still-needed if they're eligible and offered.
    required_injected: set[str] = set()
    if request.include_required:
        for code in required_left:
            if code in eligible:
                required_injected.add(code)
            elif code in offered and graph.has_node(code):
                # Offered but prereq not met — skip (validate_plan will catch it)
                pass

    return eligible, required_injected


def _parse_intents(user_request: str, model: str) -> list[str]:
    """Stage 4a: Use the configured model to split request into retrieval queries."""
    if not user_request.strip():
        return ["required courses for graduation"]

    result: IntentQueries = llm_client.call_structured(
        model=model,
        messages=[
            {"role": "system", "content": INTENT_PARSE_SYSTEM},
            {"role": "user",   "content": format_intent_user(user_request)},
        ],
        schema=IntentQueries,
        label="intent-parse",
    )
    queries = [q.strip() for q in result.queries if q.strip()]
    return queries if queries else ["relevant courses"]


def _retrieve_candidates(
    retriever: Retriever,
    queries: list[str],
    eligible_pool: set[str],
    required_injected: set[str],
    request: PlannerRequest,
) -> list:
    """Stage 4b+c: Retrieve, re-rank, merge, inject required courses."""
    from scheduler.retriever import RetrievalResult

    seen: dict[str, RetrievalResult] = {}

    filt = RetrieverFilter(
        candidate_pool=eligible_pool,
        subj=request.preferred_subjects,
        seasons=None,  # don't restrict by season — offered filter already done
        active_only=True,
    )

    for query in queries:
        hits = retriever.retrieve(
            query=query,
            k=request.k_per_query,
            filt=filt,
            rerank=True,
            first_stage_k=request.first_stage_k,
        )
        for h in hits:
            if h.code not in seen or h.reranker_score > (seen[h.code].reranker_score or -99):
                seen[h.code] = h

    # Inject required courses that weren't retrieved (they may not be semantically
    # related to the user's query but the student still needs them).
    for code in required_injected:
        if code not in seen:
            hit = retriever.get_course(code)
            if hit:
                # Give required courses a floor score so they sort near the top.
                hit.reranker_score = hit.reranker_score if hit.reranker_score else 0.5
                seen[code] = hit

    # Sort: required first (by rerank score), then the rest.
    def _sort_key(r):
        is_req = 1 if r.code in required_injected else 0
        return (is_req, r.reranker_score or r.score)

    candidates = sorted(seen.values(), key=_sort_key, reverse=True)
    return candidates


def _call_planner(
    raw: dict,
    student: Student,
    remaining,
    candidates: list,
    required_injected: set[str],
    request: PlannerRequest,
) -> PlannerOutput:
    """Stage 5: Main LLM plan proposal."""
    user_msg = format_planner_user(
        name=raw.get("name", "Student"),
        program=student.program,
        semester=raw.get("current_semester", 0),
        cgpa=raw.get("cgpa", 0.0),
        completed=list(student.completed),
        in_progress=list(student.in_progress),
        minors=raw.get("minors", []),
        remaining=remaining,
        candidates=candidates,
        required_injected=list(required_injected),
        user_request=request.user_request,
        target_term=request.target_term,
        min_credits=request.min_credits,
        max_credits=request.max_credits,
        target_credits=request.target_credits,
    )

    return llm_client.call_structured(
        model=request.planner_model,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        schema=PlannerOutput,
        label="planner",
    )


def _call_repair(
    previous_plan: list[str],
    violations: list[dict],
    candidates: list,
    request: PlannerRequest,
) -> RepairOutput:
    """Stage 7: Ask the LLM to fix a failing plan."""
    user_msg = format_repair_user(
        previous_plan=previous_plan,
        violations=violations,
        candidates=candidates,
        min_credits=request.min_credits,
        max_credits=request.max_credits,
    )
    return llm_client.call_structured(
        model=request.planner_model,
        messages=[
            {"role": "system", "content": REPAIR_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        schema=RepairOutput,
        label="repair",
    )


def _try_timetable(plan: list[str], target_term: str, offerings: dict) -> tuple[dict | None, list[str]]:
    """Stage 8: Best-effort timetable resolution. Returns (timetable_dict, warnings)."""
    warnings: list[str] = []
    if target_term not in offerings:
        warnings.append(
            f"No live section data for term {target_term} — timetable skipped. "
            f"Run scraper/scrape_offerings.py for that term."
        )
        return None, warnings

    try:
        from scheduler.timetable import build_timetable
        result = build_timetable(plan, target_term, offerings=offerings)
        if not result.ok:
            warnings.append(f"Timetable: {result.reason}")
            return None, warnings
        # Convert TimetableResult to a simple dict keyed by course code
        timetable = {p["code"]: p for p in result.picks}
        if result.missing:
            warnings.append(f"No sections found for: {', '.join(result.missing)}")
        return timetable, warnings
    except Exception as exc:
        warnings.append(f"Timetable resolution failed: {exc}")
        return None, warnings


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def plan(request: PlannerRequest) -> TermPlan:
    """Run the full Stages 1–8 pipeline and return a TermPlan."""
    warnings: list[str] = []

    # Stage 1 — Profile
    print(f"[Agent] Stage 1: loading student profile…")
    student, raw = _load_student(request)
    print(f"[Agent]   Program: {student.program} | "
          f"Completed: {len(student.completed)} | In-progress: {len(student.in_progress)}")

    # Stage 2 — Requirements
    print(f"[Agent] Stage 2: computing graduation requirements…")
    remaining = compute_remaining(
        program=student.program,
        completed_for_eligibility=student.completed,
        in_progress=student.in_progress,
    )
    print(f"[Agent]   Required left: {remaining.required_left}")

    # Stage 3 — Candidate pool
    print(f"[Agent] Stage 3: building eligible + offered candidate pool…")
    graph = _get_graph(student.program)
    offerings = load_offerings()
    eligible_pool, required_injected = _build_candidate_pool(
        graph, student, request, remaining.required_left, offerings,
    )
    print(f"[Agent]   Eligible pool: {len(eligible_pool)} courses | "
          f"Required injected: {len(required_injected)}")

    if not eligible_pool:
        warnings.append("Eligible pool is empty — check prereqs and offerings data.")

    # Stage 4 — Intent + retrieval
    print(f"[Agent] Stage 4: parsing intent and retrieving candidates…")
    queries = _parse_intents(request.user_request, model=request.intent_model)
    print(f"[Agent]   Queries: {queries}")
    retriever = _get_retriever()
    candidates = _retrieve_candidates(
        retriever, queries, eligible_pool, required_injected, request,
    )
    print(f"[Agent]   Candidate list size: {len(candidates)}")

    if not candidates:
        warnings.append("No candidates retrieved — plan may be empty.")

    # Stage 5 — LLM plan proposal
    print(f"[Agent] Stage 5: proposing plan with {request.planner_model}…")
    llm_out = _call_planner(raw, student, remaining, candidates, required_injected, request)
    print(f"[Agent]   Proposed plan: {llm_out.plan}")

    # Stage 6 + 7 — Validate → repair loop
    current_plan = llm_out.plan
    current_reasoning = {r.code: r.reason for r in llm_out.reasoning}
    current_summary = llm_out.summary
    current_alternatives = llm_out.alternatives_considered
    iterations = 0

    for attempt in range(request.max_repair_iters + 1):
        print(f"[Agent] Stage 6 (attempt {attempt+1}): validating plan…")
        report = validate_plan(graph, student, current_plan)
        if report.ok:
            print(f"[Agent]   Validation passed ✓")
            break

        print(f"[Agent]   {len(report.violations)} violation(s):")
        for v in report.violations:
            print(f"[Agent]     [{v.kind}] {v.message}")

        if attempt >= request.max_repair_iters:
            warnings.append(
                f"Plan still has violations after {request.max_repair_iters} repair "
                "attempts. Returning best partial plan — review violations."
            )
            break

        print(f"[Agent] Stage 7 (iter {attempt+1}): requesting repair…")
        iterations += 1
        try:
            repair = _call_repair(
                current_plan,
                [v.to_dict() for v in report.violations],
                candidates,
                request,
            )
            current_plan = repair.plan
            current_reasoning = {r.code: r.reason for r in repair.reasoning}
            current_summary = repair.summary
            print(f"[Agent]   Repaired plan: {current_plan}")
        except Exception as exc:
            warnings.append(f"Repair call failed ({exc}); keeping previous plan.")
            break

    # Final validation report for the output
    final_report = validate_plan(graph, student, current_plan)

    # Stage 8 — Timetable
    print(f"[Agent] Stage 8: resolving timetable…")
    timetable, tt_warnings = _try_timetable(current_plan, request.target_term, offerings)
    warnings.extend(tt_warnings)

    return TermPlan(
        plan=current_plan,
        reasoning=current_reasoning,
        summary=current_summary,
        total_credits=final_report.total_credits,
        validation_ok=final_report.ok,
        violations=[v.to_dict() for v in final_report.violations],
        auto_added_coreqs=final_report.auto_added_coreqs,
        timetable=timetable,
        iterations=iterations,
        alternatives=current_alternatives,
        candidate_pool_size=len(candidates),
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(
        description="SuSchedule-r one-shot planner agent",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument(
        "--transcript", "-t", required=True,
        help="Path to transcript JSON (e.g. data/transcript_cagan.json) or PDF."
    )
    ap.add_argument(
        "--request", "-r", default="",
        help='Student\'s free-text request, e.g. "I want ML and a database course".'
    )
    ap.add_argument(
        "--term", default="202601",
        help="Target term code (default: 202601 = Fall 2026-2027)."
    )
    ap.add_argument("--min-credits", type=float, default=12.0)
    ap.add_argument("--max-credits", type=float, default=21.0)
    ap.add_argument("--target-credits", type=float, default=17.0)
    ap.add_argument("--planner-model", default="gpt-4o")
    ap.add_argument("--intent-model", default="gpt-4o")
    ap.add_argument("--no-required", action="store_true",
                    help="Don't auto-inject required courses.")
    ap.add_argument("--json", action="store_true",
                    help="Output raw JSON instead of pretty-printed result.")
    args = ap.parse_args()

    transcript_path = Path(args.transcript)
    is_json = transcript_path.suffix.lower() == ".json"

    req = PlannerRequest(
        target_term=args.term,
        user_request=args.request,
        transcript_json=transcript_path if is_json else None,
        transcript_pdf=transcript_path if not is_json else None,
        min_credits=args.min_credits,
        max_credits=args.max_credits,
        target_credits=args.target_credits,
        planner_model=args.planner_model,
        intent_model=args.intent_model,
        include_required=not args.no_required,
    )

    result = plan(req)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return

    # Pretty print
    print("\n" + "═" * 60)
    print(f"  TERM PLAN — {term_label(args.term)}")
    print("═" * 60)
    print(f"\n📚 Courses ({len(result.plan)}, {result.total_credits:.1f} credits total):\n")
    for code in result.plan:
        reason = result.reasoning.get(code, "")
        print(f"  • {code}")
        if reason:
            print(f"    {reason}")
    print(f"\n📝 Summary:\n  {result.summary}")

    if result.auto_added_coreqs:
        print(f"\n⚡ Auto-added co-requisites: {', '.join(result.auto_added_coreqs)}")

    if result.timetable:
        print("\n🗓️  Timetable:")
        for code, section in result.timetable.items():
            crn = section.get("crn", "?")
            sec = section.get("section", "?")
            meetings = section.get("meetings", [])
            mtg_str = " / ".join(
                f"{m.get('days_raw','?')} {m.get('start','?')}–{m.get('end','?')}"
                for m in meetings[:2]
            )
            print(f"  • {code} — CRN {crn} (Sec {sec}) {mtg_str}")

    if not result.validation_ok:
        print(f"\n⚠️  Validation issues ({len(result.violations)}):")
        for v in result.violations:
            print(f"  [{v['kind']}] {v['message']}")

    if result.warnings:
        print(f"\n💬 Warnings:")
        for w in result.warnings:
            print(f"  • {w}")

    usage = llm_client.get_session_usage()
    print(f"\n📊 LLM usage: {usage['total_tokens']} tokens total "
          f"(prompt={usage['prompt_tokens']}, completion={usage['completion_tokens']})")
    print(f"   Repair iterations: {result.iterations} | "
          f"Candidate pool: {result.candidate_pool_size}")


if __name__ == "__main__":
    main()
