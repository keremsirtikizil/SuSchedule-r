"""
End-to-end agent evaluation for SuSchedule-r (LIVE -- calls OpenAI).

Runs a labeled set of student questions through the full ReAct agent, then
measures:

  - Answer grounding : structured checks for required course codes, terms,
                       credit facts, and forbidden claims.
  - Action success   : whether required semantic actions such as catalog
                       retrieval or timetable construction appear in the trace.
  - Task success     : both answer grounding and required actions pass.
  - Latency / tokens / tool calls.

Full answers and scoring failures are saved to eval/agent_transcript.md for
manual error analysis. Structured observed actions are saved in
eval/results_agent.json. Cost: a handful of gpt-4o calls per question.

Usage (needs OPENAI_API_KEY in .env):
    python -m eval.run_agent
    python -m eval.run_agent --dataset general_conversation_questions.json
"""
from __future__ import annotations

import argparse
import json
import hashlib
import re
import time
from pathlib import Path

from scheduler import llm_client
from scheduler.catalog import Catalog
from scheduler.schemas import PlannerRequest
from scheduler.session import PlannerSession

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"


COURSE_CODE_RE = re.compile(r"\b([A-Z]{2,5})\s+(\d{2,6}[A-Z]?)\b", re.I)


def _normalized_code(subject: str, number: str) -> str:
    return f"{subject.upper()} {number.upper()}"


def extract_course_codes(answer: str, known_subjects: set[str]) -> set[str]:
    """Find course-like codes without mistaking phrases such as FALL 2026 for one."""
    return {
        _normalized_code(match.group(1), match.group(2))
        for match in COURSE_CODE_RE.finditer(answer)
        if match.group(1).upper() in known_subjects
    }


def _credit_fact_present(answer: str, code: str, credits: float) -> bool:
    code_pattern = re.escape(code).replace(r"\ ", r"\s+")
    credit_value = rf"{credits:g}(?:\.0)?"
    credit_pattern = rf"\b{credit_value}\s*(?:SU(?:\s+credits?)?|credits?)\b"
    return bool(
        re.search(rf"{code_pattern}.{{0,180}}{credit_pattern}", answer, re.I | re.S)
        or re.search(rf"{credit_pattern}.{{0,180}}{code_pattern}", answer, re.I | re.S)
    )


def evaluate_answer(answer: str, q: dict, known_subjects: set[str]) -> tuple[bool, list[str], list[str]]:
    """Check one answer against its labels and explain any misses."""
    a = answer.lower()
    failures: list[str] = []
    mentioned = extract_course_codes(answer, known_subjects)

    # Keep support for the first version of the dataset.
    for s in q.get("must_contain", []):
        if s.lower() not in a:
            failures.append(f"missing required text: {s}")
    anyof = q.get("any_of")
    if anyof and not any(s.lower() in a for s in anyof):
        failures.append(f"missing any accepted text: {anyof}")
    for s in q.get("must_not", []):
        if s.lower() in a:
            failures.append(f"contains forbidden text: {s}")

    for term in q.get("required_terms", []):
        if term.lower() not in a:
            failures.append(f"missing required term: {term}")
    any_terms = q.get("any_terms", [])
    if any_terms and not any(term.lower() in a for term in any_terms):
        failures.append(f"missing any accepted term: {any_terms}")
    for term in q.get("forbidden_terms", []):
        if term.lower() in a:
            failures.append(f"contains forbidden term: {term}")

    required_codes = set(q.get("required_course_codes", []))
    missing_codes = sorted(required_codes - mentioned)
    if missing_codes:
        failures.append(f"missing required course codes: {missing_codes}")

    any_codes = set(q.get("any_course_codes", []))
    if any_codes and not (any_codes & mentioned):
        failures.append(f"missing any accepted course code: {sorted(any_codes)}")

    forbidden_codes = set(q.get("forbidden_course_codes", []))
    present_forbidden = sorted(forbidden_codes & mentioned)
    if present_forbidden:
        failures.append(f"contains forbidden course codes: {present_forbidden}")

    allowed_codes = set(q.get("allowed_course_codes", []))
    if allowed_codes:
        unexpected = sorted(mentioned - allowed_codes)
        if unexpected:
            failures.append(f"contains unexpected course codes: {unexpected}")

    credit_fact = q.get("course_credit")
    if credit_fact and not _credit_fact_present(
        answer,
        str(credit_fact["code"]),
        float(credit_fact["credits"]),
    ):
        failures.append(
            f"missing linked credit fact: {credit_fact['code']} = {credit_fact['credits']} SU"
        )

    for pattern in q.get("required_patterns", []):
        if not re.search(pattern, answer, re.I | re.S):
            failures.append(f"missing required regex: {pattern}")
    any_patterns = q.get("any_patterns", [])
    if any_patterns and not any(re.search(pattern, answer, re.I | re.S) for pattern in any_patterns):
        failures.append(f"missing any accepted regex: {any_patterns}")
    for pattern in q.get("forbidden_patterns", []):
        if re.search(pattern, answer, re.I | re.S):
            failures.append(f"matched forbidden regex: {pattern}")

    return not failures, failures, sorted(mentioned)


def trace_actions(trace: list[dict]) -> set[str]:
    """Collect tool calls and automatic prefetches under the same action names."""
    actions: set[str] = set()
    event_action = {
        "retrieval": "retrieve_catalog_courses",
        "build_timetable": "build_timetable",
        "selected_graphs": "select_degree_graphs",
        "requirements": "get_requirement_state",
        "check_courses_against_current_schedule": "check_courses_against_current_schedule",
    }
    for ev in trace or []:
        event_type = ev.get("type")
        if event_type in event_action:
            actions.add(event_action[event_type])
        if event_type == "assistant_step":
            for call in ev.get("tool_calls", []) or []:
                name = call.get("name")
                if name:
                    actions.add(name)
    return actions


def count_tool_calls(trace: list[dict]) -> int:
    n = 0
    for ev in trace or []:
        if ev.get("type") == "assistant_step":
            n += len(ev.get("tool_calls", []) or [])
    return n


def _summarize_rows(rows: list[dict], spec: dict, dataset_bytes: bytes, total_session_tokens: int) -> dict:
    n = len(rows)
    grounding_acc = sum(r["answer_grounded"] for r in rows) / n if n else 0.0
    action_acc = sum(r["action_ok"] for r in rows) / n if n else 0.0
    task_acc = sum(r["task_success"] for r in rows) / n if n else 0.0
    avg_lat = sum(r["latency_s"] for r in rows) / n if n else 0.0
    avg_tok = sum(r["tokens"] for r in rows) / n if n else 0.0
    avg_calls = sum(r["tool_calls"] for r in rows) / n if n else 0.0
    by_review_status = {}
    for status in sorted({r["review_status"] for r in rows}):
        subset = [r for r in rows if r["review_status"] == status]
        by_review_status[status] = {
            "n": len(subset),
            "answer_grounding_accuracy": round(
                sum(r["answer_grounded"] for r in subset) / len(subset), 4
            ),
            "action_success_accuracy": round(
                sum(r["action_ok"] for r in subset) / len(subset), 4
            ),
            "task_success_accuracy": round(
                sum(r["task_success"] for r in subset) / len(subset), 4
            ),
        }
    return {
        "n": n,
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "mode": spec["mode"],
        "answer_grounding_accuracy": round(grounding_acc, 4),
        "action_success_accuracy": round(action_acc, 4),
        "task_success_accuracy": round(task_acc, 4),
        "avg_latency_s": round(avg_lat, 2),
        "avg_tokens": round(avg_tok, 1),
        "avg_tool_calls": round(avg_calls, 2),
        "by_review_status": by_review_status,
        "per_question": rows,
        "total_session_tokens": total_session_tokens,
    }


def _print_summary(summary: dict) -> None:
    n = summary["n"]
    print(f"\n=== End-to-end agent ({n} questions, mode={summary['mode']}) ===")
    print(
        f"  answer grounding   : {summary['answer_grounding_accuracy']:.3f}  "
        f"({sum(r['answer_grounded'] for r in summary['per_question'])}/{n})"
    )
    print(
        f"  action success     : {summary['action_success_accuracy']:.3f}  "
        f"({sum(r['action_ok'] for r in summary['per_question'])}/{n})"
    )
    print(
        f"  task success       : {summary['task_success_accuracy']:.3f}  "
        f"({sum(r['task_success'] for r in summary['per_question'])}/{n})"
    )
    print(f"  avg latency        : {summary['avg_latency_s']:.1f} s")
    print(f"  avg tokens/question: {summary['avg_tokens']:.0f}")
    print(f"  avg tool calls     : {summary['avg_tool_calls']:.1f}")
    for status, metrics in summary["by_review_status"].items():
        print(
            f"  {status:<18}: task_success={metrics['task_success_accuracy']:.3f} "
            f"(n={metrics['n']})"
        )


def _render_transcript(spec: dict, rows: list[dict]) -> str:
    transcript_label = spec.get("transcript") or "none"
    md = [
        "# SuSchedule-r — end-to-end agent transcript\n",
        f"Transcript: `{transcript_label}` · term {spec['target_term']} · mode {spec['mode']}\n",
    ]
    for row in rows:
        md.append(f"\n## [{row['id']}] {row['q']}\n")
        md.append(
            f"*task_success={row['task_success']} · answer_grounded={row['answer_grounded']} · "
            f"action_ok={row['action_ok']} · review_status={row['review_status']} · "
            f"{row['latency_s']:.1f}s · {row['tokens']} tokens · {row['tool_calls']} tool calls*\n"
        )
        failures = row["failures"]
        missing_actions = row["missing_actions"]
        forbidden_actions = row.get("forbidden_actions_seen", [])
        if failures or missing_actions or forbidden_actions:
            scoring_failures = (
                failures
                + ([f"missing actions: {missing_actions}"] if missing_actions else [])
                + ([f"forbidden actions seen: {forbidden_actions}"] if forbidden_actions else [])
            )
            md.append(f"\nScoring failures: `{scoring_failures}`\n")
        clean_answer = "\n".join(line.rstrip() for line in row["answer"].splitlines()).strip()
        md.append(f"\n{clean_answer}\n")
    return "\n".join(md)


def _write_outputs(spec: dict, summary: dict) -> None:
    paths = _result_paths(spec)
    paths["json"].write_text(json.dumps(summary, indent=2))
    paths["transcript"].write_text(_render_transcript(spec, summary["per_question"]))
    print(f"\nSaved -> {paths['json'].relative_to(ROOT)}, {paths['transcript'].relative_to(ROOT)}")


def _score_saved_answers(spec: dict, dataset_bytes: bytes, known_subjects: set[str]) -> dict:
    result_path = _result_paths(spec)["json"]
    if not result_path.exists():
        raise SystemExit(f"No {result_path} exists to rescore. Run the live evaluation first.")
    previous = json.loads(result_path.read_text())
    previous_by_id = {row["id"]: row for row in previous.get("per_question", [])}
    rows = []
    for q in spec["questions"]:
        old = previous_by_id.get(q["id"])
        if not old or "answer" not in old:
            raise SystemExit(
                f"Saved result for {q['id']} has no raw answer. Run the live evaluation once "
                "with the current evaluator before using --rescore."
            )
        answer = old["answer"]
        actions = set(old.get("observed_actions", []))
        answer_ok, failures, mentioned_codes = evaluate_answer(answer, q, known_subjects)
        missing_actions = sorted(set(q.get("required_actions", [])) - actions)
        forbidden_actions = sorted(set(q.get("forbidden_actions", [])) & actions)
        action_ok = not missing_actions
        if forbidden_actions:
            action_ok = False
        rows.append({
            **old,
            "kind": q["kind"],
            "q": q["q"],
            "review_status": q.get("review_status", "existing"),
            "answer_grounded": answer_ok,
            "action_ok": action_ok,
            "task_success": answer_ok and action_ok,
            "failures": failures,
            "mentioned_course_codes": mentioned_codes,
            "missing_actions": missing_actions,
            "forbidden_actions_seen": forbidden_actions,
        })
    return _summarize_rows(
        rows,
        spec,
        dataset_bytes,
        int(previous.get("total_session_tokens", 0)),
    )


def _result_paths(spec: dict) -> dict[str, Path]:
    stem = spec.get("output_stem") or "agent"
    return {
        "json": EVAL_DIR / f"results_{stem}.json",
        "transcript": EVAL_DIR / f"{stem}_transcript.md",
    }


def _resolve_dataset_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = EVAL_DIR / path
    return path


def _meeting_when(meetings: list[dict]) -> str:
    return "; ".join(
        f"{'/'.join(m.get('day_labels', []))} {m.get('start_label', '')}-{m.get('end_label', '')}"
        for m in meetings
    )


def _resolve_schedule_fixture(session: PlannerSession, picks: list[dict]) -> list[dict]:
    """Resolve eval [{code, crn?}] picks into the same shape the UI stores."""
    from scheduler.catalog import corequisite_codes
    from scheduler.timetable import (
        annotate_meeting,
        load_offerings,
        resolve_schedule_term,
        sections_for,
    )

    session._ensure_heavy_state()
    offerings = session._offerings or load_offerings()
    schedule_term, _ = resolve_schedule_term(offerings, session.base_request.target_term)
    catalog = session.catalog

    provided_crns: dict[str, str] = {}
    requested_order: list[str] = []
    for pick in picks:
        code = str(pick.get("code", "")).upper().strip()
        crn = str(pick.get("crn") or "").strip()
        if not code:
            continue
        if code not in provided_crns:
            requested_order.append(code)
        if crn or code not in provided_crns:
            provided_crns[code] = crn

    expanded: list[tuple[str, str, bool, str | None]] = []
    seen: set[str] = set()
    for code in requested_order:
        if code in seen:
            continue
        expanded.append((code, provided_crns.get(code, ""), False, None))
        seen.add(code)
        for coreq in corequisite_codes(catalog, code):
            if coreq not in seen:
                expanded.append((coreq, provided_crns.get(coreq, ""), True, code))
                seen.add(coreq)

    resolved: list[dict] = []
    for code, crn, is_coreq, parent_code in expanded:
        sections = sections_for(offerings, schedule_term, code) if schedule_term else []
        section = next((s for s in sections if crn and str(s.get("crn")) == crn), None)
        if section is None and sections:
            section = sections[0]
        course = catalog.get(code)
        meetings = [annotate_meeting(dict(m)) for m in (section.get("meetings", []) if section else [])]
        resolved.append({
            "code": code,
            "crn": str(section.get("crn")) if section else None,
            "section": section.get("section") if section else None,
            "title": course.title if course else (section.get("title") if section else ""),
            "su_credit": course.su_credit if course else None,
            "meetings": meetings,
            "when": _meeting_when(meetings),
            "offered": section is not None,
            "is_corequisite": is_coreq,
            "corequisite_for": parent_code,
        })
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="agent_questions.json",
        help="Dataset JSON path. Relative paths are resolved under eval/.",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Re-evaluate saved raw answers against the current labels without calling OpenAI.",
    )
    args = parser.parse_args()

    dataset_path = _resolve_dataset_path(args.dataset)
    dataset_bytes = dataset_path.read_bytes()
    spec = json.loads(dataset_bytes)
    catalog = Catalog.load()
    known_subjects = {course.subj.upper() for course in catalog.courses}
    if args.rescore:
        summary = _score_saved_answers(spec, dataset_bytes, known_subjects)
        _print_summary(summary)
        _write_outputs(spec, summary)
        return

    req = PlannerRequest(
        target_term=spec["target_term"],
        user_request="",
        transcript_json=(ROOT / spec["transcript"]) if spec.get("transcript") else None,
        mode=spec["mode"],
    )
    session = PlannerSession.from_request(req)
    session._ensure_heavy_state()

    rows = []

    for q in spec["questions"]:
        if "schedule_picks" in q:
            session.user_schedule = _resolve_schedule_fixture(session, q["schedule_picks"])
        elif q.get("clear_schedule"):
            session.user_schedule = []

        usage_before = llm_client.get_session_usage()["total_tokens"]
        t0 = time.time()
        try:
            answer = session.handle_turn(q["q"])
        except Exception as exc:
            raise RuntimeError(
                f"Agent evaluation aborted on {q['id']}: {type(exc).__name__}: {exc}. "
                "No result files were written."
            ) from exc
        dt = time.time() - t0
        usage_after = llm_client.get_session_usage()["total_tokens"]
        tokens = usage_after - usage_before
        calls = count_tool_calls(session.last_trace)
        answer_ok, failures, mentioned_codes = evaluate_answer(answer, q, known_subjects)
        actions = trace_actions(session.last_trace)
        required_actions = set(q.get("required_actions", []))
        missing_actions = sorted(required_actions - actions)
        forbidden_actions = sorted(set(q.get("forbidden_actions", [])) & actions)
        action_ok = not missing_actions and not forbidden_actions
        ok = answer_ok and action_ok
        rows.append({
            "id": q["id"], "kind": q["kind"], "q": q["q"],
            "review_status": q.get("review_status", "existing"),
            "schedule_fixture_codes": [p.get("code") for p in q.get("schedule_picks", [])],
            "answer": answer,
            "answer_grounded": answer_ok,
            "action_ok": action_ok,
            "task_success": ok,
            "failures": failures,
            "mentioned_course_codes": mentioned_codes,
            "observed_actions": sorted(actions),
            "missing_actions": missing_actions,
            "forbidden_actions_seen": forbidden_actions,
            "latency_s": round(dt, 2),
            "tokens": tokens,
            "tool_calls": calls,
        })
        print(
            f"[{q['id']}] success={ok} answer={answer_ok} actions={action_ok} "
            f"{dt:5.1f}s tokens={tokens:6d} calls={calls}  {q['q']}"
        )

    summary = _summarize_rows(
        rows,
        spec,
        dataset_bytes,
        llm_client.get_session_usage()["total_tokens"],
    )
    _print_summary(summary)
    _write_outputs(spec, summary)


if __name__ == "__main__":
    main()
