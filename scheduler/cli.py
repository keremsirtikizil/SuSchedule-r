"""
SuSchedule-r interactive CLI — Phase B conversational planner.

Usage
-----
    python -m scheduler.cli \\
        --transcript data/transcript_cagan.json \\
        --term 202601 \\
        --mode react \\
        --debug-trace

Commands during the REPL
-------------------------
  Any natural language message   → processed by the planner agent
  /plan <request>                → explicitly trigger a new plan
  /state                         → show session state (pool size, history turns, …)
  /trace                         → print the last agent/tool trace
  /raw-trace                     → print the last full agent/tool trace as JSON
  /json                          → print the current plan as JSON
  /usage                         → show token usage for this session
  /help                          → show this help
  /exit  or  quit                → exit

Examples
--------
  >>> Plan my Fall 2026 semester. I want a deep learning course and databases.
  >>> Swap CS 406 for something about natural language processing.
  >>> Drop PHYS 113 — I'll take it next semester.
  >>> Add a free elective in economics.
  >>> Why did you pick CS 306?
  >>> /json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scheduler.schemas import PlannerRequest
from scheduler.session import PlannerSession
from scheduler import llm_client


# --------------------------------------------------------------------------- #
# REPL helpers
# --------------------------------------------------------------------------- #

HELP_TEXT = """
Commands:
  /plan <request>   Force a new plan with the given request.
  /state            Show session state (pool size, turns, current plan).
  /trace            Pretty-print the last agent/tool trace.
  /raw-trace        Print the last full agent/tool trace as raw JSON.
  /json             Print the current plan as JSON.
  /usage            Show OpenAI token usage for this session.
  /help             Show this message.
  /exit | quit      Exit the planner.

Or just type naturally:
  "Plan my semester. I want ML and OS."
  "Swap CS 406 for something about NLP."
  "Drop PHYS 113."
  "Add a free elective."
  "Why did you pick CS 306?"
"""

WELCOME_BANNER = """
╔══════════════════════════════════════════════════════╗
║          SuSchedule-r  — Course Planner              ║
║          Powered by GPT-4o + BAAI/bge reranker       ║
╚══════════════════════════════════════════════════════╝
Type your request in natural language, or /help for commands.
"""


def _print_wrapped(text: str, width: int = 80) -> None:
    """Print text, wrapping bullet points nicely."""
    print(text)


def _pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + f"\n... <truncated {len(text) - max_chars} chars>"


def _print_json_block(value: Any, max_chars: int, raw: bool = False, indent: str = "    ") -> None:
    text = _pretty_json(value)
    if not raw:
        text = _truncate(text, max_chars)
    for line in text.splitlines() or ["null"]:
        print(f"{indent}{line}")


def _format_tool_args(raw_args: Any) -> Any:
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            return raw_args
    return raw_args


def _trace_header(title: str) -> None:
    print(f"\n[trace] {title}")


def print_trace_event(event: dict, *, raw: bool = False, max_chars: int = 2500) -> None:
    """Pretty-print one trace event for live CLI demos."""
    etype = event.get("type", "event")

    if etype == "turn_start":
        _trace_header("turn start")
        print(f"  user_message     : {event.get('user_message', '')}")
        print(f"  planning_allowed : {event.get('planning_allowed')}")
        print("  state:")
        _print_json_block(event.get("state", {}), max_chars=max_chars, raw=raw)
        return

    if etype == "assistant_step":
        _trace_header(f"assistant step {event.get('step')}")
        content = (event.get("content") or "").strip()
        if content:
            print("  draft/content:")
            print(_truncate(content, max_chars) if not raw else content)
        calls = event.get("tool_calls") or []
        if not calls:
            print("  tool_calls       : none; final-answer step")
            return
        for idx, call in enumerate(calls, start=1):
            print(f"  tool_call {idx}     : {call.get('name')}")
            print("  arguments:")
            _print_json_block(_format_tool_args(call.get("arguments", {})), max_chars=max_chars, raw=raw)
        return

    if etype == "tool_result":
        suffix = f" (step {event.get('step')})" if event.get("step") is not None else ""
        _trace_header(f"tool result: {event.get('name', 'tool')}{suffix}")
        print("  arguments:")
        _print_json_block(event.get("arguments", {}), max_chars=max_chars, raw=raw)
        print("  result:")
        _print_json_block(event.get("result", {}), max_chars=max_chars, raw=raw)
        return

    if etype == "prefetched_recommendation_context":
        _trace_header("prefetched recommendation context")
        content = event.get("content", "")
        print(_truncate(content, max_chars) if not raw else content)
        return

    if etype in {"profile_reask_repair", "catalog_title_repair"}:
        _trace_header(etype.replace("_", " "))
        _print_json_block(event, max_chars=max_chars, raw=raw, indent="  ")
        return

    if etype == "final_response":
        _trace_header("final response")
        content = event.get("content", "")
        print(_truncate(content, max_chars) if not raw else content)
        return

    _trace_header(etype)
    _print_json_block(event, max_chars=max_chars, raw=raw, indent="  ")


def print_trace(trace: list[dict], *, raw: bool = False, max_chars: int = 2500) -> None:
    if not trace:
        print("No trace yet.")
        return
    print("\n================ Agent Trace ================")
    for event in trace:
        print_trace_event(event, raw=raw, max_chars=max_chars)
    print("\n============== End Agent Trace ==============\n")


def _handle_slash_command(
    cmd: str,
    session: PlannerSession,
    *,
    trace_chars: int = 2500,
) -> bool:
    """Handle /commands. Returns True if handled, False if it's a normal message."""
    cmd = cmd.strip()

    if cmd in ("/exit", "/quit", "quit", "exit"):
        print("Goodbye!")
        sys.exit(0)

    if cmd == "/help":
        print(HELP_TEXT)
        return True

    if cmd == "/state":
        s = session.state_summary()
        print(f"\nSession state:")
        print(f"  Program      : {s['program']}")
        print(f"  Target term  : {s['target_term']}")
        print(f"  Eligible pool: {s['eligible_pool_size']} courses")
        print(f"  History turns: {s['history_turns']}")
        if s["current_plan"]:
            print(f"  Current plan : {', '.join(s['current_plan'])}")
        else:
            print("  Current plan : (none yet)")
        return True

    if cmd == "/trace":
        print_trace(session.last_trace, raw=False, max_chars=trace_chars)
        return True

    if cmd == "/raw-trace":
        print_trace(session.last_raw_trace or session.last_trace, raw=True, max_chars=trace_chars)
        return True

    if cmd == "/json":
        if not session.current_plan:
            print("No plan yet.")
        else:
            print(json.dumps(session.current_plan.to_dict(), indent=2, ensure_ascii=False))
        return True

    if cmd == "/usage":
        u = llm_client.get_session_usage()
        print(f"\nOpenAI token usage this session:")
        print(f"  Prompt     : {u['prompt_tokens']:,}")
        print(f"  Completion : {u['completion_tokens']:,}")
        print(f"  Total      : {u['total_tokens']:,}")
        # Rough cost estimate at gpt-4o pricing (May 2025)
        cost = (u['prompt_tokens'] / 1_000_000 * 2.50) + (u['completion_tokens'] / 1_000_000 * 10.00)
        print(f"  Est. cost  : ~${cost:.4f} (gpt-4o list pricing)")
        return True

    if cmd.startswith("/plan "):
        request_text = cmd[len("/plan "):].strip()
        # Rewrite as a natural plan message and process normally
        response = session.handle_turn(f"Plan my semester. {request_text}")
        _print_wrapped(response)
        return True

    # Unknown slash command
    if cmd.startswith("/"):
        print(f"Unknown command: {cmd}. Type /help for a list.")
        return True

    return False


# --------------------------------------------------------------------------- #
# Main REPL
# --------------------------------------------------------------------------- #

def _install_live_trace(session: PlannerSession, *, enabled: bool, raw: bool, max_chars: int) -> None:
    if not enabled:
        session.trace_callback = None
        session.trace_raw_callback = None
        return

    def _callback(event: dict) -> None:
        print_trace_event(event, raw=raw, max_chars=max_chars)

    if raw:
        session.trace_callback = None
        session.trace_raw_callback = _callback
    else:
        session.trace_callback = _callback
        session.trace_raw_callback = None


def _print_session_intro(session: PlannerSession) -> None:
    print(WELCOME_BANNER)
    name = session._raw.get("name", "Student")
    program = session.student.program if session.student else "not set"
    from scheduler.prompts import term_label
    target = term_label(session.base_request.target_term)
    print(f"Student  : {name} ({program})")
    print(f"Planning for: {target}")
    print(f"Mode     : {session.base_request.mode}")
    print()


def run_repl(
    session: PlannerSession,
    *,
    debug_trace: bool = False,
    raw_trace: bool = False,
    trace_chars: int = 2500,
) -> None:
    _install_live_trace(session, enabled=debug_trace, raw=raw_trace, max_chars=trace_chars)
    _print_session_intro(session)

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Slash commands
        if _handle_slash_command(user_input, session, trace_chars=trace_chars):
            continue

        # Natural language turn
        try:
            response = session.handle_turn(user_input)
            print()
            _print_wrapped(response)
            print()
        except Exception as exc:
            print(f"\n[Error] {exc}")
            print("(Session state preserved — you can continue.)\n")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(
        description="SuSchedule-r conversational course planner (Phase B)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument(
        "--transcript", "-t",
        help="Path to a Degree Evaluation HTML, transcript JSON "
             "(e.g. data/transcript_cagan.json), or Academic Records PDF."
    )
    ap.add_argument(
        "--term", default="202601",
        help="Target term code (default: 202601 = Fall 2026-2027)."
    )
    ap.add_argument("--min-credits",    type=float, default=12.0)
    ap.add_argument("--max-credits",    type=float, default=21.0)
    ap.add_argument("--target-credits", type=float, default=17.0)
    ap.add_argument("--planner-model",  default="gpt-4o")
    ap.add_argument("--intent-model",   default="gpt-4o")
    ap.add_argument("--no-required",    action="store_true",
                    help="Don't auto-inject required courses.")
    ap.add_argument("--mode",           choices=["pipeline", "react"], default="react",
                    help="Agent style: 'react' (tool-using loop, default) or 'pipeline' "
                         "(legacy 8-stage).")
    ap.add_argument("--debug-trace",    action="store_true",
                    help="Live-print every ReAct trace event: assistant steps, tool calls, arguments, results, repairs, and final answer.")
    ap.add_argument("--raw-trace",      action="store_true",
                    help="With --debug-trace or /raw-trace, print the full untruncated trace payloads.")
    ap.add_argument("--trace-result-chars", type=int, default=2500,
                    help="Max characters per trace JSON/text block before truncation (default: 2500; <=0 disables truncation).")
    ap.add_argument("--first-message",  default="",
                    help="Optional: send an initial message automatically on startup.")
    ap.add_argument("--one-shot",       action="store_true",
                    help="Exit after --first-message instead of entering the REPL.")
    args = ap.parse_args()

    if args.debug_trace and args.mode != "react":
        print("[Warning] --debug-trace is most useful with --mode react; pipeline mode has fewer trace events.")

    transcript_path = Path(args.transcript) if args.transcript else None
    if transcript_path and not transcript_path.exists():
        print(f"[Error] Transcript not found: {transcript_path}", file=sys.stderr)
        sys.exit(1)

    suffix = transcript_path.suffix.lower() if transcript_path else ""
    is_html = suffix in (".html", ".htm")
    is_json = suffix == ".json"

    req = PlannerRequest(
        target_term=args.term,
        user_request="",
        transcript_html=transcript_path if transcript_path and is_html else None,
        transcript_json=transcript_path if transcript_path and is_json else None,
        transcript_pdf=transcript_path if transcript_path and not (is_html or is_json) else None,
        min_credits=args.min_credits,
        max_credits=args.max_credits,
        target_credits=args.target_credits,
        planner_model=args.planner_model,
        intent_model=args.intent_model,
        include_required=not args.no_required,
        mode=args.mode,
    )

    print("Loading session…")
    session = PlannerSession.from_request(req)

    # Optional first message (useful for scripted demos)
    if args.first_message:
        if args.one_shot:
            _install_live_trace(
                session,
                enabled=args.debug_trace,
                raw=args.raw_trace,
                max_chars=args.trace_result_chars,
            )
            _print_session_intro(session)
            print(f">>> {args.first_message}")
            response = session.handle_turn(args.first_message)
            print()
            print(response)
            print()
            return
        run_repl_with_first(
            session,
            args.first_message,
            debug_trace=args.debug_trace,
            raw_trace=args.raw_trace,
            trace_chars=args.trace_result_chars,
        )
    else:
        run_repl(
            session,
            debug_trace=args.debug_trace,
            raw_trace=args.raw_trace,
            trace_chars=args.trace_result_chars,
        )


def run_repl_with_first(
    session: PlannerSession,
    first_message: str,
    *,
    debug_trace: bool = False,
    raw_trace: bool = False,
    trace_chars: int = 2500,
) -> None:
    """Start the REPL but send *first_message* automatically before the prompt."""
    _install_live_trace(session, enabled=debug_trace, raw=raw_trace, max_chars=trace_chars)
    _print_session_intro(session)

    print(f">>> {first_message}")
    try:
        response = session.handle_turn(first_message)
        print()
        print(response)
        print()
    except Exception as exc:
        print(f"[Error] {exc}\n")

    # Continue normally
    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user_input:
            continue
        if _handle_slash_command(user_input, session, trace_chars=trace_chars):
            continue
        try:
            response = session.handle_turn(user_input)
            print()
            print(response)
            print()
        except Exception as exc:
            print(f"\n[Error] {exc}\n")


if __name__ == "__main__":
    main()
