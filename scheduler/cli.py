"""
SuSchedule-r interactive CLI — Phase B conversational planner.

Usage
-----
    python -m scheduler.cli \\
        --transcript data/transcript_cagan.json \\
        --term 202601

Commands during the REPL
-------------------------
  Any natural language message   → processed by the planner agent
  /plan <request>                → explicitly trigger a new plan
  /state                         → show session state (pool size, history turns, …)
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


def _handle_slash_command(cmd: str, session: PlannerSession) -> bool:
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

def run_repl(session: PlannerSession) -> None:
    print(WELCOME_BANNER)
    name = session._raw.get("name", "Student")
    program = session.student.program
    from scheduler.prompts import term_label
    target = term_label(session.base_request.target_term)
    print(f"Student  : {name} ({program})")
    print(f"Planning for: {target}")
    print()

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Slash commands
        if _handle_slash_command(user_input, session):
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
        "--transcript", "-t", required=True,
        help="Path to transcript JSON (e.g. data/transcript_cagan.json) or PDF."
    )
    ap.add_argument(
        "--term", default="202601",
        help="Target term code (default: 202601 = Fall 2026-2027)."
    )
    ap.add_argument("--min-credits",    type=float, default=12.0)
    ap.add_argument("--max-credits",    type=float, default=21.0)
    ap.add_argument("--target-credits", type=float, default=17.0)
    ap.add_argument("--planner-model",  default="gpt-4o")
    ap.add_argument("--intent-model",   default="gpt-4o-mini")
    ap.add_argument("--no-required",    action="store_true",
                    help="Don't auto-inject required courses.")
    ap.add_argument("--first-message",  default="",
                    help="Optional: send an initial message automatically on startup.")
    args = ap.parse_args()

    transcript_path = Path(args.transcript)
    if not transcript_path.exists():
        print(f"[Error] Transcript not found: {transcript_path}", file=sys.stderr)
        sys.exit(1)

    is_json = transcript_path.suffix.lower() == ".json"

    req = PlannerRequest(
        target_term=args.term,
        user_request="",
        transcript_json=transcript_path if is_json else None,
        transcript_pdf=transcript_path if not is_json else None,
        min_credits=args.min_credits,
        max_credits=args.max_credits,
        target_credits=args.target_credits,
        planner_model=args.planner_model,
        intent_model=args.intent_model,
        include_required=not args.no_required,
    )

    print("Loading session…")
    session = PlannerSession.from_request(req)

    # Optional first message (useful for scripted demos)
    if args.first_message:
        run_repl_with_first(session, args.first_message)
    else:
        run_repl(session)


def run_repl_with_first(session: PlannerSession, first_message: str) -> None:
    """Start the REPL but send *first_message* automatically before the prompt."""
    print(WELCOME_BANNER)
    name = session._raw.get("name", "Student")
    program = session.student.program
    from scheduler.prompts import term_label
    target = term_label(session.base_request.target_term)
    print(f"Student  : {name} ({program})")
    print(f"Planning for: {target}\n")

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
        if _handle_slash_command(user_input, session):
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
