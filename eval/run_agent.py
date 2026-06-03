"""
End-to-end agent evaluation for SuSchedule-r (LIVE -- calls OpenAI).

Runs a labeled set of student questions (eval/agent_questions.json) through the
full ReAct agent with a real transcript loaded, then measures:

  - Grounding accuracy : fraction of answers that contain the correct,
                         catalog-grounded fact (`must_contain` / `any_of`) and
                         none of the `must_not` hallucination markers.
  - Latency            : wall-clock seconds per question.
  - Token cost         : OpenAI tokens per question (from llm_client usage).
  - Tool calls         : how many tool invocations the ReAct loop made.

Full answers + traces are saved to eval/agent_transcript.md for manual error
analysis. Cost: a handful of gpt-4o calls per question.

Usage (needs OPENAI_API_KEY in .env):
    python -m eval.run_agent
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from scheduler import llm_client
from scheduler.schemas import PlannerRequest
from scheduler.session import PlannerSession

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"


def grounded(answer: str, q: dict) -> bool:
    a = answer.lower()
    for s in q.get("must_contain", []):
        if s.lower() not in a:
            return False
    anyof = q.get("any_of")
    if anyof and not any(s.lower() in a for s in anyof):
        return False
    for s in q.get("must_not", []):
        if s.lower() in a:
            return False
    return True


def count_tool_calls(trace: list[dict]) -> int:
    n = 0
    for ev in trace or []:
        if ev.get("type") == "assistant_step":
            n += len(ev.get("tool_calls", []) or [])
    return n


def main() -> None:
    spec = json.loads((EVAL_DIR / "agent_questions.json").read_text())
    req = PlannerRequest(
        target_term=spec["target_term"],
        user_request="",
        transcript_json=ROOT / spec["transcript"],
        mode=spec["mode"],
    )
    session = PlannerSession.from_request(req)
    session._ensure_heavy_state()

    rows = []
    md = ["# SuSchedule-r — end-to-end agent transcript\n",
          f"Transcript: `{spec['transcript']}` · term {spec['target_term']} · mode {spec['mode']}\n"]

    for q in spec["questions"]:
        usage_before = llm_client.get_session_usage()["total_tokens"]
        t0 = time.time()
        try:
            answer = session.handle_turn(q["q"])
        except Exception as exc:
            answer = f"[ERROR] {type(exc).__name__}: {exc}"
        dt = time.time() - t0
        usage_after = llm_client.get_session_usage()["total_tokens"]
        tokens = usage_after - usage_before
        calls = count_tool_calls(session.last_trace)
        ok = grounded(answer, q)
        rows.append({
            "id": q["id"], "kind": q["kind"], "q": q["q"],
            "grounded": ok, "latency_s": round(dt, 2),
            "tokens": tokens, "tool_calls": calls,
        })
        print(f"[{q['id']}] grounded={ok} {dt:5.1f}s tokens={tokens:6d} calls={calls}  {q['q']}")
        md.append(f"\n## [{q['id']}] {q['q']}\n")
        md.append(f"*grounded={ok} · {dt:.1f}s · {tokens} tokens · {calls} tool calls*\n")
        md.append(f"\n{answer}\n")

    n = len(rows)
    acc = sum(r["grounded"] for r in rows) / n if n else 0.0
    avg_lat = sum(r["latency_s"] for r in rows) / n if n else 0.0
    avg_tok = sum(r["tokens"] for r in rows) / n if n else 0.0
    avg_calls = sum(r["tool_calls"] for r in rows) / n if n else 0.0

    print(f"\n=== End-to-end agent ({n} questions, mode={spec['mode']}) ===")
    print(f"  grounding accuracy : {acc:.3f}  ({sum(r['grounded'] for r in rows)}/{n})")
    print(f"  avg latency        : {avg_lat:.1f} s")
    print(f"  avg tokens/question: {avg_tok:.0f}")
    print(f"  avg tool calls     : {avg_calls:.1f}")

    summary = {
        "n": n, "mode": spec["mode"],
        "grounding_accuracy": round(acc, 4),
        "avg_latency_s": round(avg_lat, 2),
        "avg_tokens": round(avg_tok, 1),
        "avg_tool_calls": round(avg_calls, 2),
        "per_question": rows,
        "total_session_tokens": llm_client.get_session_usage()["total_tokens"],
    }
    (EVAL_DIR / "results_agent.json").write_text(json.dumps(summary, indent=2))
    (EVAL_DIR / "agent_transcript.md").write_text("\n".join(md))
    print("\nSaved -> eval/results_agent.json, eval/agent_transcript.md")


if __name__ == "__main__":
    main()
