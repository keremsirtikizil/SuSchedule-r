"""
SuSchedule-r — ReAct (tool-using) agent.

Alternative to the 8-stage pipeline in ``scheduler.agent``: the LLM is given a
small toolbox and decides which tools to call in what order. It runs in a loop
until it returns a final assistant message with no further tool calls (or hits
the iteration cap).

Tools exposed to the model
--------------------------
  get_student_profile()           — name, program, semester, cgpa, completed
  get_remaining_requirements()    — required/core/area still needed
  get_current_plan()              — current plan, credits, validation status
  retrieve_courses(query, k, subj)— semantic search over the catalog
  check_prereqs(code)             — prereq text + whether student is eligible
  get_offerings(code)             — historical offerings + likely-offered flag
  validate_plan(plan)             — eligibility.validate_plan wrapper
  set_plan(plan, reasoning, summary)
                                   — commit a new plan after validating it

Wired into :class:`scheduler.session.PlannerSession` via
``base_request.mode == "react"``.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from scheduler import llm_client
from scheduler.eligibility import (
    EligibilityOptions,
    prereqs_satisfied,
    validate_plan as elig_validate_plan,
)
from scheduler.offerings import likely_offered_in, offered_history
from scheduler.retriever import RetrieverFilter
from scheduler.schemas import TermPlan

if TYPE_CHECKING:
    from scheduler.session import PlannerSession


# --------------------------------------------------------------------------- #
# System prompt
# --------------------------------------------------------------------------- #

REACT_SYSTEM = """\
You are an academic-advising agent for Sabancı University (SU). You help
undergraduates plan a single upcoming semester, swap/drop/add courses in
an existing plan, and explain your choices.

You decide everything by calling tools. Do not invent course codes, prereqs,
credits, or offerings — read them from the tools.

Available tools
---------------
- get_student_profile()        : transcript snapshot (program, completed, in-progress)
- get_remaining_requirements() : what is still needed for graduation
- get_current_plan()           : the plan you have already committed (may be empty)
- retrieve_courses(query, k=8, subj=None)
                               : semantic search; returns code/title/credits/desc
                                 + whether the student is eligible and whether
                                 the course is likely offered next term
- check_prereqs(code)          : full prereq/coreq text + eligibility verdict
- get_offerings(code)          : terms in which the course has historically run
- validate_plan(plan)          : ok/violations/total_credits for a candidate plan
- set_plan(plan, reasoning, summary)
                               : commit a plan after validate_plan returns ok

Workflow heuristics
-------------------
1. ALWAYS call get_student_profile + get_remaining_requirements on the very
   first turn (use them to ground every later decision).
2. For new plans: retrieve_courses for each interest the student named, then
   add still-needed required courses (you can ask retrieve_courses for them
   by name/code), then validate_plan, then set_plan.
3. For swap/drop/add: call get_current_plan first; only modify what the
   student asked about; re-validate before set_plan.
4. For "explain" questions: read get_current_plan and answer directly — no
   tool calls needed beyond that.
5. NEVER call set_plan with a plan that validate_plan reported as not ok.
   If validation fails, retrieve alternatives and try again (max 2 attempts).
6. Stay within the credit range reported in the profile (12–21 by default,
   target ~17). 0-credit graduation-project courses are fine to include.

Final assistant message
-----------------------
After you finish calling tools, reply with a short markdown summary the
student will read directly. Use bullet points for course lists. If you
committed a plan, list the courses + total credits. If you explained
something, give the reasoning concisely. Do not dump raw tool output.
"""


# --------------------------------------------------------------------------- #
# Tool schemas (OpenAI function-calling format)
# --------------------------------------------------------------------------- #

def _tool(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


TOOLS: list[dict] = [
    _tool(
        "get_student_profile",
        "Return the student's transcript snapshot: name, program, semester, "
        "CGPA, completed courses, in-progress courses, declared minors, and "
        "the credit range for the target term.",
        properties={},
    ),
    _tool(
        "get_remaining_requirements",
        "Return what the student still needs to graduate: required, core "
        "elective, area elective, and free elective courses still left.",
        properties={},
    ),
    _tool(
        "get_current_plan",
        "Return the currently committed plan for the target term: list of "
        "course codes, total credits, validation status, and per-course "
        "reasoning. Returns an empty plan if none has been set yet.",
        properties={},
    ),
    _tool(
        "retrieve_courses",
        "Semantic search over the SU course catalog. Pre-filtered to courses "
        "the student is eligible for AND likely to be offered next term. "
        "Each hit includes code, title, credits, a short description, prereq "
        "text, and the cross-encoder relevance score.",
        properties={
            "query": {
                "type": "string",
                "description": "A focused search query, e.g. 'machine learning' "
                "or 'introductory economics'. One concept per query — make "
                "multiple calls for multiple interests.",
            },
            "k": {
                "type": "integer",
                "description": "Number of results to return (default 8, max 15).",
                "minimum": 1,
                "maximum": 15,
            },
            "subj": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional subject filter, e.g. ['CS','IE']. Omit "
                "to search every subject.",
            },
        },
        required=["query"],
    ),
    _tool(
        "check_prereqs",
        "Inspect a specific course's prereqs and co-reqs, and report whether "
        "the student is eligible to take it next term. Use this when you "
        "want certainty about a single course before committing to a plan.",
        properties={
            "code": {
                "type": "string",
                "description": "Course code, e.g. 'CS 306'.",
            },
        },
        required=["code"],
    ),
    _tool(
        "get_offerings",
        "Return the terms in which a course has historically been offered "
        "(and how many sections), plus whether it is likely to run in the "
        "target term.",
        properties={
            "code": {
                "type": "string",
                "description": "Course code, e.g. 'CS 306'.",
            },
        },
        required=["code"],
    ),
    _tool(
        "validate_plan",
        "Validate a candidate plan (list of course codes) against prereqs, "
        "co-reqs, duplicates, already-completed courses, and the credit "
        "range. Returns ok=true/false, total_credits, violations, and any "
        "co-requisite courses that would be auto-added.",
        properties={
            "plan": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Candidate list of course codes to validate.",
            },
        },
        required=["plan"],
    ),
    _tool(
        "set_plan",
        "Commit a plan as the student's current plan. Only call this after "
        "validate_plan returned ok=true for the same list of codes. The "
        "reasoning dict maps every course code in the plan to a 1–2 sentence "
        "explanation of why it was chosen.",
        properties={
            "plan": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Final list of course codes.",
            },
            "reasoning": {
                "type": "object",
                "description": "Map of course code → 1-2 sentence rationale. "
                "Must include every code in `plan`.",
                "additionalProperties": {"type": "string"},
            },
            "summary": {
                "type": "string",
                "description": "2-3 sentence narrative describing the semester "
                "balance (workload, subject mix, graduation progress).",
            },
        },
        required=["plan", "reasoning", "summary"],
    ),
]


# --------------------------------------------------------------------------- #
# Toolbox — bound to a PlannerSession so tools can read/write its state
# --------------------------------------------------------------------------- #

class Toolbox:
    """One Toolbox per turn, bound to a PlannerSession.

    Each method matches a tool name in :data:`TOOLS` and returns a JSON-safe
    Python object (dict / list / scalar). The ReAct loop json-encodes the
    return value before handing it back to the model as a tool message.
    """

    def __init__(self, session: "PlannerSession") -> None:
        self.session = session
        session._ensure_heavy_state()

    # -- Read-only inspection ------------------------------------------------ #

    def get_student_profile(self) -> dict:
        s = self.session.student
        raw = self.session._raw
        req = self.session.base_request
        return {
            "name": raw.get("name", "Student"),
            "program": s.program,
            "current_semester": raw.get("current_semester", 0),
            "cgpa": raw.get("cgpa", 0.0),
            "cumulative_credits": s.cumulative_credits,
            "completed_count": len(s.completed),
            "completed": sorted(s.completed),
            "in_progress": sorted(s.in_progress),
            "minors": raw.get("minors", []),
            "target_term": req.target_term,
            "credit_range": {
                "min": req.min_credits,
                "max": req.max_credits,
                "target": req.target_credits,
            },
        }

    def get_remaining_requirements(self) -> dict:
        r = self.session._remaining
        if r is None:
            return {"error": "requirements not loaded"}
        return {
            "program": r.program,
            "required_left": r.required_left,
            "core_elective_left": r.core_left,
            "area_elective_left": r.area_left,
            "free_elective_left": r.free_left,
            "required_in_progress": r.required_in_progress,
            "required_credits_left": r.required_credits_left,
        }

    def get_current_plan(self) -> dict:
        p = self.session.current_plan
        if p is None:
            return {"plan": [], "total_credits": 0.0, "validation_ok": None,
                    "reasoning": {}, "summary": ""}
        return {
            "plan": p.plan,
            "total_credits": p.total_credits,
            "validation_ok": p.validation_ok,
            "violations": p.violations,
            "reasoning": p.reasoning,
            "summary": p.summary,
            "auto_added_coreqs": p.auto_added_coreqs,
        }

    # -- Search + lookup ----------------------------------------------------- #

    def retrieve_courses(self, query: str, k: int = 8, subj: list[str] | None = None) -> list[dict]:
        k = max(1, min(int(k), 15))
        filt = RetrieverFilter(
            candidate_pool=self.session._eligible_pool,
            subj=subj or None,
            active_only=True,
        )
        hits = self.session.retriever.retrieve(query, k=k, filt=filt, rerank=True)
        return [
            {
                "rank": h.rank,
                "code": h.code,
                "title": h.title,
                "credits": h.su_credit,
                "description": (h.description or "")[:280],
                "prereq_text": h.prereq_text,
                "rerank_score": (round(h.reranker_score, 3)
                                 if h.reranker_score is not None else None),
                "is_required_left": h.code in self.session._required_injected,
            }
            for h in hits
        ]

    def check_prereqs(self, code: str) -> dict:
        code = code.upper().strip()
        graph = self.session._graph
        student = self.session._student
        if not graph.has_node(code):
            return {"code": code, "error": "course not in catalog"}
        node = graph.nodes[code]
        eligible = prereqs_satisfied(
            graph, code, student.completed,
            concurrent_ok=student.in_progress,
        )
        return {
            "code": code,
            "title": node.get("title", ""),
            "credits": node.get("su_credit"),
            "prereq_text": node.get("prereq_text", ""),
            "coreq_text": node.get("coreq_text", ""),
            "is_eligible_next_term": bool(eligible),
            "in_eligible_pool": code in self.session._eligible_pool,
            "already_completed": code in student.completed,
            "currently_taking": code in student.in_progress,
        }

    def get_offerings(self, code: str) -> dict:
        code = code.upper().strip()
        offerings = self.session._offerings or {}
        history = offered_history(offerings, code)
        target_term = self.session.base_request.target_term
        likely_set = likely_offered_in(offerings, target_term, same_season_only=True)
        return {
            "code": code,
            "target_term": target_term,
            "likely_offered_target_term": code in likely_set,
            "history": history,  # {term_code: section_count}
        }

    # -- Plan write path ----------------------------------------------------- #

    def validate_plan(self, plan: list[str]) -> dict:
        plan = [c.upper().strip() for c in plan]
        opts = EligibilityOptions()
        opts.min_term_credits = self.session.base_request.min_credits
        opts.max_term_credits = self.session.base_request.max_credits
        report = elig_validate_plan(self.session._graph, self.session._student, plan, opts)
        return {
            "ok": report.ok,
            "total_credits": report.total_credits,
            "violations": [v.to_dict() for v in report.violations],
            "auto_added_coreqs": report.auto_added_coreqs,
        }

    def set_plan(self, plan: list[str], reasoning: dict, summary: str) -> dict:
        plan = [c.upper().strip() for c in plan]
        # Re-validate as a guardrail; do not commit if not ok.
        opts = EligibilityOptions()
        opts.min_term_credits = self.session.base_request.min_credits
        opts.max_term_credits = self.session.base_request.max_credits
        report = elig_validate_plan(self.session._graph, self.session._student, plan, opts)
        if not report.ok:
            return {
                "committed": False,
                "reason": "validate_plan failed; not committing",
                "violations": [v.to_dict() for v in report.violations],
            }
        # Normalise reasoning keys to upper-case codes
        norm_reason = {str(k).upper().strip(): str(v) for k, v in (reasoning or {}).items()}
        # Reuse existing TermPlan shape so the rest of the app keeps working.
        self.session.current_plan = TermPlan(
            plan=plan,
            reasoning=norm_reason,
            summary=summary,
            total_credits=report.total_credits,
            validation_ok=True,
            violations=[],
            auto_added_coreqs=report.auto_added_coreqs,
            timetable=None,
            iterations=0,
            alternatives=[],
            candidate_pool_size=len(self.session._eligible_pool),
            warnings=[],
        )
        return {
            "committed": True,
            "plan": plan,
            "total_credits": report.total_credits,
            "auto_added_coreqs": report.auto_added_coreqs,
        }

    # -- Dispatcher ---------------------------------------------------------- #

    def dispatch(self, name: str, arguments: dict) -> Any:
        method = getattr(self, name, None)
        if method is None or name.startswith("_"):
            return {"error": f"unknown tool: {name}"}
        try:
            return method(**(arguments or {}))
        except TypeError as exc:
            return {"error": f"bad arguments for {name}: {exc}"}
        except Exception as exc:
            return {"error": f"{name} raised {type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- #
# ReAct loop
# --------------------------------------------------------------------------- #

MAX_ITERATIONS = 12


def handle_turn(session: "PlannerSession", user_message: str) -> str:
    """Run the ReAct loop for one user turn and return the assistant's reply."""
    toolbox = Toolbox(session)
    model = session.base_request.planner_model

    messages: list[dict] = [{"role": "system", "content": REACT_SYSTEM}]
    # Trim history to last 10 turns of plain user/assistant text so prior
    # tool_call structures don't leak in (we only kept text in session.history).
    messages.extend(session.history[-20:])
    messages.append({"role": "user", "content": user_message})

    final_text = ""

    for step in range(MAX_ITERATIONS):
        assistant_msg = llm_client.call_with_tools(
            model=model,
            messages=messages,
            tools=TOOLS,
            label=f"react/step{step+1}",
        )

        tool_calls = assistant_msg.tool_calls or []

        # Always append the assistant turn (with tool_calls intact) so the
        # next call sees the full reasoning trace.
        messages.append({
            "role": "assistant",
            "content": assistant_msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ] if tool_calls else None,
        })

        if not tool_calls:
            final_text = assistant_msg.content or ""
            break

        # Dispatch every tool call in this step.
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                result: Any = {"error": f"could not parse arguments: {exc}"}
            else:
                result = toolbox.dispatch(tc.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })
    else:
        # Loop exhausted without a final text message.
        final_text = (
            "I ran out of reasoning steps before settling on a plan. "
            "Could you narrow the request a bit?"
        )

    if not final_text.strip():
        final_text = "(no response)"
    return final_text
