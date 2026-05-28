"""
Intent classification and per-intent handlers for the conversational planner.

Flow
----
1. ``classify_intent(message, history, model)`` — cheap gpt-4o-mini call that
   returns an ``IntentClassification`` with the intent type and its arguments.
2. Per-intent handler functions (``handle_plan``, ``handle_swap``, etc.) are
   called by ``PlannerSession.handle_turn`` with the session as context.
3. Handlers return a formatted string that goes into the CLI output AND is
   appended to the conversation history as the assistant's reply.

Intents supported in Phase B
-----------------------------
  plan    — generate a brand-new term plan (full Stage 1-8 pipeline)
  swap    — replace one course with another or a better alternative
  drop    — remove a course from the current plan
  add     — add a course matching some description
  explain — explain why a course was chosen (whole plan or one course)
  unknown — catch-all: politely ask for clarification
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from scheduler import llm_client

if TYPE_CHECKING:
    # Avoid circular import; session imports intents, intents references session type.
    from scheduler.session import PlannerSession


# --------------------------------------------------------------------------- #
# Intent classification schema
# --------------------------------------------------------------------------- #

class IntentClassification(BaseModel):
    intent: Literal["plan", "swap", "drop", "add", "explain", "unknown"]

    # "plan" args
    plan_request: str = Field(
        default="",
        description="The student's interest/goal description for a new plan. Empty for other intents."
    )

    # "swap" args
    swap_remove: str = Field(
        default="",
        description="Course code or name to REMOVE from the plan. Empty for other intents."
    )
    swap_add_request: str = Field(
        default="",
        description=(
            "Course code, name, or description of what to ADD in place of the removed course. "
            "Empty for other intents."
        )
    )

    # "drop" args
    drop_course: str = Field(
        default="",
        description="Course code or name to drop from the plan. Empty for other intents."
    )

    # "add" args
    add_request: str = Field(
        default="",
        description="Course code, name, or description of what to add. Empty for other intents."
    )

    # "explain" args
    explain_course: str = Field(
        default="",
        description=(
            "Course code or name to explain. "
            "Empty means explain the whole plan rationale."
        )
    )


# --------------------------------------------------------------------------- #
# Intent classifier prompt
# --------------------------------------------------------------------------- #

INTENT_CLASSIFIER_SYSTEM = """\
You are a parser for a university course-planning chatbot.

Classify the student's latest message into exactly one intent and extract
its arguments. Use the conversation history for context (e.g. "that course"
might refer to one mentioned in the previous message).

Intent definitions
------------------
plan    — student wants a new/fresh semester plan
            ("plan my semester", "suggest courses for fall", "start over")
swap    — student wants to replace one course with another
            ("swap CS 406 for something about NLP",
             "replace the physics course with CS 415",
             "I'd rather take CS 440 instead of CS 449")
drop    — student wants to remove a course without replacing it
            ("drop PHYS 113", "remove the physics course", "take that out")
add     — student wants to add a course without removing anything
            ("also add a math course", "can you fit CS 403 in?")
explain — student wants to understand why a course was chosen
            ("why CS 306?", "explain the database course",
             "why did you pick that?", "tell me about the plan")
unknown — anything else; ask for clarification

Rules
-----
- Always fill in ONLY the fields relevant to the chosen intent.
- Leave all other string fields as "".
- For swap/drop/add: if the student refers to a course by title or position
  ("the first one", "the ML course"), fill in your best guess at the code
  or a short identifying description.
- Never output more than one intent.
"""


def classify_intent(
    message: str,
    history: list[dict],
    model: str = "gpt-4o-mini",
) -> IntentClassification:
    """Classify the user's message into an intent + its args.

    *history* is the recent conversation (last 6 messages max) to help
    resolve references like "that course" or "the first one".
    """
    # Build context: last 6 messages + current user message
    ctx = history[-6:] if len(history) > 6 else history
    messages = [
        {"role": "system", "content": INTENT_CLASSIFIER_SYSTEM},
        *ctx,
        {"role": "user", "content": message},
    ]
    return llm_client.call_structured(
        model=model,
        messages=messages,
        schema=IntentClassification,
        label="intent-classify",
    )


# --------------------------------------------------------------------------- #
# Helper — resolve a fuzzy course reference to a code
# --------------------------------------------------------------------------- #

def _resolve_code(session: "PlannerSession", text: str) -> str | None:
    """Try to resolve a code or partial name to an exact catalog code.

    Checks (in order):
      1. Exact code match in the retriever DataFrame (e.g. "CS 415")
      2. Exact code match in current plan
      3. Case-insensitive title substring match in the retriever DataFrame
    Returns the code string or None if unresolvable.
    """
    if not text:
        return None
    text_up = text.upper().strip()
    df = session.retriever._df

    # 1. Exact code
    if text_up in df["code"].values:
        return text_up

    # 2. In current plan (handles "the last one" partially)
    if session.current_plan:
        for code in session.current_plan.plan:
            if code.upper() == text_up:
                return code

    # 3. Title substring (case-insensitive)
    mask = df["title"].str.lower().str.contains(text.lower(), regex=False)
    matches = df[mask]["code"].tolist()
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Return the one already in the current plan if possible
        if session.current_plan:
            for code in session.current_plan.plan:
                if code in matches:
                    return code
        return matches[0]

    return None


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #

def handle_plan(session: "PlannerSession", intent: IntentClassification) -> str:
    """Re-run the full Stage 1–8 pipeline with a new request."""
    from scheduler.agent import plan as run_plan
    from scheduler.schemas import PlannerRequest

    new_request = intent.plan_request or session.base_request.user_request

    req = PlannerRequest(
        target_term=session.base_request.target_term,
        user_request=new_request,
        transcript_json=session.base_request.transcript_json,
        transcript_pdf=session.base_request.transcript_pdf,
        min_credits=session.base_request.min_credits,
        max_credits=session.base_request.max_credits,
        target_credits=session.base_request.target_credits,
        planner_model=session.base_request.planner_model,
        intent_model=session.base_request.intent_model,
        include_required=session.base_request.include_required,
        max_repair_iters=session.base_request.max_repair_iters,
        preferred_subjects=session.base_request.preferred_subjects,
    )

    term_plan = run_plan(req)

    # Cache state from the result so swap/drop/add can work without re-running
    session.current_plan = term_plan
    # Refresh candidates cache from the retriever (stored on session after first plan)
    session._last_candidates = session._candidates_cache

    return _format_plan_response(term_plan, session.base_request.target_term)


def handle_swap(session: "PlannerSession", intent: IntentClassification) -> str:
    """Replace one course in the current plan with a better alternative."""
    if not session.current_plan:
        return "No plan yet — ask me to plan your semester first!"

    # 1. Resolve the course to remove
    remove_code = _resolve_code(session, intent.swap_remove)
    if not remove_code or remove_code not in session.current_plan.plan:
        return (
            f"I couldn't identify \"{intent.swap_remove}\" in the current plan.\n"
            f"Current plan: {', '.join(session.current_plan.plan)}\n"
            "Please specify the exact course code."
        )

    # 2. Resolve the replacement
    add_text = intent.swap_add_request.strip()
    add_code = _resolve_code(session, add_text) if add_text else None

    if add_code and add_code in session.current_plan.plan:
        return f"{add_code} is already in your plan!"

    if not add_code:
        # Retrieve a replacement semantically
        query = add_text if add_text else f"alternative to {remove_code}"
        current_set = set(session.current_plan.plan) - {remove_code}
        from scheduler.retriever import RetrieverFilter
        filt = RetrieverFilter(
            candidate_pool=session._eligible_pool,
            active_only=True,
        )
        hits = session.retriever.retrieve(query, k=5, filt=filt, rerank=True)
        candidates = [h for h in hits if h.code not in current_set]
        if not candidates:
            return f"I couldn't find an eligible replacement for {remove_code}. Try specifying a course code directly."
        add_code = candidates[0].code

    # 3. Build and validate new plan
    new_plan = [add_code if c == remove_code else c for c in session.current_plan.plan]
    report = _validate_and_update(session, new_plan, f"Swapped {remove_code} → {add_code}")
    return report


def handle_drop(session: "PlannerSession", intent: IntentClassification) -> str:
    """Remove a course from the current plan."""
    if not session.current_plan:
        return "No plan yet — ask me to plan your semester first!"

    drop_code = _resolve_code(session, intent.drop_course)
    if not drop_code or drop_code not in session.current_plan.plan:
        return (
            f"I couldn't find \"{intent.drop_course}\" in the current plan.\n"
            f"Current plan: {', '.join(session.current_plan.plan)}"
        )

    new_plan = [c for c in session.current_plan.plan if c != drop_code]
    return _validate_and_update(session, new_plan, f"Dropped {drop_code}")


def handle_add(session: "PlannerSession", intent: IntentClassification) -> str:
    """Add a course to the current plan."""
    if not session.current_plan:
        return "No plan yet — ask me to plan your semester first!"

    add_text = intent.add_request.strip()
    if not add_text:
        return "What course would you like to add? Describe it or give me the code."

    current_set = set(session.current_plan.plan)

    # Try direct code resolution first
    add_code = _resolve_code(session, add_text)

    if not add_code:
        # Retrieve semantically
        from scheduler.retriever import RetrieverFilter
        filt = RetrieverFilter(
            candidate_pool=session._eligible_pool,
            active_only=True,
        )
        hits = session.retriever.retrieve(add_text, k=8, filt=filt, rerank=True)
        candidates = [h for h in hits if h.code not in current_set]
        if not candidates:
            return "I couldn't find an eligible course matching that description."
        add_code = candidates[0].code
    elif add_code in current_set:
        return f"{add_code} is already in your plan!"

    new_plan = session.current_plan.plan + [add_code]
    return _validate_and_update(session, new_plan, f"Added {add_code}")


def handle_explain(session: "PlannerSession", intent: IntentClassification) -> str:
    """Return the reasoning for a specific course or the whole plan."""
    if not session.current_plan:
        return "No plan yet — ask me to plan your semester first!"

    course_text = intent.explain_course.strip()

    if not course_text:
        # Explain the whole plan
        lines = [f"**Plan for {len(session.current_plan.plan)} courses "
                 f"({session.current_plan.total_credits:.1f} credits):**\n"]
        lines.append(session.current_plan.summary)
        lines.append("\n**Per-course reasoning:**")
        for code in session.current_plan.plan:
            reason = session.current_plan.reasoning.get(code, "(no reasoning stored)")
            lines.append(f"\n• **{code}** — {reason}")
        if session.current_plan.alternatives:
            lines.append(f"\n\n*Alternatives considered but not selected: "
                         f"{', '.join(session.current_plan.alternatives)}*")
        return "\n".join(lines)

    # Explain a specific course
    code = _resolve_code(session, course_text)
    if not code:
        return f"I couldn't identify \"{course_text}\" — did you mean a course in the current plan?"

    reason = session.current_plan.reasoning.get(code)
    if reason:
        return f"**{code}**: {reason}"

    # Course is in catalog but reasoning not stored (shouldn't normally happen)
    hit = session.retriever.get_course(code)
    if hit:
        return (
            f"**{code} — {hit.title}** ({hit.su_credit} credits)\n"
            f"{hit.description[:400]}…"
        )
    return f"No information found for {code}."


def handle_unknown(session: "PlannerSession", _intent: IntentClassification) -> str:
    """Catch-all handler when intent is unclear."""
    ctx = ""
    if session.current_plan:
        ctx = f"\nCurrent plan: {', '.join(session.current_plan.plan)}"
    return (
        "I didn't quite catch that. Here's what you can ask me:\n"
        '  • **Plan**: "plan my semester, I want ML and OS"\n'
        '  • **Swap**: "swap CS 406 for something about NLP"\n'
        '  • **Drop**: "drop PHYS 113"\n'
        '  • **Add**: "add a databases course"\n'
        '  • **Explain**: "why did you pick CS 306?" or "explain the plan"'
        + ctx
    )


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _validate_and_update(session: "PlannerSession", new_plan: list[str], action: str) -> str:
    """Validate *new_plan*, update session if ok, return formatted response."""
    graph = session._graph
    student = session._student

    from scheduler.eligibility import validate_plan
    report = validate_plan(graph, student, new_plan)

    if report.ok:
        # Update the session's current plan in-place
        session.current_plan.plan = new_plan
        session.current_plan.total_credits = report.total_credits
        session.current_plan.validation_ok = True
        session.current_plan.violations = []
        session.current_plan.auto_added_coreqs = report.auto_added_coreqs

        lines = [f"✅ {action} — plan updated.\n"]
        lines.append(f"**Updated plan** ({len(new_plan)} courses, {report.total_credits:.1f} credits):")
        for code in new_plan:
            lines.append(f"  • {code}")
        if report.auto_added_coreqs:
            lines.append(f"\n⚡ Auto-added co-reqs: {', '.join(report.auto_added_coreqs)}")
        return "\n".join(lines)
    else:
        viol_lines = [f"  [{v.kind}] {v.message}" for v in report.violations]
        return (
            f"⚠️  {action} would create violations:\n"
            + "\n".join(viol_lines)
            + "\nPlan was NOT changed."
        )


def _format_plan_response(term_plan, target_term: str) -> str:
    from scheduler.prompts import term_label
    lines = [f"📅 **{term_label(target_term)} — Proposed Plan**\n"]
    lines.append(f"{len(term_plan.plan)} courses · {term_plan.total_credits:.1f} credits\n")
    for code in term_plan.plan:
        reason = term_plan.reasoning.get(code, "")
        lines.append(f"• **{code}** — {reason}" if reason else f"• {code}")
    lines.append(f"\n{term_plan.summary}")
    if term_plan.auto_added_coreqs:
        lines.append(f"\n⚡ Auto-added co-reqs: {', '.join(term_plan.auto_added_coreqs)}")
    if not term_plan.validation_ok:
        lines.append(f"\n⚠️  Validation issues: "
                     + "; ".join(v["message"] for v in term_plan.violations))
    if term_plan.warnings:
        lines.append("\n" + "\n".join(f"💬 {w}" for w in term_plan.warnings))
    return "\n".join(lines)
