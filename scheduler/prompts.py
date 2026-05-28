"""
Prompt templates for the SuSchedule-r planner agent.

All system prompts are defined as module-level constants so they sit at the
top of the message list and benefit from OpenAI's automatic prefix caching
(prompts > 1024 tokens are cached; the planner system prompt comfortably
exceeds that threshold).

Format functions accept structured Python objects and return formatted strings
ready to paste into the ``content`` field of a message dict.
"""
from __future__ import annotations

from scheduler.requirements import RequirementsReport
from scheduler.retriever import RetrievalResult

# --------------------------------------------------------------------------- #
# Term code → human-readable label
# --------------------------------------------------------------------------- #

_SEASON = {"01": "Fall", "02": "Spring", "03": "Summer"}

def term_label(term_code: str) -> str:
    """'202601' → 'Fall 2026-2027'"""
    year = int(term_code[:4])
    suffix = term_code[4:]
    season = _SEASON.get(suffix, "?")
    return f"{season} {year}-{year + 1}"


# --------------------------------------------------------------------------- #
# INTENT PARSER  (gpt-4o-mini, ~50 tokens in/out)
# --------------------------------------------------------------------------- #

INTENT_PARSE_SYSTEM = """\
You are a course-planning assistant for Sabancı University.

Your only job is to split the student's free-text semester request into a
list of 1–4 short, distinct retrieval queries that can each be matched
against course descriptions individually.

Rules:
- Each query must be a single, focused concept (e.g. "machine learning",
  "introductory economics", "easy humanities elective").
- Do NOT include scheduling constraints ("morning classes", "no Friday",
  "light workload") — those are handled elsewhere.
- Do NOT repeat the same concept twice.
- If the request is empty or vague, return ["required courses for my program"].
"""

def format_intent_user(user_request: str) -> str:
    return (
        f"Student's request:\n\"{user_request}\"\n\n"
        "Return a JSON list of retrieval queries."
    )


# --------------------------------------------------------------------------- #
# MAIN PLANNER  (gpt-4o, ~800–2000 tokens in, ~300 out)
# --------------------------------------------------------------------------- #

PLANNER_SYSTEM = """\
You are an expert academic advisor at Sabancı University (SU), helping \
undergraduate students build a feasible, graduation-aligned semester plan.

════════════════════════════════════════
HARD RULES — NEVER BREAK THESE
════════════════════════════════════════
1. SELECT ONLY from the "Candidate Courses" list provided in the user message.
   Never invent or hallucinate course codes.
2. Total SU credits must fall within the stated [min, max] range.
3. Courses marked [REQUIRED] are graduation requirements the student still
   needs — always include at least one unless the student explicitly asks not to.
4. Never select a course the student has already completed or is currently
   enrolled in (the candidate list is already filtered, but double-check).
5. Return valid JSON matching the output schema exactly. No extra keys.

════════════════════════════════════════
SOFT GUIDELINES (use judgment)
════════════════════════════════════════
- Aim for 5–6 courses. Fewer is fine if credit targets are met.
- Don't put more than 3 courses from the same subject in one semester.
- Prefer courses that unlock more downstream prerequisites (higher "prereq fan-out").
- If the student has stated interests, match them as closely as the constraint
  space allows — but don't sacrifice required courses to do so.
- 0-credit courses (like ENS 491 Graduation Project Design) don't count
  toward the credit target; include them only if they appear in the candidate
  list and the student still needs them.
- When two courses are equally good, prefer the one with the higher
  `rerank` score shown in the candidate list.

════════════════════════════════════════
OUTPUT SCHEMA (strict JSON)
════════════════════════════════════════
{
  "plan": ["CS 306", "CS 415", ...],
  "reasoning": [
    {"code": "CS 306", "reason": "Required course; enables CS 406 next term."},
    ...
  ],
  "summary": "Balanced CS-heavy semester...",
  "alternatives_considered": ["CS 400", ...]
}
- `plan` and `reasoning` must have the same length and order.
- `alternatives_considered` may be an empty list.
"""


def format_planner_user(
    name: str,
    program: str,
    semester: int,
    cgpa: float,
    completed: list[str],
    in_progress: list[str],
    minors: list[str],
    remaining: RequirementsReport,
    candidates: list[RetrievalResult],
    required_injected: list[str],
    user_request: str,
    target_term: str,
    min_credits: float,
    max_credits: float,
    target_credits: float,
) -> str:
    lines: list[str] = []

    # --- Student profile ---
    lines.append("## Student Profile")
    lines.append(
        f"Name: {name} | Program: {program} | "
        f"Semester: {semester} | CGPA: {cgpa:.2f}"
    )
    if minors:
        lines.append(f"Minors declared: {', '.join(minors)}")
    lines.append(f"Completed ({len(completed)} courses): {', '.join(sorted(completed))}")
    if in_progress:
        lines.append(f"Currently enrolled: {', '.join(sorted(in_progress))}")
    lines.append("")

    # --- Graduation requirements ---
    lines.append("## Graduation Requirements — Still Needed")
    for line in remaining.summary_lines():
        lines.append(line)
    lines.append("")

    # --- Target term ---
    lines.append("## Target Term")
    lines.append(
        f"{term_label(target_term)} (code: {target_term}) | "
        f"Target credits: {target_credits} (range {min_credits}–{max_credits})"
    )
    lines.append("")

    # --- Student request ---
    lines.append("## Student's Request")
    req_text = user_request.strip() if user_request.strip() else "(none — default to required courses)"
    lines.append(f'"{req_text}"')
    lines.append("")

    # --- Candidate list ---
    lines.append("## Candidate Courses")
    lines.append(
        "(Pre-filtered: eligible + likely offered this term + semantically relevant)\n"
    )

    # Build set for quick lookup
    required_set = set(required_injected)
    seen = set()
    idx = 1
    for hit in candidates:
        if hit.code in seen:
            continue
        seen.add(hit.code)

        tags = []
        if hit.code in required_set:
            tags.append("[REQUIRED]")
        score_parts = [f"bienc={hit.score:.2f}"]
        if hit.reranker_score is not None:
            score_parts.append(f"rerank={hit.reranker_score:.2f}")
        tags.append(f"[{', '.join(score_parts)}]")

        prereq = f"Prereq: {hit.prereq_text}" if hit.has_prereq else "No prereq"
        seasons = ", ".join(hit.seasons_offered) if hit.seasons_offered else "?"
        desc_snip = hit.description[:200].rstrip()
        if len(hit.description) > 200:
            desc_snip += "…"

        lines.append(
            f"{idx}. {hit.code} — {hit.title} {' '.join(tags)}\n"
            f"   Credits: {hit.su_credit} | {prereq} | Offered: {seasons}\n"
            f"   {desc_snip}\n"
        )
        idx += 1

    # --- Task ---
    lines.append("## Your Task")
    lines.append(
        f"Propose a semester plan totalling {min_credits}–{max_credits} credits "
        f"(target: {target_credits}). "
        "Select ONLY from the candidates above. "
        "Return JSON matching the output schema."
    )

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# REPAIR PROMPT  (gpt-4o, fed violations back to LLM)
# --------------------------------------------------------------------------- #

REPAIR_SYSTEM = """\
You are an academic advisor at Sabancı University.

The plan you previously proposed failed validation. You will receive:
1. The rejected plan.
2. A list of specific violations (prereq failures, credit overloads, etc.).
3. The original candidate course list.

Your job is to produce a corrected plan that:
- Fixes EVERY listed violation.
- Still satisfies the original credit constraints.
- Still selects ONLY from the provided candidate list.
- Still respects the student's stated interests as much as possible.

Return valid JSON matching the RepairOutput schema.
"""

def format_repair_user(
    previous_plan: list[str],
    violations: list[dict],
    candidates: list[RetrievalResult],
    min_credits: float,
    max_credits: float,
) -> str:
    lines: list[str] = []

    lines.append("## Rejected Plan")
    lines.append(", ".join(previous_plan))
    lines.append("")

    lines.append("## Violations to Fix")
    for v in violations:
        lines.append(f"- [{v.get('kind', 'error')}] {v.get('message', '')}")
    lines.append("")

    lines.append("## Available Candidates (same as before)")
    seen = set()
    idx = 1
    for hit in candidates:
        if hit.code in seen:
            continue
        seen.add(hit.code)
        prereq = f"Prereq: {hit.prereq_text}" if hit.has_prereq else "No prereq"
        lines.append(
            f"{idx}. {hit.code} — {hit.title} | Credits: {hit.su_credit} | {prereq}"
        )
        idx += 1
    lines.append("")

    lines.append(
        f"Produce a corrected plan within {min_credits}–{max_credits} credits. "
        "Return JSON matching the RepairOutput schema."
    )
    return "\n".join(lines)
