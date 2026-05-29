"""SuSchedule-r ReAct course-advising agent."""
from __future__ import annotations

import json
import re
from datetime import date
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from scheduler import llm_client
from scheduler.eligibility import (
    EligibilityOptions,
    prereqs_satisfied,
    validate_plan as elig_validate_plan,
)
from scheduler.offerings import likely_offered_in, offered_history
from scheduler.schemas import TermPlan

if TYPE_CHECKING:
    from scheduler.session import PlannerSession


REACT_SYSTEM = """\
You are an academic-advising agent for Sabancı University (SU). You help
students explore courses, understand degree requirements, inspect prereqs and
offering patterns, and plan semesters only when the user explicitly asks.

Use tools for facts. Do not invent course codes, prereqs, credits, offerings,
degree requirements, or eligibility.

Available tools
---------------
- get_student_profile()        : current known student context; may be partial
- set_student_context()        : store degree/admit/completed info user provided
- select_degree_graphs()       : selected program/cohort graph summary
- get_requirement_state()      : what is still needed for graduation
- get_degree_section_courses() : RAG-rank courses inside required/core/area/free graph sections
- get_current_plan()           : current committed plan
- rewrite_retrieval_queries()  : rewrite latest user wording for retrieval only
- retrieve_catalog_courses()   : RAG search over course descriptions with filters
- get_course_details()         : exact catalog details for course codes
- get_eligible_courses()       : eligible scoped courses when transcript exists
- check_prereqs(code)          : prereq/coreq text + eligibility verdict
- get_offering_pattern(code)   : historical offering pattern
- validate_plan(plan)          : validate a candidate plan
- validate_courses(codes)      : check courses individually
- set_plan(plan, reasoning, summary)
- list_minors()                : all available SU undergraduate minors
- get_minor_requirements(minor): a minor's courses + the student's progress
- get_science_engineering_progress(): Engineering & Basic-Science ECTS gap toward graduation
- find_courses_by_credit_type(): courses that actually carry Basic-Science / Engineering ECTS
- build_timetable(courses)     : conflict-free weekly schedule from real section meeting times
- get_current_schedule()       : the courses/times the student assembled in the UI builder
- check_courses_against_current_schedule(codes): test candidate courses against the builder schedule

Workflow heuristics
-------------------
1. If the user asks about exact course codes, call get_course_details.
2. For course-content, focus-area, comparison, or catalog-exploration
   questions, use rewrite_retrieval_queries then retrieve_catalog_courses.
3. If the user names a section ("Required", "Core Elective", "Area Elective",
   "Free Elective"), do not retrieve from the whole catalog. Select the
   degree/cohort graph, restrict to that section's in-slice courses, then rank
   those courses by catalog-description relevance.
4. For degree requirement questions, call select_degree_graphs and
   get_degree_section_courses/get_requirement_state. If the user asks about a
   focus such as supply chain, distinguish official degree requirements from
   focus-relevant electives.
   When get_requirement_state returns a 'requirement' / 'section_credit_status'
   block, that is AUTHORITATIVE (from the degree evaluation): answer "how many
   do I still need" from its su_credits_remaining and satisfied flag. The
   *_left / 'options' lists are only the pool of eligible courses, NOT
   outstanding requirements — a section can be fully satisfied while its pool is
   non-empty, so never count the pool as courses the student still needs and
   never invent a required/completed course count. If the section is satisfied,
   say so plainly. If a section has no credit status (completion_unknown), say
   the required amount is unknown rather than guessing.
5. For IE operations/supply-chain/logistics/production questions, use the BSIE
   scoped graph and prefer Required + Core Elective IE courses first.
6. Answer the immediate question first. Offer 2-4 concrete next actions when useful.
7. Do not create, validate, or commit a semester plan unless the user explicitly
   asks for planning/checking or provides courses to validate.
8. Never call set_plan unless validate_plan returned ok=true.
9. Treat messages labeled [runtime environment], [student context loaded], and
   [retrieved recommendation context] as authoritative runtime context. If
   get_student_profile or [student context loaded] says transcript_loaded=true,
   use that parsed academic profile as authoritative and do not ask again for
   degree/program, admit term, completed courses, in-progress courses, transcript,
   or degree evaluation.
10. If degree/admit/transcript is missing, say exactly what is missing. You may
   still answer general RAG questions, but do not claim exact takeability.
11. For minor questions ("can I minor in X", "what does the finance minor need",
    "how close am I to a math minor", "how many more courses/credits do I need"),
    call get_minor_requirements (or list_minors to show options). Answer counts
    from the returned fields: per section use min_courses / min_su_credits and
    courses_remaining / su_credits_remaining; for the whole minor use
    required_total and the top-level courses_remaining / su_credits_remaining.
    Each elective section requires only the listed min_courses from its
    'options' pool, NOT every option — never imply all options are mandatory.
    Some minors are credit-based (min_courses null); answer those in SU credits.
    Minor progress is tracked ONLY in SU credits and course counts. NEVER
    convert between SU credits and ECTS or estimate completed/remaining ECTS.
    The only ECTS value available is total_ects_whole_minor (the whole minor's
    grand total); if asked about ECTS, state that number as-is and say the
    per-course ECTS breakdown isn't tracked, so completed ECTS can't be given.
12. For science/engineering credit questions ("do I have enough engineering
    credits", "how many basic science ECTS do I still need"), call
    get_science_engineering_progress. When recommending or validating a plan for
    an engineering degree, check this: if Engineering or Basic-Science ECTS are
    still short, prefer courses that fill the gap and pass the candidate codes to
    the tool to compare their contributions.
    For "suggest/list courses WITH science (or engineering) credits" (optionally
    by subject, e.g. "CS courses with science credits"), call
    find_courses_by_credit_type — NEVER answer this from catalog retrieval and
    NEVER treat a course's SU credits as science/engineering credits. Only list
    courses the tool returns, and report each course's basic_sci_ects / eng_ects
    (a course can have 0 science ECTS even though it has SU credits). If a
    subject has few or no such courses, say so honestly.
13. For scheduling/time questions ("what time is CS 412", "do these courses
    clash", "build/show me a weekly schedule", "is this plan conflict-free"),
    call build_timetable with the course codes and answer ONLY from its result.
    NEVER invent or recall meeting days, times, classrooms, or CRNs from memory.
    A weekly grid is rendered for the user in the UI, so keep your text SHORT: a
    one-line verdict (conflict-free or which pairs clash) plus each course's day/
    time using the per-pick 'when' field — do not paste a full ASCII grid.
    Report only the days/times the tool returns; if 'crn' is null, do NOT state a
    CRN or section number (they apply only to a past proxy term). If the tool
    reports proxy_term_used, add one sentence that the times come from the most
    recent same-season term because the target term isn't published yet and may
    shift. If a course is in 'missing' (not offered) or 'no_meetings' (TBA), say
    so plainly. If ok is false, state no conflict-free combination exists and
    name the clashing courses from 'reason'.
14. When the user asks you to check/review "my schedule" or "my current
    schedule" (e.g. via the Check Schedule button), call get_current_schedule.
    Comment on: time conflicts (from has_conflicts/conflicts), total SU credit
    load vs the min/max range, and — if a transcript is loaded — how the picked
    courses fit remaining degree requirements (cross-reference get_requirement_state
    / required-left). Be concise and concrete; do not invent meeting times beyond
    what the tool returns. If has_schedule is false, tell the user to add courses
    to the builder on the right first.
    Recitation/lab codes returned by the schedule tool (for example CS 412R or
    CS 308L) are valid offering rows even if they are absent from the catalog;
    use the returned title and never label them UNKNOWN_COURSE_CODE.
15. When the user asks whether "this/these course(s)" conflict with their current
    schedule, call check_courses_against_current_schedule with the explicit
    candidate course codes from the latest recommendation or user message. Do
    not use build_timetable for this case because the current builder selections
    must stay fixed.

Answer style
------------
1. Synthesize; do not mirror database rows.
2. For a single-course question like "What does CS 412 cover?", answer in 1-2
   short paragraphs. Do not list fields like "Credits:" or "Prerequisites:"
   unless the user asks for logistics.
3. For multi-course recommendations, group by why each course fits the stated
   interest. Prefer 3-6 strong courses over a long list.
4. End with one specific next step when helpful. Do not mention rewritten queries.
"""


class QueryRewrite(BaseModel):
    queries: list[str] = Field(description="1-5 concise retrieval queries.")


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
    _tool("get_student_profile", "Return known student context.", {}),
    _tool("set_student_context", "Store degree/admit/completed/in-progress info from chat.", {
        "program": {"type": "string"},
        "admit_term": {"type": "string"},
        "completed": {"type": "array", "items": {"type": "string"}},
        "in_progress": {"type": "array", "items": {"type": "string"}},
    }),
    _tool("select_degree_graphs", "Return selected scoped degree/cohort graph summary.", {
        "sections": {"type": "array", "items": {"type": "string"}},
    }),
    _tool("get_requirement_state", "Return remaining degree requirements.", {
        "section": {"type": "string"},
    }),
    _tool("get_degree_section_courses", "Return courses from selected graph section. RAG-ranked by topic when a query is given; otherwise returned alphabetically.", {
        "section": {"type": "string"},
        "query": {"type": "string"},
        "k": {"type": "integer", "minimum": 1, "maximum": 80},
    }),
    _tool("get_current_plan", "Return current committed plan.", {}),
    _tool("rewrite_retrieval_queries", "Rewrite user message into retrieval queries; internal only.", {
        "original_message": {"type": "string"},
    }, ["original_message"]),
    _tool("retrieve_catalog_courses", "RAG search over catalog descriptions. degree_filter=true (default) scopes to student's degree requirements; degree_filter=false searches the full catalog for open exploration.", {
        "queries": {"type": "array", "items": {"type": "string"}},
        "k": {"type": "integer", "minimum": 1, "maximum": 15},
        "subj": {"type": "array", "items": {"type": "string"}},
        "degree_filter": {"type": "boolean"},
        "section": {"type": "string"},
        "eligible_only": {"type": "boolean"},
    }, ["queries"]),
    _tool("get_course_details", "Return exact catalog details for course codes.", {
        "codes": {"type": "array", "items": {"type": "string"}},
    }, ["codes"]),
    _tool("get_eligible_courses", "Return eligible scoped courses.", {
        "section": {"type": "string"},
        "k": {"type": "integer", "minimum": 1, "maximum": 50},
    }),
    _tool("check_prereqs", "Inspect prereqs/coreqs and eligibility for a course.", {
        "code": {"type": "string"},
    }, ["code"]),
    _tool("get_offering_pattern", "Return historical offering pattern for a course.", {
        "code": {"type": "string"},
    }, ["code"]),
    _tool("validate_plan", "Validate a candidate plan.", {
        "plan": {"type": "array", "items": {"type": "string"}},
    }, ["plan"]),
    _tool("validate_courses", "Check a list of courses individually.", {
        "codes": {"type": "array", "items": {"type": "string"}},
    }, ["codes"]),
    _tool("set_plan", "Commit a validated plan only after explicit user request.", {
        "plan": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "object", "additionalProperties": {"type": "string"}},
        "summary": {"type": "string"},
    }, ["plan", "reasoning", "summary"]),
    _tool("list_minors", "List all available SU undergraduate minors (code + name).", {}),
    _tool("get_minor_requirements", "Return a minor's required/elective courses and, when a transcript is loaded, the student's progress toward it. Accepts a code ('FIN-MINOR'), short code ('FIN'), or name ('finance').", {
        "minor": {"type": "string"},
    }, ["minor"]),
    _tool("get_science_engineering_progress", "Return Engineering and Basic-Science ECTS progress toward graduation (minimum required, completed, remaining). Engineering degrees must satisfy both. Optionally pass candidate course codes to see how much Engineering/Basic-Science ECTS each would contribute.", {
        "courses": {"type": "array", "items": {"type": "string"}},
    }),
    _tool("find_courses_by_credit_type", "Find courses that actually carry Basic-Science or Engineering ECTS credit, using the per-course credit table. Use this for any 'courses with science/engineering credits' request — do NOT infer credit type from the catalog. credit_type is 'basic_science' or 'engineering'; subject optionally restricts to a code prefix like 'CS'.", {
        "credit_type": {"type": "string", "enum": ["basic_science", "engineering"]},
        "subject": {"type": "string"},
        "limit": {"type": "integer"},
    }, ["credit_type"]),
    _tool("build_timetable", "Build a conflict-free weekly timetable (or report clashes) for a set of courses, using real per-section meeting times. The tool auto-adds direct catalog corequisites such as recitations/labs and returns auto_added_coreqs. Pass the lecture course codes to schedule. Returns chosen sections (CRN, day/time, location, instructor) when a clash-free combination exists, or the conflicting pairs when none does. Also flags courses not offered or with no scheduled meeting time. Use this for any 'do these courses clash', 'build/show my schedule', or 'what time is course X' request.", {
        "courses": {"type": "array", "items": {"type": "string"}},
    }, ["courses"]),
    _tool("get_current_schedule", "Return the schedule the student has assembled in the UI schedule builder: the picked courses, their meeting times, total SU credits, and any time conflicts between them. Recitation/lab rows such as CS 412R or CS 308L may not exist in the catalog; use the title returned by this tool and never call them UNKNOWN_COURSE_CODE. Use this whenever the user asks to review, check, or comment on 'my schedule' / 'my current schedule'.", {}),
    _tool("check_courses_against_current_schedule", "Deterministically test candidate courses against the student's current UI builder schedule. Keeps existing selected sections fixed, auto-adds direct catalog recitations/labs for the candidate courses, searches candidate section choices, and returns whether the candidates can be added without introducing new time conflicts. Use this for questions like 'do these courses conflict with my current schedule?', 'can I add CS 405?', or 'check this course against my schedule'.", {
        "codes": {"type": "array", "items": {"type": "string"}},
    }, ["codes"]),
]


def _fallback_queries(text: str) -> list[str]:
    base = text.strip() or "course recommendations"
    low = base.lower()
    queries = [base]
    expansions = {
        "llm": ["natural language processing", "deep learning", "machine learning"],
        "large language model": ["natural language processing", "deep learning"],
        "theoretical": ["algorithms", "computational complexity", "theory of computation"],
        "theory": ["algorithms", "computational complexity", "theory of computation"],
        "optimization": ["industrial engineering optimization", "operations research", "linear programming", "integer programming"],
        "operations research": ["linear programming", "integer programming", "decision analysis"],
        "operational": ["industrial engineering operations management", "production and service systems operations", "supply chain analysis", "logistics systems", "quality planning and control"],
        "operations": ["industrial engineering operations management", "production and service systems operations", "supply chain analysis", "logistics systems", "quality planning and control"],
        "supply chain": ["supply chain analysis", "logistics systems planning and design", "inventory replenishment", "production planning", "operations management"],
        "logistics": ["logistics systems planning and design", "supply chain analysis"],
        "ai": ["artificial intelligence", "machine learning"],
        "ml": ["machine learning", "deep learning"],
        "systems": ["operating systems", "distributed systems", "computer networks"],
        "networking": ["computer networks", "network science", "communication systems"],
    }
    for key, vals in expansions.items():
        if key in low:
            queries.extend(vals)
    out: list[str] = []
    for query in queries:
        if query not in out:
            out.append(query)
    return out[:5]


def _planning_requested(text: str) -> bool:
    return bool(re.search(r"\b(plan|schedule|semester plan|term plan|two terms|next term|next semester)\b", text.lower()))


def _exclude_current_schedule_requested(text: str) -> bool:
    return bool(re.search(
        r"\b(other|another|alternative|instead|swap|replace|resolve|avoid|not in my schedule)\b",
        text.lower(),
    ))


def _course_recommendation_requested(text: str) -> bool:
    low = text.lower()
    asks_for_courses = re.search(
        r"\b(suggest|recommend|recommendation|which courses|what courses|"
        r"what should i take|should i take|courses? should i take|"
        r"learn|knowledge|focus|interested|interests?|about)\b",
        low,
    )
    scoped_by_topic_or_term = re.search(
        r"\b(networks?|networking|security|cybersecurity|systems?|"
        r"ai|artificial intelligence|machine learning|ml|deep learning|"
        r"llm|nlp|theory|theoretical|algorithms?|database|data science|"
        r"optimization|operations?|supply chain|logistics|core|area|"
        r"electives?|next semester|next term|fall\s*20\d{2}|"
        r"spring\s*20\d{2}|summer\s*20\d{2}|graduation|graduate)\b",
        low,
    )
    return bool(asks_for_courses and scoped_by_topic_or_term)


def _reasks_for_known_profile(text: str) -> bool:
    low = text.lower()
    asks = re.search(r"\b(need|provide|confirm|tell me|let me know|could you)\b", low)
    profile_fields = re.search(
        r"\b(program|degree|major|admit term|admission|cohort|started|completed courses|in-progress|in progress|transcript|degree evaluation)\b",
        low,
    )
    return bool(asks and profile_fields)


_ANSWER_COURSE_LINE_RE = re.compile(
    r"(?m)(?P<code>[A-Z]{2,5}\s*\d{3,5}[A-Z]?)\s*(?:[-:–—])\s*(?P<title>[^\n\r*`#]+)"
)


def _norm_title(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"[*_`~]", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.split(r"\s+(?:is|covers|examines|explores|focuses|will|would|can)\b", text, maxsplit=1)[0]
    return re.sub(r"[^a-z0-9 ]+", "", text).strip()


def _runtime_course_title(session: "PlannerSession", code: str) -> str | None:
    """Return a title for non-catalog schedule rows such as recitations/labs."""
    normalized = code.upper().strip()
    for pick in session.user_schedule or []:
        if str(pick.get("code", "")).upper().strip() == normalized and pick.get("title"):
            return str(pick["title"])

    offerings = session._offerings or {}
    for term in sorted(offerings.keys(), reverse=True):
        sections = offerings.get(term, {}).get(normalized, [])
        for section in sections:
            title = section.get("title")
            if title:
                return str(title)
    return None


def _catalog_title_mismatches(session: "PlannerSession", text: str) -> list[dict[str, str]]:
    mismatches: list[dict[str, str]] = []
    if not text.strip():
        return mismatches
    catalog = session.catalog
    seen: set[tuple[str, str]] = set()
    for match in _ANSWER_COURSE_LINE_RE.finditer(text):
        code = re.sub(r"\s+", " ", match.group("code").upper().strip())
        stated = match.group("title").strip(" .:-")
        key = (code, stated)
        if key in seen:
            continue
        seen.add(key)
        course = catalog.get(code)
        if course is None:
            expected = _runtime_course_title(session, code)
            if expected:
                stated_norm = _norm_title(stated)
                expected_norm = _norm_title(expected)
                ratio = SequenceMatcher(None, stated_norm, expected_norm).ratio() if expected_norm else 0.0
                if stated_norm and expected_norm and expected_norm not in stated_norm and ratio < 0.72:
                    mismatches.append({"code": code, "stated": stated, "expected": expected})
            elif not code.endswith(("R", "L")):
                mismatches.append({"code": code, "stated": stated, "expected": ""})
            continue
        stated_norm = _norm_title(stated)
        expected_norm = _norm_title(course.title)
        if not stated_norm:
            continue
        if expected_norm and expected_norm in stated_norm:
            continue
        ratio = SequenceMatcher(None, stated_norm, expected_norm).ratio() if expected_norm else 0.0
        if ratio < 0.72:
            mismatches.append({"code": code, "stated": stated, "expected": course.title})
    return mismatches


def _catalog_mismatch_message(mismatches: list[dict[str, str]]) -> str:
    lines = [
        "Revise the previous answer. It contained course-code/title mismatches.",
        "Use these exact titles and remove any unsupported course names:",
    ]
    for item in mismatches[:12]:
        if item.get("expected"):
            lines.append(f"- {item['code']}: stated `{item['stated']}`, exact title `{item['expected']}`")
        else:
            lines.append(f"- {item['code']}: stated `{item['stated']}`, but this code is not present in catalog or current schedule context. Remove it unless a tool result supports it.")
    lines.append("Do not invent course titles. If a relevant course is blocked, completed, currently in progress, or only adjacent to the topic, say so explicitly.")
    return "\n".join(lines)


def _compact_codes(codes: set[str] | list[str] | tuple[str, ...], limit: int = 24) -> str:
    ordered = sorted(str(c).upper().strip() for c in codes if str(c).strip())
    if not ordered:
        return "none"
    shown = ordered[:limit]
    suffix = f", ... (+{len(ordered) - limit} more)" if len(ordered) > limit else ""
    return ", ".join(shown) + suffix


def _current_schedule_codes(session: "PlannerSession") -> set[str]:
    return {
        str(p.get("code", "")).upper().strip()
        for p in session.user_schedule or []
        if str(p.get("code", "")).strip()
    }


def _meeting_time_label(meeting: dict) -> str:
    start = meeting.get("start_label")
    end = meeting.get("end_label")
    if start and end:
        return f"{start}-{end}"
    if "start" in meeting and "end" in meeting:
        return f"{meeting['start']}-{meeting['end']}"
    return ""


def _conflict_dicts(conflicts: list[tuple[str, str, dict, dict]]) -> list[dict]:
    rows = []
    for a, b, ma, mb in conflicts:
        rows.append({
            "between": [a, b],
            "day": ma.get("days_raw") or "/".join(ma.get("day_labels", [])),
            "times": f"{_meeting_time_label(ma)} vs {_meeting_time_label(mb)}",
        })
    return rows


# NOTE: convenience pre-pass only; the LLM's set_student_context tool is the
# authoritative way to update degree context. This regex must NOT fire on
# casual phrasing like "tell me…" — so bare 2-letter codes are excluded.
# Short codes require an explicit degree-marker word to avoid false matches.
_PROGRAM_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Full program names — always safe
    (re.compile(r"\bindustrial\s+engineering\b", re.I), "BSIE-DM"),
    (re.compile(r"\bcomputer\s+science\b", re.I), "BSCS-DM"),
    (re.compile(r"\belectronics?\s+engineering\b|\belectrical\s+engineering\b", re.I), "BSEE-DM"),
    (re.compile(r"\bdata\s+science\b", re.I), "BSDSA-DM"),
    (re.compile(r"\bmechatronics\b", re.I), "BSME-DM"),
    (re.compile(r"\bbusiness\s+administration\b|\bmanagement\s+engineering\b", re.I), "BAMAN-DM"),
    (re.compile(r"\beconomics\b", re.I), "BAECON-DM"),
    (re.compile(r"\bpsychology\b", re.I), "BAPSY-DM"),
]

# Short codes (ie/cs/ee/me/dsa) only when paired with a degree-marker word
# to prevent matching everyday tokens ("tell me…", "access control", "cs" in urls).
_SHORT_CODE_PATTERN = re.compile(
    r"\b(ie|cs|ee|me|dsa)\s+(?:major|student|degree|program|department)\b",
    re.I,
)
_SHORT_CODE_MAP: dict[str, str] = {
    "ie": "BSIE-DM", "cs": "BSCS-DM", "ee": "BSEE-DM",
    "me": "BSME-DM", "dsa": "BSDSA-DM",
}


def _infer_context_from_message(session: "PlannerSession", text: str) -> None:
    # Only call update_manual_context when confident — wrong inferences reset
    # the graph/eligibility state for the whole session.
    program = None
    for pattern, code in _PROGRAM_PATTERNS:
        if pattern.search(text):
            program = code
            break
    if program is None:
        sc_match = _SHORT_CODE_PATTERN.search(text)
        if sc_match:
            program = _SHORT_CODE_MAP.get(sc_match.group(1).lower())
    admit_term = None
    at_match = re.search(r"\b(?:admit(?:ted)?|entry|entrance|started|start|freshman|cohort)\D{0,20}(20\d{2})(?:\s*(fall|spring|summer))?\b", text, re.I)
    if at_match:
        year = at_match.group(1)
        season = (at_match.group(2) or "fall").lower()
        admit_term = year + {"fall": "01", "spring": "02", "summer": "03"}[season]
    if program or admit_term:
        session.update_manual_context(program=program, admit_term=admit_term)


def _is_ie_operations_query(text: str) -> bool:
    return any(term in text.lower() for term in (
        "operational", "operations", "operation", "supply chain", "logistics",
        "production", "inventory", "quality", "scheduling", "manufacturing",
        "service systems",
    ))


def _infer_section_from_text(text: str) -> str | None:
    low = text.lower().replace("_", " ")
    if re.search(r"\b(required|must take|mandatory)\b", low):
        return "Required"
    if re.search(r"\bcore\s+elective?s?\b|\bcore courses?\b", low):
        return "Core Elective"
    if re.search(r"\barea\s+elective?s?\b|\barea courses?\b", low):
        return "Area Elective"
    if re.search(r"\bfree\s+elective?s?\b|\bfree courses?\b", low):
        return "Free Elective"
    return None


def _normalize_section_name(name: str) -> str:
    aliases = {
        "required": "Required",
        "requirement": "Required",
        "requirements": "Required",
        "core": "Core Elective",
        "core elective": "Core Elective",
        "core electives": "Core Elective",
        "area": "Area Elective",
        "area elective": "Area Elective",
        "area electives": "Area Elective",
        "free": "Free Elective",
        "free elective": "Free Elective",
        "free electives": "Free Elective",
    }
    key = name.lower().replace("_", " ").strip()
    return aliases.get(key, name)


def _clean_section_query(text: str) -> str:
    low = text.lower()
    if "supply chain" in low:
        return "supply chain logistics operations production inventory"
    if any(term in low for term in ("operational", "operations", "operation")):
        return "operations management production service systems logistics quality"
    if "optimization" in low:
        return "industrial engineering optimization operations research"
    cleaned = re.sub(
        r"\b(required|mandatory|must take|core elective|core electives|area elective|area electives|free elective|free electives|courses?|course|industrial engineering|degree|find|show|about|for|in)\b",
        " ",
        text,
        flags=re.I,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or text


def _title_topic_boost(query: str, title: str) -> float:
    low_q = query.lower()
    low_t = title.lower()
    boost = 0.0
    phrase_boosts = {
        "deep learning": 3.0,
        "machine learning": 2.2,
        "natural language": 1.8,
        "large language": 1.8,
        "computer vision": 1.6,
        "artificial intelligence": 1.4,
        "neural": 1.2,
        "data science": 1.0,
    }
    for phrase, value in phrase_boosts.items():
        if phrase in low_q and phrase in low_t:
            boost += value
    if "supply chain" in low_q and "supply chain" in low_t:
        boost += 1.5
    if "logistics" in low_q and "logistics" in low_t:
        boost += 1.0
    if any(term in low_q for term in ("operations", "operational")) and "operations" in low_t:
        boost += 0.8
    if "production" in low_q and "production" in low_t:
        boost += 0.6
    if "quality" in low_q and "quality" in low_t:
        boost += 0.8
    if "optimization" in low_q and "optimization" in low_t:
        boost += 1.0
    if "operations research" in low_q and "operations research" in low_t:
        boost += 1.0
    return boost


def _focus_terms_for_queries(queries: list[str]) -> list[str]:
    joined = " ".join(queries).lower()
    groups = [
        (
            ("deep learning", "neural", "machine learning", "artificial intelligence", " ai ", "nlp", "natural language", "computer vision", "large language"),
            ["deep learning", "neural", "machine learning", "artificial intelligence", " ai ", "natural language", "computer vision", "large language", "nlp"],
        ),
        (
            ("network", "networking", "security", "cybersecurity", "cryptography"),
            ["network", "networking", "security", "cybersecurity", "cryptography", "tcp/ip", "firewall"],
        ),
        (
            ("optimization", "operations research", "linear programming", "integer programming"),
            ["optimization", "operations research", "linear programming", "integer programming", "decision analysis"],
        ),
        (
            ("supply chain", "logistics", "production", "inventory", "operations management"),
            ["supply chain", "logistics", "production", "inventory", "operations management", "quality"],
        ),
    ]
    padded = f" {joined} "
    for triggers, terms in groups:
        if any(trigger in padded for trigger in triggers):
            return terms
    return []


def _subject_filter_from_text(text: str) -> list[str] | None:
    matches = re.findall(r"\b([A-Z]{2,5})[-\s]*(?:CODED|COURSES?|SUBJECT)\b", text.upper())
    subjects = []
    for subj in matches:
        if subj not in {"THE", "FOR", "AND", "CORE", "AREA", "FREE"} and subj not in subjects:
            subjects.append(subj)
    return subjects or None


def _filter_rows_by_focus(rows: list[dict], queries: list[str]) -> list[dict]:
    terms = _focus_terms_for_queries(queries)
    if not terms:
        return rows
    filtered = []
    for row in rows:
        haystack = f" {row.get('code', '')} {row.get('title', '')} {row.get('description', '')} ".lower()
        if any(term in haystack for term in terms):
            filtered.append(row)
    return filtered or rows


def _apply_topic_boosts(rows: list[dict], query: str) -> list[dict]:
    boosted = []
    for row in rows:
        row = dict(row)
        boost = _title_topic_boost(query, str(row.get("title", "")))
        if boost:
            row["title_boost"] = round(boost, 3)
            row["score"] = float(row.get("score", 0) or 0) + boost
        boosted.append(row)
    boosted.sort(key=lambda r: (-float(r.get("score", 0) or 0), str(r.get("code", ""))))
    for i, row in enumerate(boosted, start=1):
        row["rank"] = i
    return boosted


def _compact_for_trace(value: Any, max_items: int = 25, max_string: int = 700) -> Any:
    if isinstance(value, str):
        return value if len(value) <= max_string else value[:max_string] + "..."
    if isinstance(value, list):
        compacted = [_compact_for_trace(v, max_items, max_string) for v in value[:max_items]]
        if len(value) > max_items:
            compacted.append({"truncated_items": len(value) - max_items})
        return compacted
    if isinstance(value, dict):
        return {str(k): _compact_for_trace(v, max_items, max_string) for k, v in value.items()}
    return value


def _trace(session: "PlannerSession", event: dict) -> None:
    compacted = _compact_for_trace(event)
    session.last_trace.append(compacted)
    if len(session.last_trace) > 120:
        session.last_trace = session.last_trace[-120:]
    session.last_raw_trace.append(event)
    if len(session.last_raw_trace) > 120:
        session.last_raw_trace = session.last_raw_trace[-120:]
    raw_callback = getattr(session, "trace_raw_callback", None)
    if raw_callback:
        try:
            raw_callback(event)
        except Exception:
            pass
    callback = getattr(session, "trace_callback", None)
    if callback:
        try:
            callback(compacted)
        except Exception:
            pass


def _prefetch_recommendation_context(
    session: "PlannerSession",
    toolbox: "Toolbox",
    user_message: str,
) -> str | None:
    if not _course_recommendation_requested(user_message):
        return None

    rewrite = toolbox.rewrite_retrieval_queries(user_message)
    queries = rewrite.get("queries") or _fallback_queries(user_message)
    _trace(session, {
        "type": "tool_result",
        "step": 0,
        "name": "rewrite_retrieval_queries",
        "arguments": {"original_message": user_message},
        "result": rewrite,
    })

    if session.student is not None:
        selected = toolbox.select_degree_graphs()
        _trace(session, {
            "type": "tool_result",
            "step": 0,
            "name": "select_degree_graphs",
            "arguments": {},
            "result": selected,
        })

    needs_requirements = bool(re.search(
        r"\b(graduat|requirement|core|area|free|elective|help my graduation|count toward)\b",
        user_message.lower(),
    ))
    requirement_state: dict | None = None
    if needs_requirements and session.student is not None:
        requirement_state = toolbox.get_requirement_state()
        _trace(session, {
            "type": "tool_result",
            "step": 0,
            "name": "get_requirement_state",
            "arguments": {},
            "result": requirement_state,
        })

    section = _infer_section_from_text(user_message)
    eligible_results = toolbox.retrieve_catalog_courses(
        queries=queries,
        k=15,
        degree_filter=session.student is not None,
        section=section,
        eligible_only=session.has_course_context(),
    )
    scoped_results: list[dict] = []
    if session.has_course_context():
        scoped_results = toolbox.retrieve_catalog_courses(
            queries=queries,
            k=15,
            degree_filter=session.student is not None,
            section=section,
            eligible_only=False,
        )

    lines = [
        "Prefetched RAG context for this recommendation turn:",
        f"- rewritten_retrieval_queries: {', '.join(queries)}",
        f"- degree_filter: {session.student is not None}",
        f"- eligible_only: {session.has_course_context()}",
        f"- section_filter: {section or 'auto/all relevant requirement sections'}",
    ]
    scheduled_codes = _current_schedule_codes(session)
    if scheduled_codes:
        lines.append(f"- current_schedule_codes: {_compact_codes(scheduled_codes)}")
        if _exclude_current_schedule_requested(user_message):
            lines.append("- recommendation_instruction: user asked for other/alternative courses; do not recommend courses already in current_schedule_codes.")

    if requirement_state and not requirement_state.get("error"):
        lines.extend([
            f"- requirement_scope: {requirement_state.get('program')} / {requirement_state.get('cohort_term')}",
            f"- required_left: {_compact_codes(requirement_state.get('required_left', []), limit=12)}",
            f"- core_elective_left_count: {len(requirement_state.get('core_elective_left', []) or [])}",
            f"- area_elective_left_count: {len(requirement_state.get('area_elective_left', []) or [])}",
            f"- free_elective_left_count: {len(requirement_state.get('free_elective_left', []) or [])}",
            f"- authoritative_section_credit_status: {json.dumps(requirement_state.get('section_credit_status', {}), ensure_ascii=False)}",
        ])

    def append_rows(label: str, rows: list[dict]) -> None:
        lines.append(f"- {label}:")
        if not rows or (len(rows) == 1 and rows[0].get("error")):
            lines.append(f"  - retrieval_error: {rows[0].get('error') if rows else 'no results'}")
            return
        focused_rows = _filter_rows_by_focus(rows, queries)
        for row in focused_rows[:8]:
            code = row.get("code", "")
            title = row.get("title", "")
            credits = row.get("credits")
            eligible = row.get("in_eligible_pool")
            completed = row.get("already_completed")
            taking = row.get("currently_taking")
            scheduled = row.get("in_current_schedule")
            likely = row.get("likely_offered_target_term")
            desc = str(row.get("description") or "").strip().replace("\n", " ")
            if len(desc) > 260:
                desc = desc[:257].rstrip() + "..."
            lines.append(
                f"  - {code} - {title}"
                f"{f' ({credits:g} SU)' if isinstance(credits, (int, float)) else ''}"
                f"; eligible_for_target_term={eligible}"
                f"; already_completed={completed}"
                f"; currently_taking={taking}"
                f"; in_current_schedule={scheduled}"
                f"; likely_offered_target_term={likely}; {desc}"
            )

    append_rows("eligible_retrieved_courses", eligible_results)
    if scoped_results:
        append_rows("relevant_degree_courses_even_if_blocked_completed_or_in_progress", scoped_results)

    lines.append("Use this context to answer the original user message. Do not expose rewritten query mechanics.")
    return "\n".join(lines)


# Normalized degree-section name -> the matching key in a parsed degree
# evaluation's ``section_requirements`` (which carries the authoritative
# min / completed SU-credit figures per section).
_SECTION_TO_EVAL_KEY = {
    "Required": "REQUIRED COURSES",
    "Core Elective": "CORE ELECTIVES",
    "Area Elective": "AREA ELECTIVES",
    "Free Elective": "FREE ELECTIVES",
}


def _section_credit_status(section_requirements: dict, normalized: str) -> dict | None:
    """Authoritative SU-credit status for a degree section from a degree eval.

    Returns ``None`` when no degree-evaluation figures are available for the
    section (e.g. a plain JSON transcript was uploaded instead).
    """
    sr = section_requirements or {}
    entry = sr.get(_SECTION_TO_EVAL_KEY.get(normalized, ""))
    if not entry:
        return None
    min_su = entry.get("min_su")
    done_su = entry.get("completed_su")
    remaining = None
    satisfied = None
    if min_su is not None and done_su is not None:
        remaining = round(max(0.0, float(min_su) - float(done_su)), 1)
        satisfied = float(done_su) >= float(min_su)
    return {
        "min_su_credits": min_su,
        "completed_su_credits": done_su,
        "su_credits_remaining": remaining,
        "satisfied": satisfied,
        "source": "degree_evaluation",
    }


_MINOR_CATALOG: Any | None = None


def _minor_catalog() -> Any:
    """Lazy module-level singleton for the minor requirements catalog."""
    global _MINOR_CATALOG
    if _MINOR_CATALOG is None:
        from scheduler.minors import MinorCatalog
        _MINOR_CATALOG = MinorCatalog.load()
    return _MINOR_CATALOG


class Toolbox:
    def __init__(self, session: "PlannerSession") -> None:
        self.session = session
        session._ensure_heavy_state()

    def get_student_profile(self) -> dict:
        s = self.session.student
        req = self.session.base_request
        if s is None:
            return {
                "has_profile": False,
                "transcript_loaded": False,
                "missing": ["degree/program", "admit term", "completed courses"],
                "target_term": req.target_term,
                "credit_range": {"min": req.min_credits, "max": req.max_credits, "target": req.target_credits},
            }
        raw = self.session._raw
        profile = {
            "has_profile": True,
            "transcript_loaded": self.session.transcript_loaded,
            "source": raw.get("source", "transcript"),
            "name": raw.get("name", "Student"),
            "program": s.program,
            "admit_term": s.admit_term,
            "current_semester": raw.get("current_semester", 0),
            "cgpa": raw.get("cgpa", 0.0),
            "cumulative_credits": s.cumulative_credits,
            "minors": raw.get("minors", []),
            "intended_minor": raw.get("intended_minor"),
            "completed_count": len(s.completed),
            "completed": sorted(s.completed),
            "in_progress": sorted(s.in_progress),
            "target_term": req.target_term,
            "credit_range": {"min": req.min_credits, "max": req.max_credits, "target": req.target_credits},
        }
        # Compact Engineering / Basic-Science status (from a degree evaluation),
        # so the planner is aware of these graduation constraints up front.
        sr = self.session.section_requirements or {}
        eng, sci = sr.get("ENGINEERING"), sr.get("BASIC SCIENCE")
        if eng or sci:
            def _gap(section: dict | None) -> dict | None:
                if not section:
                    return None
                min_e = section.get("min_ects")
                done_e = section.get("completed_ects")
                rem = None
                if min_e is not None and done_e is not None:
                    rem = round(max(0.0, float(min_e) - float(done_e)), 1)
                return {"min_ects": min_e, "completed_ects": done_e, "remaining_ects": rem}
            profile["science_engineering"] = {
                "engineering": _gap(eng),
                "basic_science": _gap(sci),
            }
        return profile

    def set_student_context(
        self,
        program: str | None = None,
        admit_term: str | None = None,
        completed: list[str] | None = None,
        in_progress: list[str] | None = None,
    ) -> dict:
        ignored = False
        if self.session.transcript_loaded:
            ignored = any(value is not None for value in (program, admit_term, completed, in_progress))
        if admit_term and re.fullmatch(r"\d{4}", admit_term.strip()):
            admit_term = admit_term.strip() + "01"
        self.session.update_manual_context(
            program=program,
            admit_term=admit_term,
            completed={c.upper().strip() for c in completed} if completed is not None else None,
            in_progress={c.upper().strip() for c in in_progress} if in_progress is not None else None,
        )
        profile = self.get_student_profile()
        if ignored:
            profile["context_update_ignored"] = "Transcript is loaded, so parsed degree/admit/completed context was kept authoritative."
        return profile

    def select_degree_graphs(self, sections: list[str] | None = None) -> dict:
        if sections and self.session._student is not None:
            from scheduler.graph_selector import select_graphs
            selected = select_graphs(
                self.session._student.program,
                self.session._student.admit_term,
                sections=tuple(_normalize_section_name(s) for s in sections),
            )
            self.session._selected_graphs = selected
            self.session._graph = selected.merged
            self.session._remaining = None
            self.session._eligible_pool = set()
            self.session._required_injected = set()
        self.session._ensure_heavy_state()
        sg = self.session._selected_graphs
        if sg is None:
            return {"error": "degree/admit term not known", "needed": ["degree/program", "admit term"]}
        result = sg.to_dict()
        _trace(self.session, {"type": "selected_graphs", **result})
        return result

    def get_requirement_state(self, section: str | None = None) -> dict:
        self.session._ensure_heavy_state()
        r = self.session._remaining
        if r is None:
            return {"error": "requirements not loaded", "needed": ["degree/program", "admit term"]}
        data = {
            "program": r.program,
            "cohort_term": r.cohort_term,
            "required_left": r.required_left,
            "core_elective_left": r.core_left,
            "area_elective_left": r.area_left,
            "free_elective_left": r.free_left,
            "required_in_progress": r.required_in_progress,
            "required_credits_left": r.required_credits_left,
        }
        sr = self.session.section_requirements or {}
        if section:
            normalized = _normalize_section_name(section)
            _key_map = {
                "Required": "required_left",
                "Core Elective": "core_elective_left",
                "Area Elective": "area_elective_left",
                "Free Elective": "free_elective_left",
            }
            left = data.get(_key_map.get(normalized, ""), [])
            out = {"program": r.program, "cohort_term": r.cohort_term, "section": normalized}
            status = _section_credit_status(sr, normalized)
            if status is not None:
                out["requirement"] = status
                if status.get("satisfied"):
                    out["note"] = (
                        f"This section is already SATISFIED "
                        f"({status['completed_su_credits']}/{status['min_su_credits']} "
                        f"SU credits). 'options' are not additional requirements."
                    )
                else:
                    out["note"] = (
                        "'options' is the pool of courses that can count toward "
                        "this section — pick enough to cover su_credits_remaining, "
                        "not all of them."
                    )
                out["options"] = left
            else:
                # No degree-evaluation figures — we only know the remaining pool,
                # not how much of the section is already satisfied.
                out["completion_unknown"] = True
                out["note"] = (
                    "No degree-evaluation credit figures available; 'left' is the "
                    "remaining course pool only — the required count for this "
                    "section is unknown, so do not state one."
                )
                out["left"] = left
            return out
        _trace(self.session, {
            "type": "requirements",
            "program": r.program,
            "cohort_term": r.cohort_term,
            "required_left_count": len(r.required_left),
            "core_left_count": len(r.core_left),
            "area_left_count": len(r.area_left),
            "free_left_count": len(r.free_left),
            "required_left": r.required_left,
            "core_left": r.core_left,
            "area_left": r.area_left,
            "free_left": r.free_left,
            "required_credits_left": r.required_credits_left,
        })
        # Authoritative per-section credit status from the degree evaluation,
        # so the agent answers "how many do I still need" from real figures
        # rather than treating the option pools as outstanding requirements.
        if sr:
            credit_status = {}
            for normalized in _SECTION_TO_EVAL_KEY:
                st = _section_credit_status(sr, normalized)
                if st is not None:
                    credit_status[normalized] = st
            if credit_status:
                data["section_credit_status"] = credit_status
                # Align the legacy graph-derived figure with the authoritative
                # degree-evaluation remaining credits so there is a single,
                # consistent number for the Required section.
                req_status = credit_status.get("Required")
                if req_status and req_status.get("su_credits_remaining") is not None:
                    data["required_credits_left"] = req_status["su_credits_remaining"]
                data["note"] = (
                    "section_credit_status is authoritative (from the degree "
                    "evaluation): use it for how many credits each section still "
                    "needs, including required_credits_left. The *_left lists are "
                    "only the pools of eligible courses, NOT outstanding "
                    "requirements — a section can be fully satisfied even while "
                    "its *_left pool is non-empty."
                )
        return data

    def get_remaining_requirements(self) -> dict:
        return self.get_requirement_state()

    def get_degree_section_courses(self, section: str | None = None, query: str | None = None, k: int = 40) -> dict:
        self.session._ensure_heavy_state()
        sg = self.session._selected_graphs
        if sg is None:
            return {"error": "degree/admit term not known", "needed": ["degree/program", "admit term"]}

        if section is None and query:
            section = _infer_section_from_text(query)
        sections = [_normalize_section_name(section)] if section else ["Required", "Core Elective", "Area Elective"]
        pool: set[str] = set()
        section_counts: dict[str, int] = {}
        for sec in sections:
            codes = sg.candidate_courses(sec)
            section_counts[sec] = len(codes)
            pool.update(codes)

        k = max(1, min(int(k or 40), 80))
        search_query = _clean_section_query(query) if query else None
        if search_query:
            rows = self._retrieve_rows([search_query], k=k, pool=pool, subj=None, source_name="faiss_section")
        else:
            rows = []
            for code in sorted(pool)[:k]:
                course = self.session.catalog.get(code)
                rows.append({
                    "rank": len(rows) + 1,
                    "score": 1.0,
                    "code": code,
                    "title": course.title if course else "",
                    "credits": course.su_credit if course else None,
                    "description": (course.description if course else "")[:500],
                    "prereq_text": course.prereq_text if course else "",
                })

        result = {
            "program": sg.program,
            "cohort_term": sg.cohort_term,
            "sections": sections,
            "section_counts": section_counts,
            "query": query,
            "retrieval_query": search_query,
            "count": len(pool),
            "courses": rows,
        }
        _trace(self.session, {"type": "section_courses", **result})
        return result

    def get_current_plan(self) -> dict:
        p = self.session.current_plan
        if p is None:
            return {"plan": [], "total_credits": 0.0, "validation_ok": None, "reasoning": {}, "summary": ""}
        return {
            "plan": p.plan,
            "total_credits": p.total_credits,
            "validation_ok": p.validation_ok,
            "violations": p.violations,
            "reasoning": p.reasoning,
            "summary": p.summary,
            "auto_added_coreqs": p.auto_added_coreqs,
        }

    def rewrite_retrieval_queries(self, original_message: str) -> dict:
        fallback = _fallback_queries(original_message)
        if not original_message.strip():
            return {"queries": fallback}
        try:
            result = llm_client.call_structured(
                model=self.session.base_request.intent_model,
                messages=[
                    {"role": "system", "content": "Rewrite a Sabancı course-advising message into 1-5 concise semantic retrieval queries. Do not answer."},
                    *self.session.recent_history_turns(5),
                    {"role": "user", "content": original_message},
                ],
                schema=QueryRewrite,
                label="query-rewrite",
            )
            queries = [q.strip() for q in result.queries if q.strip()][:5]
        except Exception:
            queries = fallback
        merged: list[str] = []
        for q in [*(queries or []), *fallback]:
            if q and q not in merged:
                merged.append(q)
        return {"queries": merged[:5] or fallback}

    def _candidate_pool(
        self,
        degree_filter: bool = True,
        section: str | None = None,
        eligible_only: bool = False,
        queries: list[str] | None = None,
    ) -> set[str] | None:
        self.session._ensure_heavy_state()
        sg = self.session._selected_graphs
        joined = " ".join(queries or [])
        if section is None:
            section = _infer_section_from_text(joined)
        if eligible_only:
            pool = set(self.session._eligible_pool)
            if degree_filter and sg is not None:
                if (
                    self.session._student
                    and self.session._student.program == "BSIE-DM"
                    and not section
                    and _is_ie_operations_query(joined)
                ):
                    scoped = sg.candidate_courses("Required") | sg.candidate_courses("Core Elective")
                elif section:
                    scoped = sg.candidate_courses(section)
                elif re.search(r"\bfree\s+elective|free\s+electives\b", joined, re.I):
                    scoped = sg.candidate_courses("Free Elective")
                else:
                    scoped = (
                        sg.candidate_courses("Required")
                        | sg.candidate_courses("Core Elective")
                        | sg.candidate_courses("Area Elective")
                    )
                pool &= scoped
            return pool
        # degree_filter=True (default): scope to the student's degree graph.
        # degree_filter=False: search the full catalog (open exploration).
        if degree_filter and sg is not None:
            if (
                self.session._student
                and self.session._student.program == "BSIE-DM"
                and not section
                and _is_ie_operations_query(joined)
            ):
                return sg.candidate_courses("Required") | sg.candidate_courses("Core Elective")
            if not section:
                return sg.candidate_courses("Required") | sg.candidate_courses("Core Elective") | sg.candidate_courses("Area Elective")
            return sg.candidate_courses(section)
        return None

    def _retrieve_rows(
        self,
        queries: list[str],
        k: int,
        pool: set[str] | None,
        subj: list[str] | None,
        source_name: str = "faiss",
    ) -> list[dict]:
        seen: dict[str, dict] = {}
        retriever = self.session.retriever
        for query in [q for q in queries if q.strip()]:
            if retriever is not None:
                try:
                    from scheduler.retriever import RetrieverFilter
                    hits = retriever.retrieve(
                        query,
                        k=k,
                        filt=RetrieverFilter(candidate_pool=pool, subj=subj or None, active_only=True),
                        rerank=True,
                    )
                    rows = [
                        {
                            "rank": h.rank,
                            "score": h.reranker_score if h.reranker_score is not None else h.score,
                            "code": h.code,
                            "title": h.title,
                            "credits": h.su_credit,
                            "description": (h.description or "")[:500],
                            "prereq_text": h.prereq_text,
                            "rerank_score": round(h.reranker_score, 3) if h.reranker_score is not None else None,
                            "source": source_name,
                        }
                        for h in hits
                    ]
                    rows = _apply_topic_boosts(rows, query)
                except Exception as exc:
                    rows = self.session.catalog.search(query, k=k, candidate_pool=pool, subj=subj)
                    rows = _apply_topic_boosts(rows, query)
                    _trace(self.session, {"type": "retrieval_warning", "message": "FAISS retrieval failed; used lexical fallback.", "error": f"{type(exc).__name__}: {exc}"})
            else:
                rows = self.session.catalog.search(query, k=k, candidate_pool=pool, subj=subj)
                rows = _apply_topic_boosts(rows, query)
            for row in rows:
                code = row.get("code")
                if code and (code not in seen or row.get("score", 0) > seen[code].get("score", 0)):
                    seen[code] = row
        results = sorted(seen.values(), key=lambda r: (-float(r.get("score", 0)), r.get("code", "")))[:k]
        for i, row in enumerate(results, start=1):
            row["rank"] = i
        return results

    def retrieve_catalog_courses(
        self,
        queries: list[str],
        k: int = 8,
        subj: list[str] | None = None,
        degree_filter: bool = True,
        section: str | None = None,
        eligible_only: bool = False,
    ) -> list[dict]:
        k = max(1, min(int(k), 15))
        joined_queries = " ".join(queries or [])
        if section is None:
            section = _infer_section_from_text(joined_queries)
        pool = self._candidate_pool(degree_filter, section, eligible_only, queries)
        if eligible_only and not self.session.has_course_context():
            return [{"error": "eligible_only requires transcript or completed-course context", "needed": ["transcript or completed/in-progress courses"]}]
        if subj is None:
            subj = _subject_filter_from_text(joined_queries)
        if (
            subj is None
            and self.session._student is not None
            and self.session._student.program == "BSIE-DM"
            and _is_ie_operations_query(joined_queries)
        ):
            subj = ["IE", "OPIM"] if section in ("Area Elective", "Free Elective") else ["IE"]
        elif (
            subj is None
            and self.session._student is not None
            and self.session._student.program == "BSCS-DM"
            and _focus_terms_for_queries(queries)
        ):
            subj = ["CS", "DSA", "EE", "MATH"]

        exclude_scheduled = _exclude_current_schedule_requested(joined_queries)
        scheduled_codes = _current_schedule_codes(self.session)
        fetch_k = min(30, k * 2) if exclude_scheduled and scheduled_codes else k
        results = self._retrieve_rows(queries, k=fetch_k, pool=pool, subj=subj)
        if exclude_scheduled and scheduled_codes:
            results = [row for row in results if row.get("code") not in scheduled_codes][:k]
        likely_set = (
            likely_offered_in(
                self.session._offerings,
                self.session.base_request.target_term,
                same_season_only=True,
            )
            if self.session._offerings is not None
            else set()
        )
        for row in results:
            code = row.get("code")
            row["is_required_left"] = code in self.session._required_injected
            row["in_eligible_pool"] = code in self.session._eligible_pool if self.session.has_course_context() else None
            if self.session._student is not None:
                row["already_completed"] = code in self.session._student.completed
                row["currently_taking"] = code in self.session._student.in_progress
            row["in_current_schedule"] = code in scheduled_codes
            row["likely_offered_target_term"] = code in likely_set if self.session._offerings is not None else None
        self.session.last_retrieved = results
        _trace(self.session, {
            "type": "retrieval",
            "queries": queries,
            "filters": {
                "degree_filter": degree_filter,
                "eligible_only": eligible_only,
                "section": section,
                "subj": subj,
                "candidate_pool_size": len(pool) if pool is not None else None,
            },
            "merged_results": [
                {
                    "code": r.get("code"),
                    "title": r.get("title"),
                    "score": r.get("score"),
                    "rerank_score": r.get("rerank_score"),
                    "source": r.get("source", "lexical"),
                    "eligible": r.get("in_eligible_pool"),
                    "completed": r.get("already_completed"),
                    "currently_taking": r.get("currently_taking"),
                    "in_current_schedule": r.get("in_current_schedule"),
                    "likely_offered": r.get("likely_offered_target_term"),
                }
                for r in results
            ],
        })
        return results

    def retrieve_courses(self, query: str, k: int = 8, subj: list[str] | None = None) -> list[dict]:
        return self.retrieve_catalog_courses([query], k=k, subj=subj, eligible_only=True)

    def get_course_details(self, codes: list[str]) -> list[dict]:
        out = []
        for code in codes:
            course = self.session.catalog.get(code)
            if not course:
                out.append({"code": code.upper().strip(), "error": "not found"})
            else:
                out.append(course.to_result(rank=len(out) + 1, score=1.0))
        return out

    def get_eligible_courses(self, section: str | None = None, k: int = 30) -> dict:
        self.session._ensure_heavy_state()
        if not self.session.has_course_context():
            return {"error": "student profile not known", "needed": ["transcript or completed courses"]}
        pool = set(self.session._eligible_pool)
        if section and self.session._selected_graphs is not None:
            pool &= self.session._selected_graphs.candidate_courses(section)
        rows = []
        for code in sorted(pool)[: max(1, min(int(k or 30), 50))]:
            course = self.session.catalog.get(code)
            rows.append({
                "code": code,
                "title": course.title if course else self.session._graph.nodes.get(code, {}).get("title", ""),
                "credits": course.su_credit if course else self.session._graph.nodes.get(code, {}).get("su_credit"),
                "section_tags": self.session._graph.nodes.get(code, {}).get("degree_sections", []),
            })
        return {"count": len(pool), "courses": rows}

    def check_prereqs(self, code: str) -> dict:
        code = code.upper().strip()
        graph = self.session._graph
        student = self.session._student
        if graph is None or student is None or not self.session.has_course_context():
            course = self.session.catalog.get(code)
            return {
                "code": code,
                "title": course.title if course else "",
                "prereq_text": course.prereq_text if course else "",
                "is_eligible_next_term": None,
                "needed": ["transcript or completed courses for exact eligibility"],
            }
        if not graph.has_node(code):
            return {"code": code, "error": "course not in catalog"}
        node = graph.nodes[code]
        eligible = prereqs_satisfied(graph, code, student.completed, concurrent_ok=student.in_progress)
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

    def get_offering_pattern(self, code: str) -> dict:
        code = code.upper().strip()
        offerings = self.session._offerings or {}
        history = offered_history(offerings, code)
        target_term = self.session.base_request.target_term
        likely_set = likely_offered_in(offerings, target_term, same_season_only=True)
        return {"code": code, "target_term": target_term, "likely_offered_target_term": code in likely_set, "history": history}

    def get_offerings(self, code: str) -> dict:
        return self.get_offering_pattern(code)

    def validate_plan(self, plan: list[str]) -> dict:
        plan = [c.upper().strip() for c in plan]
        if self.session._graph is None or self.session._student is None or not self.session.has_course_context():
            return {
                "ok": False,
                "total_credits": 0.0,
                "violations": [{"code": None, "kind": "missing_student_context", "message": "Exact plan validation requires transcript or completed-course context."}],
                "auto_added_coreqs": [],
            }
        opts = EligibilityOptions()
        opts.min_term_credits = self.session.base_request.min_credits
        opts.max_term_credits = self.session.base_request.max_credits
        report = elig_validate_plan(self.session._graph, self.session._student, plan, opts)
        return {"ok": report.ok, "total_credits": report.total_credits, "violations": [v.to_dict() for v in report.violations], "auto_added_coreqs": report.auto_added_coreqs}

    def validate_courses(self, codes: list[str]) -> list[dict]:
        return [self.check_prereqs(code) for code in codes]

    def set_plan(self, plan: list[str], reasoning: dict, summary: str) -> dict:
        plan = [c.upper().strip() for c in plan]
        if not self.session.planning_allowed:
            return {"committed": False, "reason": "The user has not explicitly asked to create or commit a plan in this turn."}
        if self.session._graph is None or self.session._student is None or not self.session.has_course_context():
            return {"committed": False, "reason": "set_plan requires transcript or completed-course context"}
        opts = EligibilityOptions()
        opts.min_term_credits = self.session.base_request.min_credits
        opts.max_term_credits = self.session.base_request.max_credits
        report = elig_validate_plan(self.session._graph, self.session._student, plan, opts)
        if not report.ok:
            return {"committed": False, "reason": "validate_plan failed; not committing", "violations": [v.to_dict() for v in report.violations]}
        # Attach a best-effort weekly timetable (uses a same-season proxy term
        # when the target term has no published offerings yet).
        committed_courses = plan + [c for c in report.auto_added_coreqs if c not in plan]
        timetable_payload = None
        try:
            tt = self.build_timetable(committed_courses)
            if "error" not in tt:
                timetable_payload = {tt.get("schedule_term", "schedule"): tt}
        except Exception:
            timetable_payload = None
        self.session.current_plan = TermPlan(
            plan=plan,
            reasoning={str(k).upper().strip(): str(v) for k, v in (reasoning or {}).items()},
            summary=summary,
            total_credits=report.total_credits,
            validation_ok=True,
            violations=[],
            auto_added_coreqs=report.auto_added_coreqs,
            timetable=timetable_payload,
            iterations=0,
            alternatives=[],
            candidate_pool_size=len(self.session._eligible_pool),
            warnings=[],
        )
        return {"committed": True, "plan": plan, "total_credits": report.total_credits, "auto_added_coreqs": report.auto_added_coreqs}

    def list_minors(self) -> dict:
        return {"minors": _minor_catalog().list_minors()}

    def get_minor_requirements(self, minor: str) -> dict:
        mc = _minor_catalog()
        code, candidates = mc.resolve(minor)
        if code is None:
            if candidates:
                return {"error": "ambiguous minor reference", "candidates": candidates}
            return {
                "error": f"unknown minor: {minor!r}",
                "available": [m["code"] for m in mc.list_minors()],
            }
        self.session._ensure_heavy_state()
        catalog = self.session.catalog
        student = self.session.student
        if student is not None and self.session.has_course_context():
            result = mc.progress(code, student.completed, student.in_progress, catalog=catalog)
        else:
            data = mc.get(code) or {}
            reqs = data.get("requirements", {})
            total = reqs.get("total", {})
            result = {
                "code": data.get("code", code),
                "name": data.get("name", code),
                "term": data.get("term"),
                "credit_unit": "SU credit",
                "note": "No transcript loaded — showing requirements only, no progress. All credit figures are SU credits.",
                "required_total": {
                    "min_courses": total.get("min_courses"),
                    "min_su_credits": total.get("min_su"),
                },
                "sections": {
                    sec: {
                        "min_courses": (reqs.get(sec) or {}).get("min_courses"),
                        "min_su_credits": (reqs.get(sec) or {}).get("min_su"),
                        "options": [
                            {"code": c, "title": (catalog.get(c).title if catalog.get(c) else "")}
                            for c in codes
                        ],
                    }
                    for sec, codes in data.get("sections", {}).items()
                },
            }
            total_ects = total.get("min_ects")
            if total_ects is not None:
                result["total_ects_whole_minor"] = total_ects
                result["ects_note"] = (
                    "Whole-minor ECTS total only; no per-course ECTS data, so "
                    "completed/remaining ECTS is unknown. Do not estimate it."
                )
        _trace(self.session, {"type": "minor_requirements", "code": code, "name": result.get("name")})
        return result

    def get_science_engineering_progress(self, courses: list[str] | None = None) -> dict:
        """Engineering / Basic-Science ECTS progress toward graduation.

        Uses the degree-evaluation minimums when available, else estimates from
        the catalog. If ``courses`` are given, also reports how much Engineering
        / Basic-Science ECTS each would add (e.g. to fill a remaining gap).
        """
        from scheduler.eng_sci import EngSciCredits, progress

        self.session._ensure_heavy_state()
        student = self.session.student
        completed = set(student.completed) if student else set()
        in_progress = set(student.in_progress) if student else set()
        creds = EngSciCredits.load()
        result = progress(
            self.session.section_requirements,
            completed,
            in_progress,
            credits=creds,
        )
        if not self.session.section_requirements:
            result["warning"] = (
                "No degree evaluation uploaded — minimums unknown and totals are "
                "estimated from the catalog. Upload a Degree Evaluation HTML for "
                "authoritative Engineering/Basic-Science requirements."
            )
        if courses:
            result["candidate_contributions"] = creds.contributions(courses)
        _trace(self.session, {
            "type": "eng_sci_progress",
            "eng_remaining": result["engineering"].get("remaining_ects"),
            "sci_remaining": result["basic_science"].get("remaining_ects"),
        })
        return result

    def find_courses_by_credit_type(
        self, credit_type: str, subject: str | None = None, limit: int = 25
    ) -> dict:
        """List courses that genuinely carry Basic-Science or Engineering ECTS."""
        from scheduler.eng_sci import EngSciCredits

        self.session._ensure_heavy_state()
        catalog = self.session.catalog
        creds = EngSciCredits.load()
        matches = creds.find(credit_type=credit_type, subject=subject)
        key = "basic_sci_ects" if credit_type.lower().startswith("basic") else "eng_ects"
        label = "Basic-Science" if key == "basic_sci_ects" else "Engineering"
        courses = []
        for m in matches[: max(1, int(limit or 25))]:
            course = catalog.get(m["code"]) if catalog else None
            courses.append({
                "code": m["code"],
                "title": course.title if course else "",
                f"{label.lower().replace('-', '_')}_ects": m[key],
                "eng_ects": m["eng_ects"],
                "basic_sci_ects": m["basic_sci_ects"],
                "ects_total": m["ects_total"],
            })
        _trace(self.session, {
            "type": "find_by_credit_type",
            "credit_type": credit_type,
            "subject": subject,
            "match_count": len(matches),
        })
        return {
            "credit_type": label,
            "subject": (subject or "any").upper(),
            "total_matches": len(matches),
            "returned": len(courses),
            "courses": courses,
            "note": (
                f"These courses carry {label} ECTS per the catalog credit table. "
                "The reported ECTS is the credit-type contribution, NOT the "
                "course's SU credits. Courses not listed carry 0 "
                f"{label} ECTS."
            ),
        }

    def build_timetable(self, courses: list[str]) -> dict:
        """Build a conflict-free weekly timetable from real section meeting times.

        The target planning term often has no published offerings yet (it is a
        future term). When that happens we fall back to the most recent term of
        the *same season* that we do have data for (e.g. plan for Fall 2026-2027
        using Fall 2025-2026 sections) and clearly flag that the times are a
        proxy, since exact meeting times for the future term are not yet known.
        """
        from scheduler.timetable import (
            build_timetable as _bt,
            format_timetable as _fmt,
            load_offerings as _load_off,
            resolve_schedule_term,
            annotate_meeting,
        )

        codes = [c.upper().strip() for c in (courses or []) if c and c.strip()]
        if not codes:
            return {"error": "no courses given to schedule"}

        from scheduler.catalog import corequisite_codes

        expanded_codes: list[str] = []
        auto_added_coreqs: list[str] = []
        for code in codes:
            if code not in expanded_codes:
                expanded_codes.append(code)
            for coreq in corequisite_codes(self.session.catalog, code):
                if coreq not in expanded_codes:
                    expanded_codes.append(coreq)
                    auto_added_coreqs.append(coreq)
        codes = expanded_codes

        offerings = self.session._offerings or _load_off()
        target_term = self.session.base_request.target_term

        # Pick the term whose meeting times we actually schedule against.
        schedule_term, is_proxy = resolve_schedule_term(offerings, target_term)
        if not schedule_term:
            return {
                "error": "no offerings available to build a timetable",
                "target_term": target_term,
            }

        result = _bt(codes, schedule_term, offerings=offerings)
        payload = result.to_dict()
        payload["target_term"] = target_term
        payload["schedule_term"] = schedule_term
        payload["proxy_term_used"] = is_proxy
        payload["auto_added_coreqs"] = auto_added_coreqs

        # When scheduling against a proxy (past) term, the CRNs and section
        # numbers are NOT valid for the future target term — drop them so neither
        # the model nor the UI presents them as registerable identifiers.
        if is_proxy:
            for p in payload["picks"]:
                p["reference_crn_past_term"] = p.pop("crn", None)
                p["crn"] = None
                p["crn_note"] = "CRN/section apply to the proxy term only; will differ for the target term."

        # Add ready-to-render, human-readable meeting strings to every pick so
        # the frontend can draw a weekly grid without re-deriving anything.
        for p in payload["picks"]:
            for m in p.get("meetings", []):
                annotate_meeting(m)
            p["when"] = "; ".join(
                f"{'/'.join(m.get('day_labels', []))} {m['start_label']}-{m['end_label']}"
                for m in p.get("meetings", [])
            )

        # Build the LLM-facing text from the (CRN-scrubbed) payload, not the raw
        # engine output, so the model never quotes a stale CRN.
        payload["text"] = _fmt(result)
        if is_proxy:
            payload["text"] = re.sub(r",\s*CRN\s*\d+", "", payload["text"])

        if is_proxy:
            from scheduler.prompts import term_label as _term_label
            try:
                tgt_lbl = _term_label(target_term)
                proxy_lbl = _term_label(schedule_term)
            except Exception:
                tgt_lbl, proxy_lbl = target_term, schedule_term
            payload["note"] = (
                f"Exact meeting times for {tgt_lbl} (term {target_term}) are not "
                f"published yet, so this timetable uses the most recent same-season "
                f"term, {proxy_lbl} (term {schedule_term}), as a proxy. Days/times "
                "are typical for the course but may shift, and CRNs/section numbers "
                "are NOT shown because they will differ for the target term."
            )
        _trace(self.session, {
            "type": "build_timetable",
            "courses": codes,
            "auto_added_coreqs": auto_added_coreqs,
            "target_term": target_term,
            "schedule_term": schedule_term,
            "proxy_term_used": is_proxy,
            "ok": result.ok,
            "missing": result.missing,
            "no_meetings": result.no_meetings,
        })
        return payload

    def get_current_schedule(self) -> dict:
        """Return the schedule the student is assembling in the UI builder.

        Includes each picked course's meeting times, total SU credits, and a
        server-computed list of time conflicts between the chosen sections.
        Use this whenever the user asks to review/check "my schedule".
        """
        from scheduler.timetable import conflict_report

        picks = self.session.user_schedule or []
        if not picks:
            return {
                "has_schedule": False,
                "note": "The student has not added any courses to the builder yet.",
            }

        total_credits = round(sum(float(p.get("su_credit") or 0) for p in picks), 1)
        # conflict_report needs each section to carry a 'label' + 'meetings'.
        for p in picks:
            p.setdefault("label", f"{p.get('code', '?')} ({p.get('section', '?')})")
        clashes = conflict_report(picks)
        conflicts = [
            {
                "between": [a, b],
                "day": ma.get("days_raw") or "/".join(ma.get("day_labels", [])),
                "times": f"{ma.get('start_label','')}-{ma.get('end_label','')} vs "
                         f"{mb.get('start_label','')}-{mb.get('end_label','')}",
            }
            for (a, b, ma, mb) in clashes
        ]
        return {
            "has_schedule": True,
            "target_term": self.session.base_request.target_term,
            "course_count": len(picks),
            "total_su_credits": total_credits,
            "courses": [
                {
                    "code": p.get("code"),
                    "title": p.get("title", ""),
                    "section": p.get("section"),
                    "su_credit": p.get("su_credit"),
                    "when": p.get("when", ""),
                    "is_corequisite": bool(p.get("is_corequisite")),
                    "corequisite_for": p.get("corequisite_for"),
                }
                for p in picks
            ],
            "has_conflicts": bool(conflicts),
            "conflicts": conflicts,
            "credit_range": {
                "min": self.session.base_request.min_credits,
                "max": self.session.base_request.max_credits,
            },
            "note": (
                "This is the student's own assembled schedule. Meeting times come "
                "from offerings data (a same-season proxy term if the target term "
                "isn't published yet), so days/times are indicative and CRNs may "
                "differ for the actual term."
            ),
        }

    def check_courses_against_current_schedule(self, codes: list[str]) -> dict:
        """Try candidate courses against the fixed UI-builder schedule.

        Unlike build_timetable(), this does not reshuffle the student's current
        picked sections. It searches only the candidate courses' available
        sections, including direct catalog recitations/labs.
        """
        from scheduler.catalog import corequisite_codes
        from scheduler.timetable import (
            annotate_meeting,
            conflict_report,
            load_offerings as _load_off,
            resolve_schedule_term,
            sections_conflict,
            sections_for,
        )

        requested = []
        for code in codes or []:
            normalized = code.upper().strip()
            if normalized and normalized not in requested:
                requested.append(normalized)
        if not requested:
            return {"error": "no candidate courses given", "needed": ["course codes"]}

        current_raw = self.session.user_schedule or []
        if not current_raw:
            return {
                "error": "no current schedule in builder",
                "needed": ["add courses to the schedule builder first"],
            }

        offerings = self.session._offerings or _load_off()
        target_term = self.session.base_request.target_term
        schedule_term, is_proxy = resolve_schedule_term(offerings, target_term)
        if not schedule_term:
            return {
                "error": "no offerings available to check schedule conflicts",
                "target_term": target_term,
            }

        fixed: list[dict] = []
        current_codes: set[str] = set()
        for item in current_raw:
            pick = dict(item)
            code = str(pick.get("code", "")).upper().strip()
            if not code:
                continue
            current_codes.add(code)
            pick["code"] = code
            pick.setdefault("label", f"{code} ({pick.get('section') or '?'})")
            pick["meetings"] = [annotate_meeting(dict(m)) for m in pick.get("meetings", [])]
            fixed.append(pick)

        existing_conflicts = _conflict_dicts(conflict_report(fixed))

        candidate_codes: list[str] = []
        auto_added_coreqs: list[str] = []
        already_in_schedule: list[str] = []
        for code in requested:
            if code in current_codes:
                already_in_schedule.append(code)
                continue
            if code not in candidate_codes:
                candidate_codes.append(code)
            for coreq in corequisite_codes(self.session.catalog, code):
                if coreq in current_codes or coreq in candidate_codes:
                    continue
                candidate_codes.append(coreq)
                auto_added_coreqs.append(coreq)

        if not candidate_codes:
            return {
                "ok": True,
                "candidate_fits_current_schedule": True,
                "overall_schedule_conflict_free": not bool(existing_conflicts),
                "requested_codes": requested,
                "already_in_schedule": already_in_schedule,
                "candidate_codes_checked": [],
                "auto_added_coreqs": [],
                "existing_conflicts": existing_conflicts,
                "note": "All requested courses are already in the current schedule.",
            }

        missing: list[str] = []
        no_meetings: list[str] = []
        candidates: list[tuple[str, list[dict]]] = []
        for code in candidate_codes:
            sections = sections_for(offerings, schedule_term, code)
            if not sections:
                missing.append(code)
                continue
            with_times = [s for s in sections if s.get("meetings")]
            if not with_times:
                no_meetings.append(code)
                continue
            normalized_sections = []
            for section in with_times:
                sec = dict(section)
                sec["code"] = code
                sec["meetings"] = [annotate_meeting(dict(m)) for m in section.get("meetings", [])]
                sec.setdefault("label", f"{code} ({sec.get('section') or '?'})")
                normalized_sections.append(sec)
            candidates.append((code, normalized_sections))

        candidates.sort(key=lambda item: len(item[1]))
        chosen: list[dict] = []

        def backtrack(index: int) -> bool:
            if index == len(candidates):
                return True
            _, sections = candidates[index]
            for section in sections:
                if any(sections_conflict(section, pick) for pick in fixed):
                    continue
                if any(sections_conflict(section, pick) for pick in chosen):
                    continue
                chosen.append(section)
                if backtrack(index + 1):
                    return True
                chosen.pop()
            return False

        fits = (backtrack(0) if candidates else True) and not bool(missing)
        blocking_conflicts = []
        if not fits:
            seen_conflicts: set[tuple[str, str, str]] = set()
            for code, sections in candidates:
                for section in sections[:8]:
                    rows = _conflict_dicts(conflict_report([*fixed, section]))
                    for row in rows:
                        pair = tuple(row["between"])
                        if section.get("label") not in pair:
                            continue
                        key = (pair[0], pair[1], row["times"])
                        if key in seen_conflicts:
                            continue
                        seen_conflicts.add(key)
                        blocking_conflicts.append({
                            "candidate": code,
                            **row,
                        })
                    if len(blocking_conflicts) >= 12:
                        break
                if len(blocking_conflicts) >= 12:
                    break

        def pick_payload(pick: dict) -> dict:
            meetings = pick.get("meetings", [])
            return {
                "code": pick.get("code"),
                "title": pick.get("title", ""),
                "section": pick.get("section"),
                "crn": None if is_proxy else str(pick.get("crn")) if pick.get("crn") is not None else None,
                "reference_crn_past_term": str(pick.get("crn")) if is_proxy and pick.get("crn") is not None else None,
                "when": "; ".join(
                    f"{'/'.join(m.get('day_labels', []))} {_meeting_time_label(m)}"
                    for m in meetings
                ),
                "meetings": meetings,
            }

        candidate_credits = 0.0
        for code in candidate_codes:
            course = self.session.catalog.get(code)
            if course and course.su_credit:
                candidate_credits += float(course.su_credit)

        total_current = round(sum(float(p.get("su_credit") or 0) for p in current_raw), 1)
        payload = {
            "ok": fits,
            "candidate_fits_current_schedule": fits,
            "overall_schedule_conflict_free": fits and not bool(existing_conflicts),
            "requested_codes": requested,
            "already_in_schedule": already_in_schedule,
            "candidate_codes_checked": candidate_codes,
            "auto_added_coreqs": auto_added_coreqs,
            "target_term": target_term,
            "schedule_term": schedule_term,
            "proxy_term_used": is_proxy,
            "missing": missing,
            "no_meetings": no_meetings,
            "existing_conflicts": existing_conflicts,
            "blocking_conflicts": blocking_conflicts,
            "chosen_candidate_sections": [pick_payload(p) for p in chosen] if fits else [],
            "current_total_su_credits": total_current,
            "candidate_su_credits": round(candidate_credits, 1),
            "total_su_credits_if_added": round(total_current + candidate_credits, 1),
            "note": (
                "Existing builder selections were kept fixed. "
                "Candidate recitations/labs were auto-added from catalog corequisites. "
                + (
                    "Target-term times are not published, so the check uses the latest same-season proxy term."
                    if is_proxy else
                    "The check uses published target-term meeting times."
                )
            ),
        }
        _trace(self.session, {
            "type": "check_courses_against_current_schedule",
            "requested_codes": requested,
            "candidate_codes_checked": candidate_codes,
            "auto_added_coreqs": auto_added_coreqs,
            "ok": fits,
            "existing_conflict_count": len(existing_conflicts),
            "blocking_conflict_count": len(blocking_conflicts),
        })
        return payload

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


MAX_ITERATIONS = 8


def handle_turn(session: "PlannerSession", user_message: str) -> str:
    session.last_trace = []
    session.last_raw_trace = []
    _infer_context_from_message(session, user_message)
    session.planning_allowed = _planning_requested(user_message)
    toolbox = Toolbox(session)
    model = session.base_request.planner_model

    _trace(session, {
        "type": "turn_start",
        "user_message": user_message,
        "planning_allowed": session.planning_allowed,
        "state": session.state_summary(),
    })

    # Keep the system prompt stable for provider-side prompt caching. Runtime
    # values are injected as labeled context messages below.
    from scheduler.prompts import term_label as _term_label

    target_term = session.base_request.target_term
    try:
        target_label = _term_label(target_term)
    except Exception:
        target_label = target_term
    system_content = REACT_SYSTEM

    messages: list[dict] = [{"role": "system", "content": system_content}]
    messages.append({
        "role": "user",
        "content": "[runtime environment; authoritative]",
    })
    messages.append({
        "role": "assistant",
        "content": json.dumps({
            "today": date.today().isoformat(),
            "target_term": target_term,
            "target_label": target_label,
            "instruction": (
                "Answer any date or 'which/next semester' question from these "
                "values, never from prior knowledge."
            ),
        }, ensure_ascii=False),
    })

    # When student context is already loaded (transcript, degree evaluation, or
    # manual state), inject a compact profile snapshot before history without
    # changing the stable system prompt.
    if session.has_course_context():
        profile = toolbox.get_student_profile()
        messages.append({
            "role": "user",
            "content": "[student context loaded; authoritative; do not ask the student to repeat it]",
        })
        messages.append({
            "role": "assistant",
            "content": json.dumps(profile, ensure_ascii=False),
        })

    recommendation_context = _prefetch_recommendation_context(session, toolbox, user_message)
    if recommendation_context:
        _trace(session, {"type": "prefetched_recommendation_context", "content": recommendation_context})
        messages.append({
            "role": "user",
            "content": "[retrieved recommendation context for this turn; authoritative]",
        })
        messages.append({
            "role": "assistant",
            "content": recommendation_context,
        })

    messages.extend(session.recent_history_turns(25))
    messages.append({"role": "user", "content": user_message})

    final_text = ""
    profile_reask_repaired = False
    catalog_mismatch_repaired = False
    for step in range(MAX_ITERATIONS):
        assistant_msg = llm_client.call_with_tools(
            model=model,
            messages=messages,
            tools=TOOLS,
            label=f"react/step{step+1}",
        )
        tool_calls = assistant_msg.tool_calls or []
        _trace(session, {
            "type": "assistant_step",
            "step": step + 1,
            "content": assistant_msg.content or "",
            "tool_calls": [{"name": tc.function.name, "arguments": tc.function.arguments} for tc in tool_calls],
        })
        messages.append({
            "role": "assistant",
            "content": assistant_msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ] if tool_calls else None,
        })
        if not tool_calls:
            final_text = assistant_msg.content or ""
            if (
                session.transcript_loaded
                and not profile_reask_repaired
                and _reasks_for_known_profile(final_text)
            ):
                profile_reask_repaired = True
                _trace(session, {
                    "type": "profile_reask_repair",
                    "message": "Model asked for profile fields already available from the uploaded academic profile; forcing a revised answer.",
                })
                messages.append({
                    "role": "system",
                    "content": (
                        "Revise the previous answer. The student's academic profile is already loaded and authoritative. "
                        "Use the current student context and prefetched RAG context; do not ask for program, admit term, "
                        "completed courses, in-progress courses, transcript, or degree evaluation again."
                    ),
                })
                continue
            mismatches = _catalog_title_mismatches(session, final_text)
            if mismatches and not catalog_mismatch_repaired:
                catalog_mismatch_repaired = True
                _trace(session, {
                    "type": "catalog_title_repair",
                    "mismatches": mismatches,
                })
                messages.append({
                    "role": "system",
                    "content": _catalog_mismatch_message(mismatches),
                })
                continue
            break
        for tc in tool_calls:
            args: dict = {}
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                result: Any = {"error": f"could not parse arguments: {exc}"}
            else:
                result = toolbox.dispatch(tc.function.name, args)
            _trace(session, {"type": "tool_result", "step": step + 1, "name": tc.function.name, "arguments": args, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })
    else:
        final_text = "I ran out of reasoning steps before settling on an answer. Could you narrow the request a bit?"

    if not final_text.strip():
        final_text = "(no response)"
    _trace(session, {"type": "final_response", "content": final_text})
    return final_text
