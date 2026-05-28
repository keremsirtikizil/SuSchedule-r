"""SuSchedule-r ReAct course-advising agent."""
from __future__ import annotations

import json
import re
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
5. For IE operations/supply-chain/logistics/production questions, use the BSIE
   scoped graph and prefer Required + Core Elective IE courses first.
6. Answer the immediate question first. Offer 2-4 concrete next actions when useful.
7. Do not create, validate, or commit a semester plan unless the user explicitly
   asks for planning/checking or provides courses to validate.
8. Never call set_plan unless validate_plan returned ok=true.
9. If degree/admit/transcript is missing, say exactly what is missing. You may
   still answer general RAG questions, but do not claim exact takeability.
10. For minor questions ("can I minor in X", "what does the finance minor need",
    "how close am I to a math minor"), call get_minor_requirements (or
    list_minors to show options). Minor elective sections normally require a
    chosen subset, not every listed course — say so rather than implying all
    are mandatory.
11. For science/engineering credit questions ("do I have enough engineering
    credits", "how many basic science ECTS do I still need"), call
    get_science_engineering_progress. When recommending or validating a plan for
    an engineering degree, check this: if Engineering or Basic-Science ECTS are
    still short, prefer courses that fill the gap and pass the candidate codes to
    the tool to compare their contributions.

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
    session.last_trace.append(_compact_for_trace(event))
    if len(session.last_trace) > 120:
        session.last_trace = session.last_trace[-120:]


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
        if section:
            normalized = _normalize_section_name(section)
            _key_map = {
                "Required": "required_left",
                "Core Elective": "core_elective_left",
                "Area Elective": "area_elective_left",
                "Free Elective": "free_elective_left",
            }
            left = data.get(_key_map.get(normalized, ""), [])
            return {"program": r.program, "cohort_term": r.cohort_term, "section": normalized, "left": left}
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
                    *self.session.history[-10:],
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
        if eligible_only:
            return set(self.session._eligible_pool)
        sg = self.session._selected_graphs
        joined = " ".join(queries or [])
        if section is None:
            section = _infer_section_from_text(joined)
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
        if (
            subj is None
            and self.session._student is not None
            and self.session._student.program == "BSIE-DM"
            and _is_ie_operations_query(joined_queries)
        ):
            subj = ["IE", "OPIM"] if section in ("Area Elective", "Free Elective") else ["IE"]

        results = self._retrieve_rows(queries, k=k, pool=pool, subj=subj)
        for row in results:
            row["is_required_left"] = row.get("code") in self.session._required_injected
            row["in_eligible_pool"] = row.get("code") in self.session._eligible_pool if self.session.has_course_context() else None
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
        self.session.current_plan = TermPlan(
            plan=plan,
            reasoning={str(k).upper().strip(): str(v) for k, v in (reasoning or {}).items()},
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
            result = {
                "code": data.get("code", code),
                "name": data.get("name", code),
                "term": data.get("term"),
                "note": "No transcript loaded — showing requirements only, no progress.",
                "sections": {
                    sec: [
                        {"code": c, "title": (catalog.get(c).title if catalog.get(c) else "")}
                        for c in codes
                    ]
                    for sec, codes in data.get("sections", {}).items()
                },
            }
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


MAX_ITERATIONS = 12


def handle_turn(session: "PlannerSession", user_message: str) -> str:
    session.last_trace = []
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

    # When student context is already loaded (transcript or manual), inject a
    # compact profile snapshot so the LLM starts informed without needing a
    # get_student_profile() tool call on every turn.
    system_content = REACT_SYSTEM
    if session.has_course_context():
        profile = toolbox.get_student_profile()
        system_content += (
            "\n\n--- Student context (already loaded) ---\n"
            + json.dumps(profile, ensure_ascii=False)
            + "\n--- Use this; do not ask the student to repeat it. ---"
        )

    messages: list[dict] = [{"role": "system", "content": system_content}]
    messages.extend(session.history[-100:])
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
