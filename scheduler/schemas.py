"""
Pydantic models for the SuSchedule-r planner agent.

Three layers:
  - LLM output schemas   : IntentQueries, PlannerOutput, RepairOutput
    Used with OpenAI Structured Outputs (strict=True) — the API guarantees
    these shapes will be returned exactly.

  - Agent I/O dataclasses: PlannerRequest, TermPlan
    Plain Python dataclasses; serialisable to dict for the CLI / future API.

  - Shared helpers        : CourseEntry (the candidate list row the LLM sees)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# LLM output — OpenAI Structured Outputs schemas
# (all fields required; no Optional inside strict mode)
# --------------------------------------------------------------------------- #

class IntentQueries(BaseModel):
    """Stage 4a: split the user's free-text request into retrieval queries."""

    queries: list[str] = Field(
        description=(
            "1–4 short search queries distilled from the user's request. "
            "Each query targets a single concept (e.g. 'machine learning', "
            "'easy humanities elective'). Never repeat the same concept twice."
        )
    )


class CourseReasoning(BaseModel):
    """Per-course rationale in the LLM's plan."""

    code: str = Field(description="Course code, e.g. 'CS 301'")
    reason: str = Field(
        description=(
            "1–2 sentences explaining why this course was chosen: "
            "relevance to the student's goals, graduation impact, prereq chain, etc."
        )
    )


class PlannerOutput(BaseModel):
    """Stage 5: the LLM's proposed term plan."""

    plan: list[str] = Field(
        description="Ordered list of course codes selected for the term."
    )
    reasoning: list[CourseReasoning] = Field(
        description="One entry per course in `plan`, in the same order."
    )
    summary: str = Field(
        description=(
            "2–3 sentence narrative describing the overall semester balance "
            "(workload, subject mix, graduation progress)."
        )
    )
    alternatives_considered: list[str] = Field(
        description=(
            "Course codes that were strong candidates but not selected. "
            "Empty list if none."
        )
    )


class RepairOutput(BaseModel):
    """Stage 7: the LLM's revised plan after a validation failure."""

    plan: list[str] = Field(
        description="Revised list of course codes, fixing all reported violations."
    )
    reasoning: list[CourseReasoning] = Field(
        description="One entry per course in `plan`."
    )
    summary: str = Field(
        description="2–3 sentence overview of the revised semester."
    )
    change_explanation: str = Field(
        description=(
            "Brief explanation of what was changed relative to the previous plan "
            "and why (which courses were swapped, dropped, or added)."
        )
    )


# --------------------------------------------------------------------------- #
# Agent I/O
# --------------------------------------------------------------------------- #

@dataclass
class PlannerRequest:
    """Everything the agent needs to generate a term plan."""

    # -- Required --
    target_term: str              # "202601"
    user_request: str             # free-text student request (may be empty)

    # -- Transcript source (exactly one must be set) --
    transcript_json: Path | None = None    # path to transcript_*.json
    transcript_pdf:  Path | None = None    # path to Academic Records PDF

    # -- Credit constraints --
    target_credits:  float = 17.0
    max_credits:     float = 21.0
    min_credits:     float = 12.0

    # -- Retrieval knobs --
    k_per_query:     int   = 8    # candidates retrieved per intent query
    first_stage_k:   int   = 32  # bi-encoder pool before cross-encoder

    # -- Narrowing filters (None = no restriction) --
    preferred_subjects:  list[str] | None = None   # e.g. ["CS", "IE"]
    excluded_subjects:   list[str] | None = None
    sections_filter:     list[str] | None = None   # e.g. ["Required","Core Elective"]

    # -- Planner behaviour --
    include_required:    bool = True   # auto-inject still-needed required courses
    max_repair_iters:    int  = 3

    # -- Models --
    planner_model:    str = "gpt-4o"
    intent_model:     str = "gpt-4o-mini"

    # -- Agent style --
    # "pipeline" → fixed 8-stage scheduler.agent.plan (Phase A/B path).
    # "react"    → tool-using loop in scheduler.react_agent (no intent dispatch,
    #              the LLM decides what to do via tool calls).
    mode:             Literal["pipeline", "react"] = "pipeline"


@dataclass
class TermPlan:
    """The agent's final output — everything the UI / CLI needs to display."""

    # Core result
    plan:             list[str]           # course codes
    reasoning:        dict[str, str]      # code → explanation
    summary:          str

    # Credits & validation
    total_credits:    float
    validation_ok:    bool
    violations:       list[dict]          # from PlanReport.violations
    auto_added_coreqs: list[str]

    # Timetable (None if not resolvable or not requested)
    timetable:        dict[str, dict] | None

    # Diagnostics
    iterations:       int                 # how many repair loops
    alternatives:     list[str]           # courses the LLM considered but dropped
    candidate_pool_size: int              # size of pre-LLM candidate list
    warnings:         list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "plan":               self.plan,
            "reasoning":          self.reasoning,
            "summary":            self.summary,
            "total_credits":      self.total_credits,
            "validation_ok":      self.validation_ok,
            "violations":         self.violations,
            "auto_added_coreqs":  self.auto_added_coreqs,
            "timetable":          self.timetable,
            "iterations":         self.iterations,
            "alternatives":       self.alternatives,
            "candidate_pool_size":self.candidate_pool_size,
            "warnings":           self.warnings,
        }
