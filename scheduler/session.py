"""
PlannerSession — stateful conversational planner.

Holds all heavy state (student profile, graph, retriever, offerings, current
plan, conversation history) so it is loaded once per CLI session and reused
across turns without re-loading from disk.

Usage
-----
    session = PlannerSession.from_request(PlannerRequest(...))
    response = session.handle_turn("Plan my Fall 2026 semester. I want ML.")
    response = session.handle_turn("Swap CS 406 for something about NLP.")
    response = session.handle_turn("Why did you pick CS 306?")
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from scheduler.schemas import PlannerRequest, TermPlan
from scheduler.eligibility import Student, EligibilityOptions, eligible_courses
from scheduler.offerings import load_offerings, likely_offered_in
from scheduler.requirements import compute_remaining, RequirementsReport
from scheduler.retriever import Retriever

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


# --------------------------------------------------------------------------- #
# Session class
# --------------------------------------------------------------------------- #

@dataclass
class PlannerSession:
    """All state needed across conversational turns.

    Instantiate via ``PlannerSession.from_request(...)`` — the factory method
    loads the transcript and initialises cached fields.
    """

    # Config (fixed for the whole session)
    base_request: PlannerRequest

    # Loaded once from transcript
    _student: Student = field(default=None, repr=False)        # type: ignore[assignment]
    _raw: dict = field(default_factory=dict, repr=False)

    # Loaded lazily on first plan() call
    _graph: nx.DiGraph | None = field(default=None, repr=False)
    _offerings: dict | None = field(default=None, repr=False)
    _retriever: Retriever | None = field(default=None, repr=False)
    _remaining: RequirementsReport | None = field(default=None, repr=False)
    _eligible_pool: set[str] = field(default_factory=set, repr=False)
    _required_injected: set[str] = field(default_factory=set, repr=False)
    _candidates_cache: list = field(default_factory=list, repr=False)

    # Conversational state
    current_plan: TermPlan | None = None
    history: list[dict] = field(default_factory=list)   # OpenAI message list

    # ------------------------------------------------------------------ #
    # Factory
    # ------------------------------------------------------------------ #

    @classmethod
    def from_request(cls, request: PlannerRequest) -> "PlannerSession":
        """Load the transcript and return a ready session."""
        session = cls(base_request=request)
        session._load_student()
        return session

    # ------------------------------------------------------------------ #
    # Lazy loaders
    # ------------------------------------------------------------------ #

    def _load_student(self) -> None:
        req = self.base_request
        if req.transcript_json:
            raw = json.loads(Path(req.transcript_json).read_text(encoding="utf-8"))
        elif req.transcript_pdf:
            from scheduler.transcript import parse_transcript
            raw = parse_transcript(str(req.transcript_pdf))
        else:
            raise ValueError("PlannerRequest must set transcript_json or transcript_pdf.")

        self._raw = raw
        self._student = Student(
            program=raw.get("program", "BSCS-DM"),
            admit_term=str(raw.get("admit_term", "202101")),
            completed=set(raw.get("completed_for_eligibility", raw.get("completed", []))),
            in_progress=set(raw.get("in_progress", [])),
            cumulative_credits=float(raw.get("cumulative_credits", 0.0)),
        )

    def _ensure_heavy_state(self) -> None:
        """Load graph, offerings, retriever, requirements — once."""
        if self._graph is None:
            full = DATA_DIR / "SU_full_graph.gpickle"
            prog = DATA_DIR / f"{self._student.program}_202601_graph.gpickle"
            path = full if full.exists() else prog
            print(f"[Session] Loading prereq graph from {path.name}…")
            with path.open("rb") as f:
                self._graph = pickle.load(f)

        if self._offerings is None:
            self._offerings = load_offerings()

        if self._retriever is None:
            print("[Session] Loading retriever…")
            self._retriever = Retriever.load(device="cpu")

        if self._remaining is None:
            self._remaining = compute_remaining(
                program=self._student.program,
                completed_for_eligibility=self._student.completed,
                in_progress=self._student.in_progress,
            )

        # Build eligible pool if empty
        if not self._eligible_pool:
            offered = likely_offered_in(
                self._offerings,
                self.base_request.target_term,
                same_season_only=True,
            )
            self._eligible_pool = eligible_courses(
                self._graph, self._student,
                options=EligibilityOptions(),
                candidate_pool=offered,
            )
            if self.base_request.include_required:
                self._required_injected = {
                    c for c in self._remaining.required_left
                    if c in self._eligible_pool
                }

    # ------------------------------------------------------------------ #
    # Public properties (safe access with auto-load)
    # ------------------------------------------------------------------ #

    @property
    def student(self) -> Student:
        return self._student

    @property
    def retriever(self) -> Retriever:
        self._ensure_heavy_state()
        return self._retriever  # type: ignore[return-value]

    @property
    def graph(self) -> nx.DiGraph:
        self._ensure_heavy_state()
        return self._graph  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def handle_turn(self, user_message: str) -> str:
        """Process one user turn and return the assistant's response.

        Routes to either the legacy intent-classifier pipeline or the ReAct
        tool-using loop depending on ``base_request.mode``. In both cases the
        (user, assistant) pair is appended to ``self.history``.
        """
        self._ensure_heavy_state()

        if self.base_request.mode == "react":
            from scheduler.react_agent import handle_turn as react_handle_turn
            response = react_handle_turn(self, user_message)
        else:
            from scheduler.intents import (
                classify_intent,
                handle_plan, handle_swap, handle_drop,
                handle_add, handle_explain, handle_unknown,
            )

            intent = classify_intent(
                user_message,
                self.history,
                model=self.base_request.intent_model,
            )

            dispatch = {
                "plan":    handle_plan,
                "swap":    handle_swap,
                "drop":    handle_drop,
                "add":     handle_add,
                "explain": handle_explain,
                "unknown": handle_unknown,
            }
            handler = dispatch.get(intent.intent, handle_unknown)
            response = handler(self, intent)

        # Update conversation history
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": response})

        # Keep history bounded (last 20 messages = 10 turns)
        if len(self.history) > 20:
            self.history = self.history[-20:]

        return response

    # ------------------------------------------------------------------ #
    # State snapshot (for debugging / serialisation)
    # ------------------------------------------------------------------ #

    def state_summary(self) -> dict:
        return {
            "program": self._student.program if self._student else None,
            "target_term": self.base_request.target_term,
            "eligible_pool_size": len(self._eligible_pool),
            "current_plan": self.current_plan.plan if self.current_plan else None,
            "history_turns": len(self.history) // 2,
        }
