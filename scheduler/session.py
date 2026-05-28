"""Stateful conversational planner session."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

from scheduler.catalog import Catalog
from scheduler.eligibility import EligibilityOptions, Student, eligible_courses
from scheduler.graph_selector import SelectedGraphs, normalize_program_code, select_graphs
from scheduler.offerings import likely_offered_in, load_offerings
from scheduler.requirements import RequirementsReport, compute_remaining
from scheduler.schemas import PlannerRequest, TermPlan

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


@dataclass
class PlannerSession:
    """All state needed across conversational turns."""

    base_request: PlannerRequest

    _student: Student | None = field(default=None, repr=False)
    _raw: dict = field(default_factory=dict, repr=False)
    transcript_loaded: bool = False
    known_interests: list[str] = field(default_factory=list)
    last_retrieved: list[dict] = field(default_factory=list)

    _graph: nx.DiGraph | None = field(default=None, repr=False)
    _selected_graphs: SelectedGraphs | None = field(default=None, repr=False)
    _offerings: dict | None = field(default=None, repr=False)
    _retriever: Any | None = field(default=None, repr=False)
    _catalog: Catalog | None = field(default=None, repr=False)
    _retriever_error: str | None = None
    _remaining: RequirementsReport | None = field(default=None, repr=False)
    _eligible_pool: set[str] = field(default_factory=set, repr=False)
    _required_injected: set[str] = field(default_factory=set, repr=False)
    _candidates_cache: list = field(default_factory=list, repr=False)

    current_plan: TermPlan | None = None
    history: list[dict] = field(default_factory=list)
    planning_allowed: bool = False
    last_trace: list[dict] = field(default_factory=list)

    @classmethod
    def from_request(cls, request: PlannerRequest) -> "PlannerSession":
        session = cls(base_request=request)
        if request.transcript_json or request.transcript_pdf:
            session._load_student()
        return session

    def _load_student(self) -> None:
        req = self.base_request
        if req.transcript_json:
            raw = json.loads(Path(req.transcript_json).read_text(encoding="utf-8"))
        elif req.transcript_pdf:
            from scheduler.transcript import parse_transcript
            raw = parse_transcript(str(req.transcript_pdf))
        else:
            return

        self._raw = raw
        self._student = Student(
            program=normalize_program_code(raw.get("program")) or "BSCS-DM",
            admit_term=str(raw.get("admit_term", "202101")),
            completed=set(raw.get("completed_for_eligibility", raw.get("completed", []))),
            in_progress=set(raw.get("in_progress", [])),
            cumulative_credits=float(raw.get("cumulative_credits", 0.0)),
        )
        self.transcript_loaded = True
        self._reset_academic_state()

    def load_student_from_path(self, path: Path) -> None:
        is_json = path.suffix.lower() == ".json"
        self.base_request.transcript_json = path if is_json else None
        self.base_request.transcript_pdf = path if not is_json else None
        self._load_student()

    def update_manual_context(
        self,
        program: str | None = None,
        admit_term: str | None = None,
        completed: set[str] | None = None,
        in_progress: set[str] | None = None,
    ) -> None:
        if self._student is None:
            self._student = Student()
        if program:
            self._student.program = normalize_program_code(program) or program.upper().strip()
        if admit_term:
            self._student.admit_term = str(admit_term).strip()
        if completed is not None:
            self._student.completed = set(completed)
        if in_progress is not None:
            self._student.in_progress = set(in_progress)
        self._reset_academic_state()

    def _reset_academic_state(self) -> None:
        self._graph = None
        self._selected_graphs = None
        self._remaining = None
        self._eligible_pool = set()
        self._required_injected = set()
        self._candidates_cache = []

    def has_course_context(self) -> bool:
        return bool(
            self.transcript_loaded
            or (self._student and (self._student.completed or self._student.in_progress))
        )

    def _ensure_heavy_state(self) -> None:
        if self._offerings is None:
            self._offerings = load_offerings()

        if self._catalog is None:
            self._catalog = Catalog.load()

        if self._student is None:
            return

        if self._selected_graphs is None or self._graph is None:
            self._selected_graphs = select_graphs(
                self._student.program,
                self._student.admit_term,
            )
            self._graph = self._selected_graphs.merged

        if self._remaining is None:
            self._remaining = compute_remaining(
                program=self._student.program,
                completed_for_eligibility=self._student.completed,
                in_progress=self._student.in_progress,
                admit_term=self._student.admit_term,
            )

        if not self._eligible_pool and self.has_course_context():
            offered = likely_offered_in(
                self._offerings,
                self.base_request.target_term,
                same_season_only=True,
            )
            scoped_candidates = self._selected_graphs.all_requirement_courses()
            self._eligible_pool = eligible_courses(
                self._graph,
                self._student,
                options=EligibilityOptions(),
                candidate_pool=scoped_candidates & offered,
            )
            if self.base_request.include_required:
                self._required_injected = {
                    c for c in self._remaining.required_left
                    if c in self._eligible_pool
                }

    def _ensure_retriever(self) -> Any | None:
        if self._retriever is not None or self._retriever_error is not None:
            return self._retriever
        try:
            from scheduler.retriever import Retriever
            self._retriever = Retriever.load(device="cpu")
        except Exception as exc:
            self._retriever_error = f"{type(exc).__name__}: {exc}"
            self._retriever = None
        return self._retriever

    @property
    def student(self) -> Student | None:
        return self._student

    @property
    def retriever(self) -> Any | None:
        self._ensure_heavy_state()
        return self._ensure_retriever()

    @property
    def catalog(self) -> Catalog:
        self._ensure_heavy_state()
        return self._catalog  # type: ignore[return-value]

    @property
    def graph(self) -> nx.DiGraph:
        self._ensure_heavy_state()
        if self._graph is None:
            raise ValueError("No degree graph selected yet; degree/admit term required.")
        return self._graph

    def handle_turn(self, user_message: str) -> str:
        self._ensure_heavy_state()

        if self.base_request.mode == "react":
            from scheduler.react_agent import handle_turn as react_handle_turn
            response = react_handle_turn(self, user_message)
        else:
            from scheduler.intents import (
                classify_intent,
                handle_add,
                handle_drop,
                handle_explain,
                handle_plan,
                handle_swap,
                handle_unknown,
            )

            intent = classify_intent(user_message, self.history, model=self.base_request.intent_model)
            dispatch = {
                "plan": handle_plan,
                "swap": handle_swap,
                "drop": handle_drop,
                "add": handle_add,
                "explain": handle_explain,
                "unknown": handle_unknown,
            }
            response = dispatch.get(intent.intent, handle_unknown)(self, intent)

        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": response})
        if len(self.history) > 100:
            self.history = self.history[-100:]
        return response

    def state_summary(self) -> dict:
        return {
            "program": self._student.program if self._student else None,
            "admit_term": self._student.admit_term if self._student else None,
            "cohort_term": self._selected_graphs.cohort_term if self._selected_graphs else None,
            "transcript_loaded": self.transcript_loaded,
            "target_term": self.base_request.target_term,
            "eligible_pool_size": len(self._eligible_pool),
            "current_plan": self.current_plan.plan if self.current_plan else None,
            "history_turns": len(self.history) // 2,
            "retriever_error": self._retriever_error,
            "last_trace": self.last_trace,
        }
