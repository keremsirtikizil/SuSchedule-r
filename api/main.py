"""
SuSchedule-r Web API — FastAPI backend.

Endpoints
---------
GET  /                          Serve the frontend SPA.
POST /session                   Create a session (term + credit config).
POST /session/{id}/transcript   Upload transcript JSON or PDF.
POST /session/{id}/turn         Send a chat message, get planner response.
GET  /session/{id}/state        Get current session state + plan.
DELETE /session/{id}            Clean up a session.

Sessions are held in memory — sufficient for a demo. A restart clears all
sessions. The heavy objects (retriever, graph) are singletons loaded once
at first use and reused across sessions.

Run:
    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# App setup
# --------------------------------------------------------------------------- #

app = FastAPI(title="SuSchedule-r", version="1.0")
app.mount("/static", StaticFiles(directory=ROOT / "api" / "static"), name="static")

# In-memory session store: {session_id: PlannerSession}
_sessions: dict[str, object] = {}


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #

class CreateSessionRequest(BaseModel):
    term: str = "202601"
    min_credits: float = 12.0
    max_credits: float = 21.0
    target_credits: float = 17.0
    planner_model: str = "o4-mini"
    intent_model: str = "gpt-4o-mini"
    include_required: bool = True
    # "pipeline" = legacy 8-stage agent; "react" = tool-using loop
    mode: str = "react"


class CreateSessionResponse(BaseModel):
    session_id: str
    term: str
    term_label: str


class TranscriptResponse(BaseModel):
    ok: bool
    name: str
    program: str
    semester: int
    cgpa: float
    completed_count: int
    in_progress: list[str]
    required_left: list[str]
    message: str


class TurnRequest(BaseModel):
    message: str


class TurnResponse(BaseModel):
    response: str
    plan: Optional[list[str]]
    total_credits: Optional[float]
    validation_ok: Optional[bool]
    warnings: list[str]
    intent: str
    token_usage: dict
    trace: list[dict] = []
    timetable: Optional[dict] = None


def _latest_timetable(trace: list[dict]) -> Optional[dict]:
    """Return the most recent successful build_timetable result from the trace.

    Lets the UI render a weekly grid whenever the agent built a timetable this
    turn — whether or not a full plan was committed.
    """
    for ev in reversed(trace or []):
        if ev.get("type") == "tool_result" and ev.get("name") == "build_timetable":
            result = ev.get("result") or {}
            if isinstance(result, dict) and "error" not in result:
                return result
    return None


class StateResponse(BaseModel):
    session_id: str
    program: Optional[str]
    target_term: str
    term_label: str
    eligible_pool_size: int
    history_turns: int
    plan: Optional[list[str]]
    reasoning: Optional[dict]
    total_credits: Optional[float]
    validation_ok: Optional[bool]
    warnings: list[str]
    has_transcript: bool
    last_trace: list[dict] = []


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = ROOT / "api" / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/session", response_model=CreateSessionResponse)
async def create_session(req: CreateSessionRequest):
    """Create a new planning session and return its ID."""
    from scheduler.prompts import term_label

    session_id = str(uuid.uuid4())[:8]

    from scheduler.schemas import PlannerRequest
    from scheduler.session import PlannerSession

    planner_req = PlannerRequest(
        target_term=req.term,
        user_request="",
        min_credits=req.min_credits,
        max_credits=req.max_credits,
        target_credits=req.target_credits,
        planner_model=req.planner_model,
        intent_model=req.intent_model,
        include_required=req.include_required,
        mode=req.mode,
    )

    _sessions[session_id] = {
        "config": req,
        "session": PlannerSession.from_request(planner_req),
        "transcript_raw": None,
    }

    return CreateSessionResponse(
        session_id=session_id,
        term=req.term,
        term_label=term_label(req.term),
    )


@app.post("/session/{session_id}/transcript", response_model=TranscriptResponse)
async def upload_transcript(session_id: str, file: UploadFile = File(...)):
    """Upload a Degree Evaluation HTML, transcript JSON, or PDF and initialize
    the planning session.

    The file type is detected by extension (.html/.htm -> Degree Evaluation,
    .json -> parsed transcript, otherwise -> Academic Records PDF). The Degree
    Evaluation is preferred: it also carries the per-degree Engineering /
    Basic-Science ECTS requirements used by science/engineering credit tracking.
    """
    if session_id not in _sessions:
        raise HTTPException(404, f"Session {session_id} not found.")

    slot = _sessions[session_id]
    config: CreateSessionRequest = slot["config"]

    content = await file.read()
    suffix = Path(file.filename or "t.json").suffix.lower()

    # Save to a session-keyed persistent path (NOT a temp file that gets deleted).
    # agent.plan() re-reads the transcript on every "plan" intent turn, so the
    # file must survive for the lifetime of the session.
    # It is cleaned up in DELETE /session/{id}.
    persistent_path = Path(tempfile.gettempdir()) / f"suscheduler_{session_id}{suffix}"
    persistent_path.write_bytes(content)

    try:
        from scheduler.requirements import compute_remaining

        session = slot.get("session")
        if session is None:
            from scheduler.schemas import PlannerRequest
            from scheduler.session import PlannerSession
            session = PlannerSession.from_request(PlannerRequest(
                target_term=config.term,
                user_request="",
                min_credits=config.min_credits,
                max_credits=config.max_credits,
                target_credits=config.target_credits,
                planner_model=config.planner_model,
                intent_model=config.intent_model,
                include_required=config.include_required,
                mode=config.mode,
            ))
        session.load_student_from_path(persistent_path)
        raw = session._raw

        remaining = compute_remaining(
            program=session.student.program,
            completed_for_eligibility=session.student.completed,
            in_progress=session.student.in_progress,
            admit_term=session.student.admit_term,
        )

        slot["session"] = session
        slot["transcript_raw"] = raw
        slot["transcript_path"] = persistent_path   # kept for DELETE cleanup

        return TranscriptResponse(
            ok=True,
            name=raw.get("name", "Student"),
            program=session.student.program,
            semester=raw.get("current_semester", 0),
            cgpa=raw.get("cgpa", 0.0),
            completed_count=len(session.student.completed),
            in_progress=sorted(session.student.in_progress),
            required_left=remaining.required_left,
            message=f"Transcript loaded for {raw.get('name', 'Student')}.",
        )
    except Exception as exc:
        persistent_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Failed to parse transcript: {exc}")


@app.post("/session/{session_id}/turn", response_model=TurnResponse)
async def handle_turn(session_id: str, req: TurnRequest):
    """Send a message to the planner and get a response."""
    if session_id not in _sessions:
        raise HTTPException(404, f"Session {session_id} not found.")

    slot = _sessions[session_id]
    session = slot.get("session")
    if session is None:
        raise HTTPException(400, "Session is not initialized.")

    from scheduler import llm_client

    usage_before = llm_client.get_session_usage()["total_tokens"]

    # In pipeline mode, classify intent up-front so we can return it in the
    # response badge. In react mode the model handles its own reasoning —
    # skip the extra gpt-4o-mini call.
    if session.base_request.mode == "react":
        intent_label = "react"
    else:
        from scheduler.intents import classify_intent
        intent_obj = classify_intent(
            req.message,
            session.history,
            model=session.base_request.intent_model,
        )
        intent_label = intent_obj.intent

    try:
        response_text = session.handle_turn(req.message)
    except Exception as exc:
        raise HTTPException(500, f"Planner error: {exc}")

    usage = llm_client.get_session_usage()
    plan = session.current_plan
    return TurnResponse(
        response=response_text,
        plan=plan.plan if plan else None,
        total_credits=plan.total_credits if plan else None,
        validation_ok=plan.validation_ok if plan else None,
        warnings=plan.warnings if plan else [],
        intent=intent_label,
        token_usage={
            **usage,
            "this_turn": usage["total_tokens"] - usage_before,
        },
        trace=session.last_trace,
        timetable=_latest_timetable(session.last_trace),
    )


@app.get("/session/{session_id}/state", response_model=StateResponse)
async def get_state(session_id: str):
    """Return the current session state and plan."""
    if session_id not in _sessions:
        raise HTTPException(404, f"Session {session_id} not found.")

    from scheduler.prompts import term_label

    slot = _sessions[session_id]
    session = slot.get("session")
    config: CreateSessionRequest = slot["config"]

    if session is None:
        return StateResponse(
            session_id=session_id,
            program=None,
            target_term=config.term,
            term_label=term_label(config.term),
            eligible_pool_size=0,
            history_turns=0,
            plan=None,
            reasoning=None,
            total_credits=None,
            validation_ok=None,
            warnings=[],
            has_transcript=False,
            last_trace=[],
        )

    summary = session.state_summary()
    plan = session.current_plan

    return StateResponse(
        session_id=session_id,
        program=summary["program"],
        target_term=config.term,
        term_label=term_label(config.term),
        eligible_pool_size=summary["eligible_pool_size"],
        history_turns=summary["history_turns"],
        plan=plan.plan if plan else None,
        reasoning=plan.reasoning if plan else None,
        total_credits=plan.total_credits if plan else None,
        validation_ok=plan.validation_ok if plan else None,
        warnings=plan.warnings if plan else [],
        has_transcript=summary["transcript_loaded"],
        last_trace=summary.get("last_trace", []),
    )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    slot = _sessions.pop(session_id, {})
    # Clean up the persistent transcript file if one was saved
    transcript_path: Path | None = slot.get("transcript_path")
    if transcript_path:
        transcript_path.unlink(missing_ok=True)
    return {"ok": True}
