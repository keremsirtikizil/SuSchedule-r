# SuSchedule-r

A **conversational course-planning agent for Sabancı University** — built as the term project for **CS 455 / CS 555: Large Language Models (Spring 2025/2026)**.

The system reads a student's PDF transcript, retrieves semantically relevant courses from the SU catalog via a two-stage RAG pipeline, and builds a graduation-aligned semester plan through either a **fixed 8-stage pipeline** or a **tool-using ReAct loop** backed by GPT-4o. Plans are validated deterministically against real prereq/coreq rules before being shown to the student.

---

## Table of Contents

1. [What this system does](#1-what-this-system-does)
2. [Architecture](#2-architecture)
3. [Repository layout](#3-repository-layout)
4. [Quick start](#4-quick-start)
5. [Data layer (scraper pipeline)](#5-data-layer-scraper-pipeline)
6. [FAISS index (embeddings)](#6-faiss-index-embeddings)
7. [Scheduler modules](#7-scheduler-modules)
8. [Agent modes](#8-agent-modes)
9. [CLI usage](#9-cli-usage)
10. [Web UI](#10-web-ui)
11. [Data formats](#11-data-formats)
12. [Verification & regression checks](#12-verification--regression-checks)
13. [Credits](#13-credits)

---

## 1. What this system does

| Concern | Where handled | Technology |
|---|---|---|
| Discover every SU course | `scraper/` | BeautifulSoup + Banner HTML |
| Parse prereq text → boolean tree | `scraper/prereq_parser.py` | Recursive-descent parser |
| Build prereq DAG | `scraper/build_graph.py` | NetworkX DiGraph |
| Eligibility check (hard rules) | `scheduler/eligibility.py` | Symbolic (no LLM) |
| Timetable / conflict check | `scheduler/timetable.py` | Symbolic (no LLM) |
| Transcript → student profile | `scheduler/transcript.py` | pdfplumber |
| Semantic course retrieval | `scheduler/retriever.py` | FAISS + cross-encoder |
| Graduation requirements | `scheduler/requirements.py` | Degree graph slices |
| Plan proposal | `scheduler/agent.py` | GPT-4o structured outputs |
| Plan repair loop | `scheduler/agent.py` | GPT-4o → validate → retry |
| Conversational session | `scheduler/session.py` | Stateful, history-aware |
| Intent dispatch (pipeline mode) | `scheduler/intents.py` | GPT-4o-mini classifier |
| ReAct tool loop | `scheduler/react_agent.py` | GPT-4o with function calling |
| Web API | `api/main.py` | FastAPI + uvicorn |
| Web frontend | `api/static/` | Vanilla JS, dark-theme SPA |

---

## 2. Architecture

### 2a. Data layer

```
Sabancı Banner
     │ HTTP (cached on disk)
     ▼
scraper/discover.py        →  data/discovery_manifest.json   (688 courses, 14 programs)
scraper/scrape_full.py     →  data/SU_full_catalog.json
scraper/build_graph.py     →  data/SU_full_graph.gpickle     (721 nodes, 729 prereq edges)
scraper/build_degree_graphs.py → data/degree_graphs/<PROGRAM>/{Required,Core_Elective,...}.gpickle
scraper/scrape_offerings.py→  data/offerings_<TERM>.json     (per-term CRN / meeting data)
```

### 2b. RAG index (built once in Colab)

```
data/SU_full_catalog.json
     │
     ├─ BAAI/bge-base-en-v1.5  (bi-encoder, 768-dim, L2-normalised)
     │        │
     │        ▼
     │  embeddings/su_courses.index      (FAISS IndexFlatIP, 688 vectors)
     │  embeddings/su_courses.parquet    (metadata DataFrame)
     │  embeddings/su_courses_embeddings.npy
     │  embeddings/id_map.json
     │
     └─ BAAI/bge-reranker-base  (cross-encoder, re-ranks top-k at query time)
```

### 2c. Planning session

```
User turn
   │
   ▼
PlannerSession.handle_turn()
   │
   ├── mode="pipeline" ──────────────────────────────────────────────────────┐
   │       │                                                                  │
   │   classify_intent()  (gpt-4o)                                            │
   │       │                                                                  │
   │   ┌──────────────┐                                                       │
   │   │ plan intent  │──▶ agent.plan() ──────────────────────────────────────┤
   │   │              │    Stage 1: load transcript                            │
   │   │              │    Stage 2: compute_remaining()                        │
   │   │              │    Stage 3: eligible pool (eligibility + offerings)    │
   │   │              │    Stage 4: RAG (intent split → retrieve → rerank)     │
   │   │              │    Stage 5: gpt-4o structured output → PlannerOutput   │
   │   │              │    Stage 6: validate_plan()                            │
   │   │              │    Stage 7: repair loop (max 3 iters)                  │
   │   │              │    Stage 8: timetable resolution                       │
   │   │ swap/drop/add│──▶ hardcoded handler + re-validate                    │
   │   │ explain      │──▶ read session.current_plan.reasoning                 │
   │   └──────────────┘                                                       │
   │                                                                          │
   └── mode="react" ──────────────────────────────────────────────────────────┘
           │
       react_agent.handle_turn()
           │
       [system prompt + history]
           │
       ┌────────────────────────────────────────────────────────────────┐
       │  ReAct loop (max 12 steps)                                     │
       │                                                                │
       │  call_with_tools(gpt-4o, messages, TOOLS)                      │
       │       │                                                        │
       │  if tool_calls → dispatch → append tool results → repeat       │
       │  if no tool_calls → return final assistant message             │
       └────────────────────────────────────────────────────────────────┘
```

**Tools available to the ReAct agent:**

| Tool | What it does |
|---|---|
| `get_student_profile()` | Transcript snapshot (program, completed, in-progress, CGPA) |
| `get_remaining_requirements()` | What's still needed for graduation |
| `get_current_plan()` | Currently committed plan + per-course reasoning |
| `retrieve_courses(query, k, subj)` | Two-stage semantic search (FAISS → cross-encoder) |
| `check_prereqs(code)` | Full prereq/coreq text + whether student is eligible |
| `get_offerings(code)` | Historical offering terms + likely-offered flag |
| `validate_plan(plan)` | Eligibility check — ok/violations/total_credits |
| `set_plan(plan, reasoning, summary)` | Commit a validated plan to the session |

---

## 3. Repository layout

```
SuSchedule-r/
├── README.md
├── LICENSE                        ← MIT
├── requirements.txt
├── .env.example                   ← copy to .env and add OPENAI_API_KEY
│
├── notebooks/
│   └── build_faiss_index.ipynb   ← Colab notebook to build the FAISS index
│
├── embeddings/                    ← built by the notebook; commit parquet + id_map only
│   ├── su_courses.parquet         ← 688-row metadata DataFrame
│   ├── id_map.json                ← row_int → course_code
│   ├── su_courses_embeddings.npy  ← gitignored (2 MB, rebuild from notebook)
│   └── su_courses.index           ← gitignored (2 MB, rebuild from notebook)
│
├── scheduler/
│   ├── __init__.py
│   ├── transcript.py              ← PDF transcript → student profile JSON
│   ├── eligibility.py             ← prereq / coreq / credit-load checker (symbolic)
│   ├── offerings.py               ← season-aware "likely offered?" filter
│   ├── timetable.py               ← time-conflict detection, feasible CRN combos
│   ├── requirements.py            ← graduation requirements per program
│   ├── retriever.py               ← FAISS bi-encoder + bge-reranker-base (two-stage RAG)
│   ├── schemas.py                 ← Pydantic + dataclass I/O models (PlannerRequest, TermPlan, …)
│   ├── llm_client.py              ← OpenAI wrapper (structured outputs, tool calls, retry, usage)
│   ├── prompts.py                 ← all system / user prompt templates
│   ├── agent.py                   ← 8-stage pipeline (Phase A/B entry point)
│   ├── intents.py                 ← intent classifier + swap/drop/add/explain handlers
│   ├── session.py                 ← stateful conversational session (routes pipeline vs ReAct)
│   ├── react_agent.py             ← ReAct tool-using loop + Toolbox class
│   └── cli.py                     ← interactive REPL with slash commands
│
├── api/
│   ├── __init__.py
│   ├── main.py                    ← FastAPI backend (6 REST endpoints)
│   └── static/
│       ├── index.html             ← SPA shell
│       ├── style.css              ← dark-theme CSS
│       └── app.js                 ← drag-and-drop upload, chat, plan sidebar
│
├── scraper/                       ← data-layer pipeline (run once)
│   ├── discover.py                ← Stage 1: enumerate all (subj, num) pairs
│   ├── scrape_full.py             ← Stage 2: fetch each course-detail page
│   ├── prereq_parser.py           ← prereq text → boolean expression tree
│   ├── build_graph.py             ← Stage 3: unified prereq DAG
│   ├── build_degree_graphs.py     ← Stage 3b: per-(program, section) sub-DAGs
│   ├── audit_prereqs.py           ← parser coverage diagnostic
│   └── scrape_offerings.py        ← per-term CRN / meeting scraper
│
├── data/
│   ├── SU_full_catalog.json       ← 688 courses, structured
│   ├── SU_full_graph.gpickle      ← 721-node prereq DAG (NetworkX)
│   ├── degree_graphs/             ← per-(program, section) sub-DAGs
│   │   ├── BSCS-DM/
│   │   │   ├── Required.gpickle
│   │   │   ├── Core_Elective.gpickle
│   │   │   ├── Area_Elective.gpickle
│   │   │   └── Free_Elective.gpickle
│   │   └── … (14 programs × 4 sections)
│   ├── offerings_*.json           ← per-term section listings (202401 → 202503)
│   └── transcript_cagan.json      ← sample parsed transcript
│
├── courses/                       ← cached course-detail HTML pages
├── degrees/                       ← cached degree-requirements HTML pages
├── pools/                         ← cached elective-pool HTML pages
└── offerings/                     ← cached per-term section HTML pages
```

---

## 4. Quick start

### Prerequisites

- Python 3.10+
- An OpenAI API key (GPT-4o access required)

### Install

```bash
git clone git@github.com:keremsirtikizil/SuSchedule-r.git
cd SuSchedule-r
git checkout kero

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Set your API key

```bash
cp .env.example .env
# then edit .env and set:
# OPENAI_API_KEY=your_openai_api_key_here
```

### (Re)build the FAISS index — one time only

Open `notebooks/build_faiss_index.ipynb` in Google Colab (free GPU/CPU runtime works fine), run all cells, then download the four output files from `/content/out/` and place them in `embeddings/`:

```
embeddings/
├── su_courses.parquet         ✓ committed
├── id_map.json                ✓ committed
├── su_courses_embeddings.npy  ← download from Colab
└── su_courses.index           ← download from Colab
```

---

## 5. Data layer (scraper pipeline)

The scraper runs **once** to build the catalog. All outputs are already committed; only re-run if Sabancı updates its Banner pages.

```bash
source .venv/bin/activate

# Stage 1a: enumerate courses from degree pages (~30 s)
python -m scraper.discover --degrees

# Stage 1b (optional): brute-force subject ranges for orphan courses (~1.5 h)
python -m scraper.discover --subjects

# Stage 2: fetch each course-detail page (cached in courses/)
python -m scraper.scrape_full

# Stage 3a: unified prereq DAG (721 nodes, 729 edges, 100% parse coverage)
python -m scraper.build_graph --full

# Stage 3b: per-(degree, section) sub-DAGs
python -m scraper.build_degree_graphs

# Audit parser coverage (should report: strict=426, all others=0 failures)
python -m scraper.audit_prereqs
```

Every step is **idempotent and resumable** — HTML pages are cached; Ctrl-C mid-run is safe.

---

## 6. FAISS index (embeddings)

Built in `notebooks/build_faiss_index.ipynb` using **BAAI/bge-base-en-v1.5** (768-dim).

Embedding text per course:
```
<CODE> — <Title>
<description>
```

Index type: `IndexFlatIP` (exact cosine search via L2-normalised inner product — fine for 688 vectors).

At query time `scheduler/retriever.py` runs two stages:
1. **Bi-encoder** — encode query with BGE query prefix, score against all eligible courses in the FAISS index.
2. **Cross-encoder** — `BAAI/bge-reranker-base` re-ranks the top-k bi-encoder hits with full pair attention.

> ⚠️ **macOS Apple Silicon note:** `SentenceTransformer` must be imported **before** `faiss` in the same process. Both link OpenBLAS; double-init causes a segfault (exit code 139). `retriever.py` handles this — don't reorder the imports.

---

## 7. Scheduler modules

| Module | Role | LLM? |
|---|---|---|
| `transcript.py` | Parse Sabancı "Academic Records" PDF → student profile dict | No |
| `eligibility.py` | Prereq / coreq / credit-load check; `validate_plan()` | No |
| `offerings.py` | "Is course X likely offered in Fall 26-27?" | No |
| `timetable.py` | Find a CRN combo with no time conflicts | No |
| `requirements.py` | Graduation requirements left by program | No |
| `retriever.py` | Two-stage RAG (FAISS + cross-encoder) | No |
| `schemas.py` | `PlannerRequest`, `TermPlan`, LLM output schemas | — |
| `llm_client.py` | OpenAI wrapper with retry + usage tracking | — |
| `prompts.py` | All system/user prompt templates | — |
| `agent.py` | 8-stage fixed pipeline (`plan()`) | GPT-4o |
| `intents.py` | Intent classifier + swap/drop/add/explain handlers | GPT-4o-mini |
| `session.py` | Stateful session; routes pipeline vs ReAct per turn | — |
| `react_agent.py` | ReAct loop + toolbox; all intents handled by the model | GPT-4o |
| `cli.py` | Interactive REPL | — |

---

## 8. Agent modes

`PlannerRequest.mode` selects the agent style. Default is `"pipeline"`.

### `mode="pipeline"` (default)

Fixed 8-stage execution:
1. Load student profile from transcript
2. Compute graduation requirements remaining
3. Build eligible + offered candidate pool
4. Parse user intent into retrieval queries (gpt-4o) → retrieve + re-rank
5. GPT-4o proposes a plan (structured output — guaranteed JSON schema)
6. Validate plan against hard rules (`eligibility.validate_plan`)
7. Repair loop if violations (up to 3 iterations)
8. Best-effort timetable resolution

For subsequent turns (swap / drop / add / explain), a lightweight intent classifier routes to a hardcoded handler — no full re-planning.

### `mode="react"` (tool-using ReAct loop)

The model receives the same tool suite at every turn and decides what to call and in what order. It can:
- Search for courses across multiple queries before deciding
- Look up prereqs for specific codes before committing
- Check offering history to verify a course will actually run
- Validate a plan, get the violation list, and fix it without being explicitly told to retry

The loop runs up to **8 steps** per turn. On the final step with no tool calls the model's text is returned to the user.

`set_plan()` includes a re-validation guardrail — it will not commit a plan that `validate_plan()` would reject, even if the model skips the explicit validation step.

---

## 9. CLI usage

```bash
source .venv/bin/activate

# Interactive REPL (ReAct mode, default)
python -m scheduler.cli \
    --transcript data/transcript_cagan.json \
    --term 202601

# Live debug trace for demos: every tool call, arguments, result, repair, final answer
python -m scheduler.cli \
    --transcript data/transcript_cagan.json \
    --term 202601 \
    --debug-trace

# With an automatic first message (useful for demos)
python -m scheduler.cli \
    --transcript data/transcript_cagan.json \
    --debug-trace \
    --first-message "Plan my fall semester. I want ML and a databases course."

# Full untruncated raw JSON trace, then exit after the first message
python -m scheduler.cli \
    --transcript data/transcript_cagan.json \
    --debug-trace \
    --raw-trace \
    --one-shot \
    --first-message "Suggest CS-coded Core Electives about security."
```

### Slash commands inside the REPL

| Command | What it does |
|---|---|
| `/plan` | Re-run planning (same as typing a plan request) |
| `/state` | Print current plan + session summary |
| `/trace` | Pretty-print the last agent/tool trace |
| `/raw-trace` | Print the last full trace as raw JSON |
| `/json` | Print the current plan as raw JSON |
| `/usage` | Print token usage for the session |
| `/help` | List commands |
| `/exit` or Ctrl-D | Quit |

### One-shot agent (no REPL)

```bash
python -m scheduler.agent \
    --transcript data/transcript_cagan.json \
    --request "I want ML and a database course, no early classes" \
    --term 202601 \
    --json   # raw JSON output
```

---

## 10. Web UI

### Run the server

```bash
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000
```

Then open [http://localhost:8000](http://localhost:8000).

### REST API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Serve the SPA |
| `POST` | `/session` | Create session (term, credit range, mode) |
| `POST` | `/session/{id}/transcript` | Upload transcript PDF or JSON |
| `POST` | `/session/{id}/turn` | Send a message, get a response + plan |
| `GET` | `/session/{id}/state` | Current session state + plan |
| `DELETE` | `/session/{id}` | Clean up |

### Selecting the agent mode via API

```jsonc
// POST /session
{
  "term": "202601",
  "min_credits": 12,
  "max_credits": 21,
  "target_credits": 17,
  "mode": "react"           // ← "pipeline" or "react"
}
```

### UI features

- **Drag-and-drop transcript upload** — PDF or JSON
- **Chat interface** with markdown rendering (marked.js)
- **Live plan sidebar** — updated after every turn that changes the plan
- **Intent badge** on each agent response (pipeline mode)
- **Token usage stats** in the sidebar

---

## 11. Data formats

### `data/transcript_cagan.json` (student profile)

```jsonc
{
  "name": "Student Name",
  "program": "BSCS-DM",
  "admit_term": "202101",
  "current_semester": 7,
  "cgpa": 3.40,
  "cumulative_credits": 127.0,
  "completed_for_eligibility": ["CS 201", "CS 201R", "CS 204", …],
  "in_progress": ["CS 412"],
  "minors": []
}
```

### `data/offerings_<TERM>.json`

```jsonc
{
  "CS 201": [
    {
      "crn": "10190",
      "section": "A",
      "meetings": [
        {
          "type": "lecture",
          "days": [0, 2],
          "start": 600,
          "end": 690,
          "days_raw": "MW",
          "where": "FENS L067",
          "instructor": "Erkay Savaş"
        }
      ]
    }
  ]
}
```

`start` / `end` are minutes since midnight. `days` is a list of weekday integers (0=Monday).

### `TermPlan` (agent output, also returned by the API)

```jsonc
{
  "plan": ["CS 306", "CS 412", "MATH 212", "CS 449", "ENS 492"],
  "reasoning": {
    "CS 306": "Required course; enables CS 406 and CS 437 next term.",
    "MATH 212": "Required — Differential Equations, needed for graduation."
  },
  "summary": "Five-course, 16-credit semester covering required milestones and one CS elective.",
  "total_credits": 16.0,
  "validation_ok": true,
  "violations": [],
  "auto_added_coreqs": [],
  "timetable": {
    "CS 306": { "crn": "12345", "section": "A", "meetings": […] }
  },
  "iterations": 0,
  "alternatives": ["CS 402", "CS 421"],
  "candidate_pool_size": 87,
  "warnings": ["No live section data for term 202601 — timetable skipped."]
}
```

---

## 12. Verification & regression checks

| Check | How |
|---|---|
| Prereq parser: 100% coverage | `python -m scraper.audit_prereqs` — strict=426, failures=0 |
| Full prereq graph is a DAG | `data/SU_full_graph_report.txt` → `Is DAG? True` |
| Eligibility smoke test | `python -m scheduler.eligibility` |
| Offerings smoke test | `python -m scheduler.offerings` |
| Retriever smoke test | `python -m scheduler.retriever` |

---

## 13. Credits

- **Course:** CS 455 / CS 555 — Large Language Models, Spring 2025/2026 (Sabancı University)
- **Instructor:** Inanç Arın
- **Models used:** BAAI/bge-base-en-v1.5 (bi-encoder), BAAI/bge-reranker-base (cross-encoder), OpenAI GPT-4o (planner + ReAct), OpenAI GPT-4o-mini (intent classifier)
- **License:** MIT — see `LICENSE`
