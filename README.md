# SuSchedule-r

A **conversational course-planning agent for Sabancı University** — built as the term project for **CS 455 / CS 555: Large Language Models (Spring 2025/2026)**.

The system reads a student's SU **Degree Evaluation HTML** page (preferred) or transcript PDF/JSON, selects the correct degree-requirement graphs for their program and entrance cohort, retrieves semantically relevant courses through a two-stage RAG pipeline, and answers through a **tool-using ReAct agent** backed by GPT-4o. Plans, prerequisite checks, credit calculations, and timetable conflicts are handled by deterministic Python modules rather than trusted to the LLM.

---

## Table of Contents

1. [What this system does](#1-what-this-system-does)
2. [Architecture](#2-architecture)
3. [Repository layout](#3-repository-layout)
4. [Quick start](#4-quick-start)
5. [Data layer (scraper pipeline)](#5-data-layer-scraper-pipeline)
6. [FAISS index (embeddings)](#6-faiss-index-embeddings)
7. [Python module guide](#7-python-module-guide)
8. [Agent modes](#8-agent-modes)
9. [CLI usage](#9-cli-usage)
10. [Web UI](#10-web-ui)
11. [Data formats](#11-data-formats)
12. [Evaluation & regression checks](#12-evaluation--regression-checks)
13. [Credits](#13-credits)

---

## 1. What this system does

| Concern | Where handled | Technology |
|---|---|---|
| Discover the SU catalog | `scraper/` | BeautifulSoup + Banner HTML |
| Parse prereq text → boolean tree | `scraper/prereq_parser.py` | Recursive-descent parser |
| Build prereq DAG | `scraper/build_graph.py` | NetworkX DiGraph |
| Select degree + entrance cohort scope | `scheduler/graph_selector.py` | Per-cohort degree DAGs |
| Degree Evaluation HTML → student profile | `scheduler/degree_eval.py` | BeautifulSoup |
| Eligibility check (hard rules) | `scheduler/eligibility.py` | Symbolic (no LLM) |
| Timetable / conflict check | `scheduler/timetable.py` | Symbolic (no LLM) |
| Engineering / Basic-Science ECTS | `scheduler/eng_sci.py` | Symbolic (no LLM) |
| Minor progress | `scheduler/minors.py` | Symbolic advisory lookup |
| Transcript PDF → fallback profile | `scheduler/transcript.py` | pdfplumber |
| Semantic course retrieval | `scheduler/retriever.py` | FAISS + cross-encoder |
| Graduation requirements | `scheduler/requirements.py` | Cohort-specific graph slices |
| Plan proposal | `scheduler/agent.py` | GPT-4o structured outputs |
| Plan repair loop | `scheduler/agent.py` | GPT-4o → validate → retry |
| Conversational session | `scheduler/session.py` | Stateful, history-aware |
| Intent dispatch (legacy pipeline) | `scheduler/intents.py` | GPT-4o classifier |
| ReAct tool loop | `scheduler/react_agent.py` | GPT-4o with function calling |
| Web API | `api/main.py` | FastAPI + uvicorn |
| Web frontend | `api/static/` | Vanilla JS, dark-theme SPA |
| Benchmark harness | `eval/` | Offline + API-consuming evaluation |

---

## 2. Architecture

The production path is the **ReAct agent**. A legacy fixed pipeline remains for
comparison and ablation experiments.

### 2a. End-to-end message flow

```text
User message or Degree Evaluation upload
        │
        ▼
PlannerSession
  - profile state outside chat history
  - at most 25 user/assistant turns
  - selected program + entrance cohort
  - current plan and UI schedule-builder picks
        │
        ├── upload: degree_eval.py / transcript.py
        │           └── profile, completed, in-progress, section-credit totals
        │
        ▼
react_agent.handle_turn()                         max 8 LLM iterations
        │
        ├── inject stable system prompt
        ├── inject runtime date / target term as a context message
        ├── inject loaded student profile as a context message
        └── prefetch RAG context for recommendation questions
                  │
                  ├── rewrite_retrieval_queries()
                  ├── select only data/degree_graphs/<PROGRAM>/<COHORT>/
                  ├── compute eligible + likely-offered candidate pool
                  └── FAISS bi-encoder → BGE cross-encoder reranker
        │
        ▼
GPT-4o tool loop
  - inspect requirements, prerequisites, offerings, minors, ECTS, or timetable
  - answer exploratory questions directly
  - create a plan only after explicit user permission
        │
        ▼
Deterministic guardrails
  - validate prereq AND/OR expression trees
  - reject completed courses and invalid credit loads
  - auto-add direct recitations/labs
  - check timetable conflicts using published or same-season proxy times
        │
        ▼
Chat response + compact UI trace + full raw CLI trace
```

### 2b. Data pipeline

```text
SU Banner HTML + SU catalog PDF
        │
        ├── scraper/discover.py
        │     └── data/discovery_manifest.json
        ├── scraper/scrape_full.py
        │     └── data/SU_full_catalog.json                 817 catalog rows
        ├── scraper/prereq_parser.py
        │     └── nested boolean prereq expression trees
        ├── scraper/build_graph.py
        │     └── data/SU_full_graph.gpickle                diagnostic fallback
        ├── scraper/build_degree_graphs.py
        │     └── data/degree_graphs/<PROGRAM>/<COHORT>/
        │           ├── Required.{json,gpickle}
        │           ├── Core_Elective.{json,gpickle}
        │           ├── Area_Elective.{json,gpickle}
        │           └── Free_Elective.{json,gpickle}
        ├── scraper/scrape_offerings.py
        │     └── data/offerings.json                       sections + meeting times
        ├── scraper/build_minors.py
        │     └── data/minors.json
        └── scraper/build_eng_sci.py
              └── data/eng_sci_credits.json                 per-course ECTS splits
```

Normal advising does **not** load the full prerequisite graph. It selects only
the student's degree and entrance cohort. Within each selected section graph:

- `in_slice=true` means the course is actually part of that requirement pool.
- `in_slice=false` means the node exists only because another in-slice course
  depends on it. It is kept for prerequisite traversal but is not recommended
  as a requirement-pool member.

### 2c. RAG pipeline

```text
data/SU_full_catalog.json
        │
        ├── scheduler/build_faiss_index.py
        │     ├── BAAI/bge-base-en-v1.5
        │     ├── embeddings/su_courses.parquet
        │     ├── embeddings/su_courses_embeddings.npy
        │     ├── embeddings/su_courses.index
        │     └── embeddings/id_map.json
        │
User wording
        │
        ├── internal LLM query rewrite (discarded after retrieval)
        ├── contextual filters
        │     ├── no profile: full catalog
        │     ├── degree known: selected degree/cohort sections
        │     └── academic profile known: eligible + likely-offered candidates
        ├── FAISS cosine search with BGE embeddings
        ├── BAAI/bge-reranker-base cross-encoder reranking
        └── GPT-4o answers the original user wording from retrieved context
```

The retriever refuses to load stale/misaligned artifacts. If the catalog,
metadata, vector matrix, or ID map disagree, rebuild with:

```bash
python -m scheduler.build_faiss_index
```

### 2d. ReAct tool groups

| Group | Tools |
|---|---|
| Student state | `get_student_profile`, `set_student_context` |
| Degree scope | `select_degree_graphs`, `get_requirement_state`, `get_degree_section_courses` |
| RAG | `rewrite_retrieval_queries`, `retrieve_catalog_courses`, `get_course_details` |
| Eligibility | `get_eligible_courses`, `check_prereqs`, `validate_courses`, `validate_plan`, `set_plan` |
| Offerings | `get_offering_pattern` |
| Minors and ECTS | `list_minors`, `get_minor_requirements`, `get_science_engineering_progress`, `find_courses_by_credit_type` |
| Timetable | `build_timetable`, `get_current_schedule`, `check_courses_against_current_schedule` |

`set_plan()` is permission-gated and re-validates before committing. The model
cannot silently create a schedule merely because the user asked an exploratory
question.

---

## 3. Repository layout

```
SuSchedule-r/
├── README.md
├── LICENSE                        ← MIT
├── requirements.txt
├── .env.example                   ← copy to .env and add OPENAI_API_KEY
├── DEGREE EVALUATION.html         ← sample rich academic-profile input
├── katalog_basic_eng_*.pdf        ← Engineering / Basic-Science ECTS source
│
├── notebooks/
│   └── build_faiss_index.ipynb    ← optional Colab index-building workflow
│
├── embeddings/
│   ├── su_courses.parquet         ← catalog metadata DataFrame
│   ├── id_map.json                ← row index → course code
│   ├── su_courses_embeddings.npy  ← versioned BGE passage vectors
│   └── su_courses.index           ← versioned FAISS IndexFlatIP
│
├── scheduler/
│   ├── __init__.py
│   ├── degree_eval.py             ← preferred Degree Evaluation HTML parser
│   ├── transcript.py              ← fallback PDF transcript parser
│   ├── graph_selector.py          ← program alias + entrance-cohort graph selector
│   ├── eligibility.py             ← prereq / coreq / credit-load checker (symbolic)
│   ├── offerings.py               ← season-aware "likely offered?" filter
│   ├── timetable.py               ← time-conflict detection, feasible CRN combos
│   ├── requirements.py            ← graduation pools per program + cohort
│   ├── eng_sci.py                 ← Engineering / Basic-Science ECTS progress
│   ├── minors.py                  ← minor requirements + advisory progress
│   ├── retriever.py               ← FAISS bi-encoder + bge-reranker-base (two-stage RAG)
│   ├── build_faiss_index.py       ← local RAG artifact builder
│   ├── schemas.py                 ← Pydantic + dataclass I/O models (PlannerRequest, TermPlan, …)
│   ├── llm_client.py              ← OpenAI wrapper (structured outputs, tool calls, retry, usage)
│   ├── prompts.py                 ← all system / user prompt templates
│   ├── agent.py                   ← legacy fixed 8-stage pipeline
│   ├── intents.py                 ← legacy pipeline intent routing
│   ├── session.py                 ← stateful conversational session (routes pipeline vs ReAct)
│   ├── react_agent.py             ← production ReAct loop + tool implementations
│   └── cli.py                     ← interactive REPL + raw trace demo
│
├── api/
│   ├── __init__.py
│   ├── main.py                    ← FastAPI backend + in-memory sessions
│   └── static/
│       ├── index.html             ← SPA shell
│       ├── style.css              ← dark-theme CSS
│       └── app.js                 ← upload, chat, trace, schedule builder
│
├── scraper/
│   ├── discover.py                ← Stage 1: enumerate all (subj, num) pairs
│   ├── scrape_full.py             ← Stage 2: fetch each course-detail page
│   ├── prereq_parser.py           ← prereq text → boolean expression tree
│   ├── build_graph.py             ← Stage 3: unified prereq DAG
│   ├── build_degree_graphs.py     ← Stage 3b: per-(program, cohort, section) DAGs
│   ├── audit_prereqs.py           ← parser coverage diagnostic
│   ├── scrape_offerings.py        ← per-term CRN / meeting scraper
│   ├── build_minors.py            ← minor requirements scraper/parser
│   └── build_eng_sci.py           ← catalog PDF → Engineering/Science ECTS JSON
│
├── data/
│   ├── SU_full_catalog.json       ← 817 catalog rows, structured
│   ├── SU_full_graph.gpickle      ← diagnostic global prereq DAG
│   ├── degree_graphs/             ← per-(program, cohort, section) scoped DAGs
│   │   ├── BSCS-DM/
│   │   │   ├── 202201/
│   │   │   │   ├── Required.{json,gpickle}
│   │   │   │   ├── Core_Elective.{json,gpickle}
│   │   │   │   ├── Area_Elective.{json,gpickle}
│   │   │   │   └── Free_Elective.{json,gpickle}
│   │   │   └── … 202301, 202401, 202501, 202601
│   │   └── … 14 programs
│   ├── offerings_*.json           ← per-term section listings (202401 → 202503)
│   ├── offerings.json             ← merged offerings map
│   ├── minors.json                ← undergraduate minor requirements
│   ├── eng_sci_credits.json       ← per-course Engineering/Science ECTS split
│   └── transcript_cagan.json      ← sample parsed transcript
│
├── eval/
│   ├── PLAN.md                    ← evaluation design and future ablations
│   ├── README.md                  ← runnable benchmark commands
│   ├── run_offline.py             ← zero-cost deterministic checks
│   ├── run_retrieval.py           ← lexical / FAISS / reranker metrics
│   ├── run_agent.py               ← API-consuming ReAct trace recorder
│   └── datasets/                  ← retrieval gold labels + chat prompts
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
git checkout timetable

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

### (Re)build the FAISS index

The aligned metadata, embedding matrix, ID map, and FAISS index are committed
for deterministic setup. A fresh checkout can use RAG immediately. Rebuild all
four artifacts together whenever `data/SU_full_catalog.json` changes:

```bash
python -m scheduler.build_faiss_index
```

The default builder uses the local Hugging Face model cache only. On a fresh
machine, allow the first model download explicitly:

```bash
python -m scheduler.build_faiss_index --allow-download
```

The optional `notebooks/build_faiss_index.ipynb` provides the same workflow in
Google Colab. A successful build produces:

```
embeddings/
├── su_courses.parquet
├── id_map.json
├── su_courses_embeddings.npy
└── su_courses.index
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

# Stage 3a: unified prereq DAG
python -m scraper.build_graph --full

# Stage 3b: per-(degree, entrance-cohort, section) scoped DAGs
python -m scraper.build_degree_graphs

# Offerings, minors, and science/engineering credit tables
python -m scraper.scrape_offerings
python -m scraper.build_minors
python -m scraper.build_eng_sci

# Audit parser coverage
python -m scraper.audit_prereqs
```

Every step is **idempotent and resumable** — HTML pages are cached; Ctrl-C mid-run is safe.

---

## 6. FAISS index (embeddings)

Built by `scheduler/build_faiss_index.py` (or the optional Colab notebook) using
**BAAI/bge-base-en-v1.5** (768 dimensions).

Embedding text per course:
```
<CODE> — <Title>
<description>
```

Index type: `IndexFlatIP` (exact cosine search via L2-normalised inner product;
the current catalog contains 817 vectors).

At query time `scheduler/retriever.py` runs two stages:
1. **Bi-encoder** — encode query with BGE query prefix, score against all eligible courses in the FAISS index.
2. **Cross-encoder** — `BAAI/bge-reranker-base` re-ranks the top-k bi-encoder hits with full pair attention.

`Retriever.load()` verifies that the parquet metadata, vector matrix, ID map,
and FAISS index describe the same catalog snapshot. A stale index fails with an
actionable rebuild message instead of silently returning unrelated courses.

> **macOS Apple Silicon note:** `SentenceTransformer` must be imported **before** `faiss` in the same process. Both link OpenBLAS; double-init causes a segfault (exit code 139). `retriever.py` handles this; do not reorder the imports.

---

## 7. Python module guide

This is the shortest useful map for reading the code before a demo.

### 7a. Scheduler runtime

| File | Duty |
|---|---|
| `scheduler/session.py` | Central state container. Loads academic input, keeps structured profile data outside chat history, retains at most 25 chat turns, lazily loads graphs/catalog/offerings/retriever, and routes each turn to ReAct or the legacy pipeline. |
| `scheduler/react_agent.py` | Production conversational agent. Defines the GPT-4o system prompt, tool schemas, tool implementations, RAG prefetch, permission-gated planning, trace events, title-mismatch repairs, profile re-ask repairs, and the maximum 8-step ReAct loop. |
| `scheduler/degree_eval.py` | Preferred academic parser. Reads the richer Degree Evaluation HTML file: program, entrance term, completed/in-progress courses, per-section SU-credit totals, Engineering ECTS, Basic-Science ECTS, and minors. |
| `scheduler/transcript.py` | Fallback PDF parser for Sabancı Academic Records. Produces the shared student-profile shape when a Degree Evaluation HTML file is unavailable. |
| `scheduler/graph_selector.py` | Normalizes aliases such as `CS → BSCS-DM`, maps spring admits to the matching fall entrance cohort, loads only the relevant requirement graphs, and preserves `in_slice` vs upstream-only nodes. |
| `scheduler/requirements.py` | Computes graph-derived remaining pools with precedence `Required > Core Elective > Area Elective > Free Elective`, avoiding duplicate counting when one course appears in several pools. |
| `scheduler/eligibility.py` | Symbolic hard-constraint engine. Recursively evaluates prerequisite trees, filters takeable courses, detects duplicates/completed courses, expands direct corequisites, and checks min/max SU credit load. |
| `scheduler/catalog.py` | Lightweight exact course lookup, code extraction, corequisite extraction, and lexical fallback retrieval when FAISS is unavailable. |
| `scheduler/offerings.py` | Loads historical offerings and computes same-season likely-open courses for future-term advising. |
| `scheduler/timetable.py` | Deterministic weekly schedule engine. Loads section times, detects overlaps, selects a same-season proxy term for unpublished future schedules, and backtracks over sections to find a conflict-free assignment. |
| `scheduler/retriever.py` | Two-stage semantic RAG: filtered BGE bi-encoder retrieval followed by optional BGE cross-encoder reranking. Also validates RAG artifact alignment. |
| `scheduler/build_faiss_index.py` | Rebuilds parquet metadata, BGE passage embeddings, FAISS index, and ID map from the current catalog. Uses the local Hugging Face cache by default. |
| `scheduler/eng_sci.py` | Tracks Engineering and Basic-Science ECTS minima and candidate-course contributions. Uses Degree Evaluation totals as authoritative when available. |
| `scheduler/minors.py` | Advisory minor lookup. Resolves minor names/codes and computes required/elective progress without pretending every option in an elective pool is mandatory. |
| `scheduler/schemas.py` | Shared Pydantic structured-output models and Python dataclasses (`PlannerRequest`, `TermPlan`). |
| `scheduler/llm_client.py` | OpenAI client singleton, `.env` loading, retry logic, structured outputs, function calling, and token accounting. |
| `scheduler/prompts.py` | Prompt templates for the legacy pipeline and term-label helpers. |
| `scheduler/agent.py` | Legacy fixed 8-stage planning pipeline retained as a fallback and ablation baseline. |
| `scheduler/intents.py` | Legacy pipeline classifier plus deterministic swap/drop/add/explain dispatch handlers. |
| `scheduler/cli.py` | Terminal REPL. Supports natural chat, plan commands, token usage, compact `/trace`, and full `/raw-trace` for demos. |
| `scheduler/__init__.py` | Package marker. |

### 7b. Scraper and artifact builders

| File | Duty |
|---|---|
| `scraper/discover.py` | Enumerates course pages from degree pools and optional subject-range brute force; writes the discovery manifest. |
| `scraper/scrape_cs.py` | Low-level cached SUIS/Banner HTML fetching and degree-section label handling. |
| `scraper/scrape_full.py` | Builds the structured full catalog from discovered course pages. |
| `scraper/prereq_parser.py` | Recursive-descent parser for nested prerequisite `AND` / `OR` expressions, plus a lenient fallback for unusual free text. |
| `scraper/build_graph.py` | Builds the diagnostic global NetworkX prerequisite DAG. |
| `scraper/build_degree_graphs.py` | Produces the authoritative per-program, per-entrance-cohort, per-section graph slices used during advising. |
| `scraper/scrape_offerings.py` | Scrapes section/CRN/meeting-time rows for historical terms and writes the merged offerings map. |
| `scraper/audit_prereqs.py` | Reports strict, lenient, and failed prerequisite parses so parser coverage can be verified. |
| `scraper/build_minors.py` | Scrapes and structures undergraduate minor requirements. |
| `scraper/build_eng_sci.py` | Extracts per-course Engineering and Basic-Science ECTS splits from the catalog PDF. |

### 7c. Web and evaluation

| File | Duty |
|---|---|
| `api/main.py` | FastAPI entry point. Owns in-memory sessions, academic upload, chat turns, course-section lookup, schedule-builder persistence, schedule review, state, and cleanup endpoints. |
| `api/__init__.py` | API package marker. |
| `eval/run_offline.py` | Zero-cost deterministic regression suite. |
| `eval/run_retrieval.py` | Retrieval ablations: lexical, FAISS-only, and FAISS+reranker; reports Recall, MRR, and nDCG. |
| `eval/run_agent.py` | API-consuming benchmark runner that saves answers, raw ReAct traces, tool calls, latency, token usage, and hallucinated-code checks. |
| `eval/score_agent_runs.py` | Aggregates saved online runs into a compact comparison table. |
| `eval/variants.py` | Named online agent variants for ReAct, pipeline, and required-course-injection comparison. |

---

## 8. Agent modes

`PlannerRequest.mode` selects the agent style. Default is `"react"`.

### `mode="pipeline"` (legacy baseline)

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

This is the production default. The model receives the same tool suite at every
turn and decides what to call and in what order. It can:
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
    --transcript "DEGREE EVALUATION.html" \
    --term 202601

# Live debug trace for demos: every tool call, arguments, result, repair, final answer
python -m scheduler.cli \
    --transcript "DEGREE EVALUATION.html" \
    --term 202601 \
    --debug-trace

# With an automatic first message (useful for demos)
python -m scheduler.cli \
    --transcript "DEGREE EVALUATION.html" \
    --debug-trace \
    --first-message "Plan my fall semester. I want ML and a databases course."

# Full untruncated raw JSON trace, then exit after the first message
python -m scheduler.cli \
    --transcript "DEGREE EVALUATION.html" \
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
| `POST` | `/session/{id}/transcript` | Upload Degree Evaluation HTML (preferred), transcript PDF, or JSON |
| `POST` | `/session/{id}/turn` | Send a message, get a response + plan |
| `GET` | `/session/{id}/course_sections?code=CS%20412` | List real/proxy sections and direct labs/recitations |
| `POST` | `/session/{id}/schedule` | Persist schedule-builder selections without an LLM call |
| `POST` | `/session/{id}/check_schedule` | Ask ReAct to review the persisted schedule |
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

- **Drag-and-drop academic upload** — Degree Evaluation HTML, PDF, or JSON
- **Chat interface** with markdown rendering (marked.js)
- **Degree/profile summary** after upload
- **Schedule builder** with section choices and auto-added recitations/labs
- **Weekly timetable view** and conflict markers
- **Agent trace panel** for compact tool-call inspection
- **Token usage stats**

---

## 11. Data formats

### Parsed academic profile

`scheduler.degree_eval.parse_degree_evaluation()` and
`scheduler.transcript.parse_transcript()` both produce a shared profile shape.
Degree Evaluation HTML adds the authoritative `section_requirements` block:

```jsonc
{
  "name": "Student Name",
  "program": "BSCS-DM",
  "admit_term": "202201",
  "current_semester": 7,
  "cgpa": 3.40,
  "cumulative_credits": 127.0,
  "completed_for_eligibility": ["CS 201", "CS 201R", "CS 204", …],
  "in_progress": ["CS 412"],
  "minors": [],
  "section_requirements": {
    "REQUIRED COURSES": { "min_su": 29.0, "completed_su": 26.0 },
    "ENGINEERING": { "min_ects": 90.0, "completed_ects": 100.0 },
    "BASIC SCIENCE": { "min_ects": 60.0, "completed_ects": 60.0 }
  }
}
```

### `data/offerings.json`

```jsonc
{
  "202501": {
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
            "where": "FENS L067"
          }
        ]
      }
    ]
  }
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

## 12. Evaluation & regression checks

The evaluation is split so most regressions cost no API tokens. See
[`eval/README.md`](eval/README.md) for the full command list and
[`eval/PLAN.md`](eval/PLAN.md) for the extended ablation design.

### 12a. Deterministic suite

```bash
source .venv/bin/activate
python -m eval.run_offline
```

This checks cohort mapping, graph scoping, Degree Evaluation HTML loading,
requirement de-duplication, prerequisite `AND` / `OR`, offering patterns,
timetable overlap detection, lab/recitation discovery, and scoped fallback
retrieval. It also verifies local RAG artifact alignment when the generated
vector files are present. Current result:

```text
Offline deterministic suite: 11/11 passed
```

### 12b. Retrieval ablations

```bash
python -m eval.run_retrieval --backends lexical faiss faiss_rerank
python -m eval.run_retrieval --backends lexical faiss faiss_rerank --use-rewrites
```

Metrics: `Recall@5`, `Recall@10`, `MRR`, and `nDCG@10`.

Observed locally after rebuilding the aligned 817-course artifacts:

| Retrieval variant | Rewrite? | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| Lexical fallback | No | 0.5833 | 0.6343 | 0.5356 |
| FAISS bi-encoder | No | 0.7750 | 0.5750 | 0.6017 |
| FAISS + reranker | No | 0.6667 | 0.7500 | 0.6358 |
| FAISS + reranker | Yes | 0.9167 | 0.9500 | 0.7454 |

The reranker optimizes ranking quality rather than raw coverage: it improves
`MRR` and `nDCG`, while query rewriting recovers high recall for vague prompts
such as `LLM area` and `operational areas`.

### 12c. Online ReAct evaluation

This path consumes API tokens and saves the full background trace:

```bash
# Start small
python -m eval.run_agent --variant react_full --limit 2

# Then compare the implemented online variants
python -m eval.run_agent --variant react_full
python -m eval.run_agent --variant pipeline
python -m eval.run_agent --variant react_no_required_injection
```

Each saved run records final answers, tool names and parameters, raw tool
results, latency, token usage, ReAct step count, extracted course codes, and
hallucinated-code checks under `eval/runs/`.

### 12d. Additional smoke checks

| Check | How |
|---|---|
| Prerequisite parser audit | `python -m scraper.audit_prereqs` |
| Full diagnostic graph is a DAG | Read `data/SU_full_graph_report.txt` |
| Eligibility smoke test | `python -m scheduler.eligibility` |
| Offerings smoke test | `python -m scheduler.offerings` |
| Retriever smoke test | `python -m scheduler.retriever` |
| Python syntax | `python -m py_compile scheduler/*.py scraper/*.py api/main.py eval/*.py` |

---

## 13. Credits

- **Course:** CS 455 / CS 555 — Large Language Models, Spring 2025/2026 (Sabancı University)
- **Instructor:** Inanç Arın
- **Models used:** BAAI/bge-base-en-v1.5 (bi-encoder), BAAI/bge-reranker-base (cross-encoder), OpenAI GPT-4o (planner, intent classification, query rewriting, and ReAct)
- **License:** MIT — see `LICENSE`
