# SuSchedule-r

A **course planning assistant for Sabancı University** — built as the term
project for **CS 455 / CS 555: Large Language Models (Spring 2025/2026)**.

The system ingests a student's PDF transcript, scrapes the full SU
undergraduate catalog into a structured prerequisite graph, and produces a
feasible, conflict-free term plan that combines a deterministic symbolic
core (eligibility + scheduling) with an LLM-driven recommendation layer
(RAG over course descriptions).

This branch (`kero`) contains the **catalog scraping pipeline** — the data
foundation everything else builds on.

---

## Table of Contents

1. [What this repo does](#what-this-repo-does)
2. [Quick start](#quick-start)
3. [Pipeline overview](#pipeline-overview)
4. [Repository layout](#repository-layout)
5. [The scraper modules in detail](#the-scraper-modules-in-detail)
6. [The scheduler modules](#the-scheduler-modules)
7. [Data formats](#data-formats)
8. [How to run the full pipeline](#how-to-run-the-full-pipeline)
9. [Verification & regression checks](#verification--regression-checks)
10. [Roadmap](#roadmap)

---

## What this repo does

| Concern                                  | Module                                  | Output                                              |
| ---------------------------------------- | --------------------------------------- | --------------------------------------------------- |
| Discover every course at SU              | `scraper/discover.py`                   | `data/discovery_manifest.json`                      |
| Fetch each course's detail page          | `scraper/scrape_full.py`                | `data/SU_full_catalog.json`                         |
| Build a unified prereq DAG               | `scraper/build_graph.py --full`         | `data/SU_full_graph.{json,gpickle,_report.txt}`     |
| Build per-degree, per-section DAGs       | `scraper/build_degree_graphs.py`        | `data/degree_graphs/<PROGRAM>/<SECTION>.*`          |
| Audit prereq-parser coverage             | `scraper/audit_prereqs.py`              | console report + `data/unparsed_prereqs.txt`        |
| Scrape per-term section offerings (CRNs) | `scraper/scrape_offerings.py`           | `data/offerings_<TERM>.json`                        |
| Parse a transcript PDF                   | `scheduler/transcript.py`               | profile JSON                                        |
| Eligibility check                        | `scheduler/eligibility.py`              | list of allowed courses + violations                |
| Time-conflict / timetable feasibility    | `scheduler/timetable.py`                | feasible CRN combos                                 |
| Offering-window filter                   | `scheduler/offerings.py`                | season-aware course filter                          |

---

## Quick start

This project targets **Python 3.10+**. The system Python on recent macOS is
externally-managed (PEP 668), so use a virtualenv.

```bash
# 1. clone & enter the repo
git clone git@github.com:keremsirtikizil/SuSchedule-r.git
cd SuSchedule-r
git checkout kero

# 2. create and activate a virtualenv
python3 -m venv .venv
source .venv/bin/activate           # on macOS / Linux
# .venv\Scripts\activate            # on Windows

# 3. install dependencies
pip install -r requirements.txt
```

`requirements.txt` pins four libraries:

| Package          | Purpose                                                         |
| ---------------- | --------------------------------------------------------------- |
| `requests`       | HTTP client for hitting Sabancı's Banner endpoints              |
| `beautifulsoup4` | HTML parsing for degree / pool / course pages                   |
| `networkx`       | Directed-graph data structure and DAG utilities for prereqs     |
| `pdfplumber`     | Text extraction from "Academic Records Summary" transcript PDFs |

> **Activating the venv:** every new shell needs
> `source .venv/bin/activate`. When active, your prompt is prefixed with
> `(.venv)`.

---

## Pipeline overview

```
┌─────────────────┐
│  Sabancı Banner │  (suis.sabanciuniv.edu/prod/...)
└────────┬────────┘
         │ HTTP GET (polite 0.4s delay, cached on disk)
         ▼
┌────────────────────────────────────────────────────────────┐
│  STAGE 1 — discover.py                                     │
│                                                            │
│  1a. For each of 14 UG degree codes × 3 admit terms,       │
│      fetch the degree-requirements page + every elective   │
│      pool it links to. Record every (subj, num) found,     │
│      tagged with (program, term, section).                 │
│                                                            │
│  1b. (optional) Subject brute-force: for every subject     │
│      code seen, enumerate crse_numb in 100..499 to catch   │
│      orphan courses outside any degree pool.               │
│                                                            │
│  Output: data/discovery_manifest.json                      │
└────────┬───────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  STAGE 2 — scrape_full.py                                  │
│                                                            │
│  For every (subj, num) in the manifest, fetch the          │
│  English course-detail page (cached in courses/*.html)     │
│  and parse it into a structured record:                    │
│    code, subj, num, title, su_credit, ects_text,           │
│    description, prereq_text, coreq_text,                   │
│    offered_terms[], program_sections[]                     │
│                                                            │
│  Output: data/SU_full_catalog.json                         │
└────────┬───────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  STAGE 3 — build_graph.py --full                           │
│                                                            │
│  Parse every prereq_text into a boolean expression tree    │
│  (and/or/parens). Strict parser first; lenient fallback    │
│  for non-CS grammar variants. Add a directed edge          │
│  u -> v for every "v lists u as a prereq", tagged with     │
│  role ∈ {required, alternative}.                           │
│                                                            │
│  Output: data/SU_full_graph.{json,gpickle,_report.txt}     │
└────────┬───────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  STAGE 3b — build_degree_graphs.py                         │
│                                                            │
│  Slice the unified catalog into per-(degree, section)      │
│  sub-DAGs. For each program × {Required, Core, Area, Free} │
│  emit a graph containing the in-section courses plus all   │
│  upstream prereq nodes they reference.                     │
│                                                            │
│  Output: data/degree_graphs/<PROGRAM>/<SECTION>.*          │
└────────────────────────────────────────────────────────────┘
```

Every stage is **idempotent and resumable**. HTML responses are cached on
disk, so re-runs only hit the network for missing pages.

---

## Repository layout

```
SuSchedule-r/
├── README.md                  ← you are here
├── LICENSE                    ← MIT
├── requirements.txt           ← Python dependencies
├── .gitignore
│
├── scraper/                   ← Stage 1–3 catalog pipeline
│   ├── scrape_cs.py           ← original CS-only scraper (parameterised)
│   ├── discover.py            ← NEW — Stage 1: course-set discovery
│   ├── scrape_full.py         ← NEW — Stage 2: full catalog fetch
│   ├── prereq_parser.py       ← prereq_text → boolean expression tree
│   ├── build_graph.py         ← Stage 3: unified prereq DAG
│   ├── build_degree_graphs.py ← NEW — Stage 3b: per-degree, per-section
│   ├── audit_prereqs.py       ← NEW — parser-coverage diagnostic
│   └── scrape_offerings.py    ← per-term section / CRN scraper
│
├── scheduler/                 ← downstream symbolic core
│   ├── transcript.py          ← PDF transcript → Student profile JSON
│   ├── eligibility.py         ← prereq / credit-load / coreq checking
│   ├── offerings.py           ← "is course X offered in Fall?" filter
│   └── timetable.py           ← time-conflict detection, feasible CRN combo
│
├── courses/                   ← cached <SUBJ>_<NUM>.html course pages
├── degrees/                   ← cached degree-requirements HTML
├── pools/                     ← cached elective-pool HTML
├── offerings/                 ← cached per-term section listings (HTML)
│
└── data/
    ├── BSCS-DM_202601_catalog.json   ← original CS-only baseline
    ├── BSCS-DM_202601_graph.{json,gpickle}
    ├── BSCS-DM_202601_graph_report.txt
    │
    ├── discovery_manifest.json       ← Stage 1 output
    ├── SU_full_catalog.json          ← Stage 2 output (688 courses)
    ├── SU_full_graph.{json,gpickle}  ← Stage 3 output
    ├── SU_full_graph_report.txt
    │
    ├── degree_graphs/                ← Stage 3b output
    │   ├── BSCS-DM/
    │   │   ├── Required.{json,gpickle}
    │   │   ├── Required_report.txt
    │   │   ├── Core_Elective.{json,gpickle}
    │   │   ├── Area_Elective.{json,gpickle}
    │   │   └── Free_Elective.{json,gpickle}
    │   ├── BSEE-DM/   …
    │   └── (14 programs × 4 sections)
    │
    ├── offerings_<TERM>.json         ← per-term CRN / meeting data
    └── transcript_cagan.json         ← parsed sample transcript
```

---

## The scraper modules in detail

### `scraper/scrape_cs.py` — the foundational scraper

The original single-program scraper, **refactored to be reusable**.
Hardcoded `PROGRAM = "BSCS-DM"` and `DEFAULT_TERM = "202601"` constants
remain for back-compatibility, but the URL builders and HTML parsers
(`fetch_degree_page`, `parse_degree_page`, `fetch_pool_page`,
`parse_pool_page`, `fetch_course_page`, `parse_course_page`) now accept
program / term arguments and are imported by the new modules.

Snapshot paths were corrected from the old `snapshots/{degrees,pools,courses}`
layout to the flat `degrees/`, `pools/`, `courses/` directories actually
present in the repo. A new helper `section_labels_for(program)` generates
the four-variant anchor map needed because SU's degree pages use
**four different anchor naming conventions** for Required / Core / Area / Free
sections (e.g. `BSCS-DM_R`, `BAECONDM_R`, `BAPOLSDMC1`, even a typo
`BAVACDD_C1`).

**Run standalone (CS-only, baseline):**

```bash
python -m scraper.scrape_cs 202601
```

---

### `scraper/discover.py` — Stage 1 course discovery

Builds the master set of `(subj, num)` pairs to fetch in Stage 2. Two
complementary passes:

| Pass | What it does                                                                                                                                | Network cost                     |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| 1a   | For each of **14 UG degree codes** × **3 admit terms** (`202501`, `202502`, `202601`), fetch the degree page and every elective pool it links to. Track every `(subj, num)` plus the `(program, term, section)` it appeared under. | ~50 requests, ~30 s |
| 1b   | (optional) Brute-force `crse_numb` 100–499 for every subject code seen in 1a's HTML. Early-stops a subject after 20 consecutive misses; skips entirely after 50 zero-hit probes.                                                    | up to ~21 600 requests, ~1.5 h |

The 14 degree codes are seeded from
`scheduler/transcript.py::PROGRAM_NAME_TO_CODE`:

> BSCS-DM, BSDSA-DM, BAECON-DM, BSEE-DM, BSIE-DM, BAIS-DM, BAMAN-DM,
> BSMAT-DM, BSME-DM, BSBIO-DM, BAPOLS-DM, BAPSIR-DM, BAPSY-DM, BAVACD-DM

The three terms — Fall 25-26, Spring 25-26, Fall 26-27 — were chosen so the
same scrape covers the typical "next 1–2 semesters" window a student plans
around.

**Run:**

```bash
python -m scraper.discover --degrees           # Stage 1a only (recommended first)
python -m scraper.discover --degrees --subjects  # both passes
python -m scraper.discover --all               # shortcut for both
```

---

### `scraper/scrape_full.py` — Stage 2 detail-page fetch

Consumes `data/discovery_manifest.json` and fetches one course-detail
page per unique `(subj, num)`. Course pages are **term-agnostic and
subject-agnostic** — the same `(subj, num)` returns the same content
regardless of which degree page led you there, so each pair is fetched
once.

Reuses `parse_course_page()` from `scrape_cs.py`. Each course record
carries the full `program_sections[]` list aggregating every
`(program, term, section)` it appeared under during Stage 1 — this is what
makes `build_degree_graphs.py` possible without re-scraping.

**Run:**

```bash
python -m scraper.scrape_full
# → data/SU_full_catalog.json
```

---

### `scraper/prereq_parser.py` — prereq text → expression tree

Turns a raw `prereq_text` such as

```
(MATH 201 - Undergraduate - Min Grade D or MATH 212 - Undergraduate - Min Grade D)
  and MATH 203 - Undergraduate - Min Grade D
```

into a structured boolean tree:

```jsonc
{
  "op": "and",
  "operands": [
    { "op": "or",
      "operands": [
        { "course": "MATH 201", "min_grade": "D", "concurrent": false },
        { "course": "MATH 212", "min_grade": "D", "concurrent": false }
      ]
    },
    { "course": "MATH 203", "min_grade": "D", "concurrent": false }
  ]
}
```

Two public entry points:

| Function                  | Behavior                                                                                                                                  |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `parse_prereq(text)`      | **Strict**: raises `ValueError` on anything outside the observed CS grammar. Used as a regression baseline.                               |
| `parse_prereq_lenient(text)` | **Forgiving**: tries strict first, then falls back to a broader parser. Never raises. Wraps unrecognised spans as `{"raw": ..., "unparsed": true}` leaves so downstream code can flag them. |

Helpers:

- `flatten_courses(expr)` — list every course code in the tree.
- `expr_satisfied(expr, completed_set)` — evaluate against a student's
  completed courses. Treats `note` leaves as satisfied and `unparsed`
  leaves as not-satisfied (fail-safe → forces manual review).
- `has_unparsed(expr)` — true if any leaf is `unparsed`.

---

### `scraper/build_graph.py` — Stage 3 unified DAG

Reads a catalog JSON and emits a NetworkX `DiGraph`. Two modes:

| Mode                  | Invocation                              | Reads                                | Writes                                                      |
| --------------------- | --------------------------------------- | ------------------------------------ | ----------------------------------------------------------- |
| Single-program        | `python -m scraper.build_graph 202601`  | `BSCS-DM_202601_catalog.json`        | `BSCS-DM_202601_graph.{json,gpickle,_report.txt}`           |
| Unified (full SU)     | `python -m scraper.build_graph --full`  | `SU_full_catalog.json`               | `SU_full_graph.{json,gpickle,_report.txt}` (lenient parser) |

**Edge semantics** — for every prereq tree:

- `u → v` with `role="required"` means *every* satisfying assignment of `v`'s
  tree contains `u` (i.e. `u` only appears under ANDs from the root).
- `u → v` with `role="alternative"` means `u` appears under at least one OR
  branch (one of several substitutes).

The report file at the end summarises node/edge counts, source/sink lists,
and runs `nx.is_directed_acyclic_graph()` as a sanity check.

---

### `scraper/build_degree_graphs.py` — Stage 3b per-(degree, section)

Slices `SU_full_catalog.json` into one DAG per `(program, section)` pair,
where section is one of `Required`, `Core Elective`, `Area Elective`,
`Free Elective`.

Output layout:

```
data/degree_graphs/
├── BSCS-DM/
│   ├── Required.{json,gpickle,_report.txt}
│   ├── Core_Elective.{json,gpickle,_report.txt}
│   ├── Area_Elective.{json,gpickle,_report.txt}
│   └── Free_Elective.{json,gpickle,_report.txt}
├── BSEE-DM/  …
└── (14 programs × 4 sections)
```

Each slice graph keeps **upstream prereq nodes** as `in_slice=False` so
eligibility-check traversal (which walks backwards from a target course)
stays self-contained. Each in-slice node also carries
`terms_in_section: ["202501", …]` so a planner can answer "is CS 305 a
Required course for BSCS-DM in Fall 26-27?"

**Run:**

```bash
python -m scraper.build_degree_graphs
# default = all 14 programs × 4 sections
python -m scraper.build_degree_graphs --sections Required "Core Elective"
```

---

### `scraper/audit_prereqs.py` — parser coverage diagnostic

Runs the parser over a catalog and buckets every course's `prereq_text`:

| Bucket    | Meaning                                                                              |
| --------- | ------------------------------------------------------------------------------------ |
| `empty`   | no prereq                                                                            |
| `strict`  | parsed cleanly by `parse_prereq()`                                                   |
| `lenient` | strict failed, but lenient parser produced a clean tree (no `unparsed` leaves)       |
| `partial` | lenient parser produced a tree containing some `unparsed` leaves                     |
| `failed`  | only `unparsed` leaves — parser couldn't extract any structure                       |

Also writes the residual unparsed spans to `data/unparsed_prereqs.txt` so
new patterns can be folded back into the strict parser, **and** runs a
regression check confirming every CS prereq still strict-parses exactly as
it does today.

```bash
python -m scraper.audit_prereqs
```

Current state on `data/SU_full_catalog.json`:

```
Audit of SU_full_catalog.json: 688 courses
     empty: 262
    strict: 426
   lenient: 0
   partial: 0
    failed: 0
regression OK (BSCS-DM_202601_catalog.json): all prereq_text strict-parses
```

100% coverage — every prereq in the dataset parses strictly.

---

### `scraper/scrape_offerings.py` — per-term section scraper

Separate pipeline that hits Banner's schedule-search endpoint (POST) and
extracts every section/CRN/meeting-pattern row for a given term:

```jsonc
{
  "CS 201": [
    { "crn": "10190", "section": "A",
      "meetings": [{ "type": "lecture", "days": [0, 2], "start": 600, "end": 690,
                     "where": "FENS L067", "instructor": "Erkay Savaş" }],
      "instructors": ["Erkay Savaş"], "locations": ["FENS L067"] }
  ]
}
```

Output files are already in the repo for terms 202401 → 202503. The
downstream `scheduler/timetable.py` module consumes these.

---

## The scheduler modules

These are **not** part of the scraping pipeline but they're the downstream
consumers — useful to understand why the catalog is shaped the way it is.

| Module                        | Role                                                                                                                                                                                                                                                                                       |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `scheduler/transcript.py`     | Parses Sabancı's "Academic Records Summary" PDF into a structured profile (`completed`, `in_progress`, `terms[]`, `cgpa`, `admit_term`, `program`, `minors`).                                                                                                                              |
| `scheduler/eligibility.py`    | Pure symbolic eligibility. Given a graph + student profile, returns the set of courses they can take next term. Owns the rules we refuse to delegate to an LLM: prereq satisfaction, no-retake-completed, credit-load cap, coreq pairing. Also has `validate_plan()` for post-LLM checking.|
| `scheduler/offerings.py`      | "Is course X likely to run in season S?" Backed by `offerings_<TERM>.json`. The proposal's rule: keep a course if it ran in its usual season at least once in the last four terms.                                                                                                          |
| `scheduler/timetable.py`      | Given a set of course codes + a target term, finds a CRN-section combination whose meetings don't overlap. Returns a feasible weekly timetable or reports the conflicting pair.                                                                                                            |

---

## Data formats

### `data/discovery_manifest.json`

```jsonc
{
  "discovered_at": "2026-05-25T09:24:11Z",
  "degrees": ["BSCS-DM", "BSEE-DM", ...],
  "terms":   ["202501", "202502", "202601"],
  "subjects_discovered": ["ACC", "ANTH", "BIO", ...],   // 54 subjects
  "missing_degrees": [],                                // any 404s land here
  "course_count": 688,
  "courses": {
    "CS 201": {
      "subj": "CS", "num": "201",
      "program_sections": [
        {"program": "BSCS-DM", "term": "202601", "section": "Required"},
        {"program": "BSEE-DM", "term": "202601", "section": "Engineering"}
      ]
    }
  }
}
```

### `data/SU_full_catalog.json`

```jsonc
{
  "source": "SU_full",
  "degrees": [...],
  "terms":   [...],
  "course_count": 688,
  "error_count": 0,
  "errors": [],
  "courses": [
    {
      "code": "CS 201", "subj": "CS", "num": "201",
      "title": "Programming Fundamentals",
      "su_credit": 3.0, "ects_text": "ECTS Credit: 6.00",
      "description": "Introduction to computer programming using ...",
      "prereq_text": "", "coreq_text": "CS 201R - Undergraduate - Min Grade D",
      "offered_terms": [
        {"term": "202401", "name": "Programming Fundamentals", "su_credit": "3.00"}
      ],
      "program_sections": [...]
    }
  ]
}
```

### `data/SU_full_graph.json` (and per-section graphs)

Node-link JSON. Every node carries the full course metadata
(`title`, `description`, `prereq_expr`, etc.) plus an `in_catalog` /
`in_slice` flag distinguishing real catalog courses from dangling external
prereqs (e.g. an old course code that was renamed). Every edge carries
`role`, `min_grade`, `concurrent`.

For programmatic use, prefer the pickle:

```python
import pickle, networkx as nx
G: nx.DiGraph = pickle.load(open("data/SU_full_graph.gpickle", "rb"))
print(G.number_of_nodes(), G.number_of_edges())  # 721 729
print(G.nodes["CS 305"]["prereq_text"])
list(G.predecessors("CS 305"))                   # direct prereqs
nx.ancestors(G, "CS 305")                        # everything you eventually need
```

---

## How to run the full pipeline

After the venv is set up:

```bash
source .venv/bin/activate

# Stage 1: discover the full set of (subj, num) pairs (~30 s)
python -m scraper.discover --degrees

# Optional Stage 1b: subject brute-force to catch orphan courses (~1.5 h)
python -m scraper.discover --subjects

# Stage 2: fetch each course's detail page once, cached in courses/
python -m scraper.scrape_full

# Stage 3a: unified prereq DAG
python -m scraper.build_graph --full

# Stage 3b: per-degree, per-section DAGs
python -m scraper.build_degree_graphs

# Audit how well the parser covered the data
python -m scraper.audit_prereqs
```

Every step caches its raw HTML inputs to disk — a Ctrl-C mid-run is
recoverable, and re-runs are nearly instant when the cache is warm.

The total cold-cache wall-clock for Stages 1a + 2 + 3 + 3b is about
**4 minutes**. Adding Stage 1b pushes it to roughly **1.5 hours** of mostly
idle network time.

---

## Verification & regression checks

| What                                            | How                                                                                       |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------- |
| All CS courses preserved after the big scrape   | `audit_prereqs.py` regression block — diffs the new catalog against `BSCS-DM_202601_catalog.json` |
| Prereq parser still produces identical CS trees | `audit_prereqs.py` re-parses every CS prereq and reports any mismatch                     |
| The full SU prereq graph is acyclic             | `SU_full_graph_report.txt` line `Is DAG?  True`                                           |
| Each per-degree slice is acyclic                | `data/degree_graphs/<PROGRAM>/<SECTION>_report.txt`                                       |
| Discovery didn't drop any degree pages          | `discovery_manifest.json` → `"missing_degrees": []`                                       |

---

## Roadmap

This branch covers the **data layer**. The downstream work, in rough order:

1. **FAISS index over `description` fields** — for the RAG "which course
   matches my interest in X?" retriever.
2. **End-to-end planner agent** — combine eligibility + offerings +
   timetable + retriever into a single LLM call that proposes a term plan.
3. **Hard-constraint validator** — re-run `scheduler.eligibility.validate_plan`
   on the LLM's output and surface every violation.
4. **Web UI** — upload a transcript PDF, get a planned semester.

---

## Credits

- **Course:** CS 455 / CS 555 — Large Language Models, Spring 2025/2026
  (Sabancı University, Faculty of Engineering and Natural Sciences).
- **Instructor:** Inanc Arin.
- **License:** MIT — see `LICENSE`.
