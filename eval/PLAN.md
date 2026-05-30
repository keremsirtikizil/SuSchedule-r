# SuSchedule-r — Evaluation Plan

How to measure the system's quality and prove the agentic design adds value,
grounded in the modules that already exist in this repo. The evaluation is
**mostly deterministic**: a small human rubric is used only for answer quality.

> Status: the first runnable harness is implemented. `run_offline.py`,
> `run_retrieval.py`, `run_agent.py`, and `score_agent_runs.py` cover the
> zero-cost regression suite, retrieval ablations, and online trace recording.
> The larger ablation matrix below remains the roadmap for deeper experiments.

---

## 1. Core idea

Most of the target metrics are measurable for free because the project already
contains deterministic oracles (`eligibility`, `requirements`, `graph_selector`,
`timetable`, `offerings`, `catalog`) and a clean per-turn trace. The strategy:

1. **Drive** the system through one entry point: `PlannerSession.from_request(...).handle_turn(msg)`.
2. **Capture** three artifacts per prompt: the final answer text, `session.current_plan` (a `TermPlan`), and `session.last_raw_trace`.
3. **Score** by replaying those same deterministic modules as an *independent grader*, plus a blinded human rubric only for answer quality.

This is what makes the result defensible rather than "we tried prompts and it
seemed good": the validity grader is the *same logic the agent must satisfy*,
run independently on the agent's output.

---

## 2. Code levers (what the harness hooks into)

### 2.1 Entry point — `scheduler/session.py`
```python
sess = PlannerSession.from_request(PlannerRequest(mode="react", target_term="202601", ...))
sess.load_student_from_path(Path("eval/datasets/students/bscs_sem7.json"))  # .json / .html / .pdf
answer = sess.handle_turn("Suggest theoretical CS courses.")
plan   = sess.current_plan        # TermPlan | None
trace  = sess.last_raw_trace      # cleared at the start of every react turn
usage  = llm_client.get_session_usage()   # {prompt, completion, total} — process-global
```

Two facts to design around:
- **`get_session_usage()` is a module-global counter.** Run prompts sequentially,
  one fresh `PlannerSession` per prompt, and snapshot usage deltas. Fresh sessions
  also isolate the trace and academic state.
- **The ReAct step cap is a module constant** `react_agent.MAX_ITERATIONS = 8`
  (not a `PlannerRequest` field). A step-budget experiment means patching that constant.

### 2.2 Trace events (`react_agent._trace`)
Each ReAct turn emits, in order:
`turn_start` → `prefetched_recommendation_context` → repeated
`assistant_step` (`step`, `content`, `tool_calls[]`) and `tool_result`
(`step`, `name`, `arguments`, `result`) → plus `retrieval`
(`queries`, `filters`, `merged_results[]` with `rerank_score`/`eligible`/`completed`),
`selected_graphs`, `requirements`, `build_timetable` → `final_response`.

This is the raw material for every efficiency and "right tool for the query" metric.

### 2.3 Deterministic graders that already exist

| Need | Call |
|---|---|
| Plan validity (prereqs, credit load, dupes, coreqs) | `eligibility.validate_plan(graph, student, plan)` → `PlanReport(ok, violations, total_credits, auto_added_coreqs)` |
| Per-course prereq check | `eligibility.prereqs_satisfied(graph, code, completed)` |
| Remaining requirements (gold) | `requirements.compute_remaining(program, completed, in_progress, admit_term)` → `required_left/core_left/area_left/free_left` |
| Correct scoped graph | `graph_selector.select_graphs(program, admit_term)` → `.cohort_term`, `.candidate_courses(section)` |
| Cohort mapping | `graph_selector.cohort_for_admit(program, admit_term)` |
| Course exists / title | `catalog.Catalog.load().get(code)` |
| **Extract course codes from answer text** | `catalog.extract_course_codes(text)` |
| Hallucinated-title detection | `react_agent._catalog_title_mismatches(session, text)` (already built) |
| Expected recitations/labs for a course | `catalog.corequisite_codes(catalog, code)` |
| Schedule conflict-free | `timetable.build_timetable(plan, term, offerings)` → `TimetableResult(ok, picks, missing, reason)`; `timetable.conflict_report(picks)` |
| Offering likelihood (gold) | `offerings.likely_offered_in(off, term, same_season_only=True)`, `offerings.offered_history(off, code)` |
| Token usage | `llm_client.get_session_usage()` |

### 2.4 Retriever knobs for RAG ablations (`scheduler/retriever.py`)
- `Retriever.load(load_reranker=False)` or `.retrieve(query, k, rerank=False)` → **FAISS-only**.
- `.retrieve(query, k, rerank=True)` → **FAISS + cross-encoder**.
- Query rewriting is `Toolbox.rewrite_retrieval_queries` (LLM) with
  `react_agent._fallback_queries` (deterministic) — so raw vs rewritten can be A/B'd offline.

---

## 3. Harness architecture (`eval/`)

Nothing in `scheduler/` must change for the baseline run.

```
eval/
├── PLAN.md                      # this file
├── datasets/
│   ├── prompts.jsonl            # 40–60 eval prompts (schema below)
│   ├── retrieval_gold.jsonl     # ~20 queries → relevant course codes (graded 0/1/2)
│   └── students/                # student fixtures (JSON/HTML) + gold profile labels
├── variants.py                  # ablation matrix → config objects
├── run.py                       # drives sessions; dumps eval/runs/<variant>/<id>.json
├── graders/
│   ├── validity.py              # hard-validity oracles on extracted codes
│   ├── profile.py               # extraction & requirement correctness
│   ├── retrieval.py             # Recall@k, MRR, nDCG (offline, no agent)
│   ├── efficiency.py            # trace mining: tool calls, steps, tokens, latency
│   └── tool_routing.py          # "right tool for the query" precision/recall
├── score.py                     # aggregate runs → metrics table + CIs
└── grade_sheet.py               # emit a blinded CSV for the human rubric
```

**Run/grade split:** `run.py` writes one JSON per (variant, prompt) containing
`answer`, `plan`, `raw_trace`, `usage_delta`, `latency_s`. Grading runs **offline**
over those dumps, so re-grading and adding metrics costs no API tokens.

**Prompt record schema:**
```jsonc
{
  "id": "rag_07",
  "category": "rag | degree_aware | transcript_backed | scheduling | adversarial",
  "student_fixture": "students/bscs_sem7.json",   // or null for profile-free RAG
  "turns": ["Suggest theoretical CS courses."],     // multi-turn supported
  "target_term": "202601",
  "expected_tools": ["retrieve_catalog_courses"],   // for trace completeness
  "gold": {                                          // optional, category-specific
    "relevant_section": "Core Elective",
    "relevant_codes": ["CS 301", "CS 403", "CS 412"],
    "must_not_recommend": ["CS 201"]
  }
}
```

---

## 4. Test set & fixtures

### 4.1 Prompts (target 40–60)

| Category | n | Drives |
|---|---|---|
| Pure RAG ("theoretical CS courses", "what does CS 412 cover") | 10 | Retrieval, hallucination, answer quality |
| Degree-aware ("IE + optimization, what fits") | 10 | Scoped-graph correctness, section validity, requirement engine |
| Transcript-backed ("what should I take for graduation") | 10 | Requirement correctness, prereq/credit validity |
| Scheduling ("can I add CS 412 to my schedule") | 10 | Conflict-free %, coreq inclusion, timetable routing |
| Adversarial (vague major, wrong title, "CS student" not BSCS-DM, missing transcript, impossible plan) | 10–20 | Honesty, no-hallucination, graceful degradation |

### 4.2 Student fixtures — the data gap

The repo ships **one** transcript (`data/transcript_cagan.json`) and **one**
degree-eval (`DEGREE EVALUATION.html`). That is too thin for the profile /
requirement / transcript-backed metrics, which need ~10 students.

**Recommended default (assumed unless changed): synthesize fixtures from the degree graphs.**
For a program/cohort, take `select_graphs(...).candidate_courses("Required")`,
mark a prefix of the topological order as `completed`, and the gold
`required_left` is **known by construction**. Vary progress (sem 3/5/7) and
program (BSCS, BSIE, BSEE, BSDSA) → ~12 fixtures. This turns "remaining
required-course exact match" into a closed-form check rather than a judgment call.
Keep the one real degree-eval HTML as a realism/parser anchor (hand-label its
program/admit/cohort and section credit deficits — its `section_requirements`
block is authoritative).

> **Open decision:** synthesize-only vs synthesize + real anchor vs user-provided
> real transcripts. Default above assumed; revisit before building `students/`.

### 4.3 Retrieval gold (`retrieval_gold.jsonl`)
~20 interest queries, each with a hand-labeled set of relevant course codes,
graded 0/1/2 for nDCG. Built once, reused across all retrieval configs. Offline,
so cheap to expand.

---

## 5. Ablation matrix → concrete toggles

Each variant is a config consumed by `run.py`. Cleanest implementation: a small
`AblationConfig` (dict of feature flags) threaded through the session and read by
`Toolbox`/`_prefetch_recommendation_context`, plus a tool-list filter applied in
`handle_turn`. The retriever/graph swaps need no code change.

| Variant | Switch |
|---|---|
| **Full** | `mode="react"`, reranker on, rewriting on, scoped graphs, requirement injection on |
| **No RAG** | Drop `retrieve_catalog_courses` + `get_degree_section_courses` from `TOOLS`; skip prefetch (catalog/graph/requirement tools remain) |
| **No reranker** | `Retriever.load(load_reranker=False)` (else identical) |
| **No query rewriting** | Bypass `rewrite_retrieval_queries`/prefetch; retrieve with the raw user string as the single query |
| **No scoped graph** | Force `degree_filter=False` in retrieval + validate against `data/SU_full_graph.gpickle` instead of `select_graphs(...).merged` |
| **No requirement engine** | `include_required=False` + strip `get_requirement_state` from `TOOLS`/prefetch |
| **No offering pattern** | Skip the `likely_offered_in` filter when building the eligible pool |
| **No coreq/schedule expansion** | Disable coreq auto-add in the `validate_plan`/timetable path |
| **Pipeline vs ReAct** | `mode="pipeline"` vs `mode="react"` |

---

## 6. Metric recipes (grouped as in the original spec)

### A. Profile / requirement correctness → `graders/profile.py`
- Program / admit-term / cohort accuracy: parse fixture
  (`degree_eval.parse_degree_evaluation` or `transcript.parse_transcript`),
  compare `program`, `admit_term`, `cohort_for_admit(...)` to gold → exact-match accuracy.
- Remaining required-course **exact match**:
  `set(compute_remaining(...).required_left) == gold` → exact-set accuracy + Jaccard for partial credit.
- Elective credit deficit (exact/near): compare parsed
  `section_requirements[*].{min_su, completed_su}` to gold → MAE in SU credits + within-±1 rate.
- Correct graph selected: `select_graphs(program, admit_term)` program+cohort == gold.

### B. Hard validity → `graders/validity.py`
Extract recommended codes via `catalog.extract_course_codes(answer)` (∪ `current_plan.plan`).
For each code, score and report as a **percentage over all recommended courses**
(Wilson 95% CIs):
- exists in catalog (`Catalog.get`)
- inside correct section (`code ∈ select_graphs(...).candidate_courses(gold_section)`)
- not already completed/in-progress (student sets)
- prereq-satisfied (`prereqs_satisfied`)
- schedule conflict-free (`build_timetable(...).ok`)
- required recitations/labs included (`corequisite_codes(...)` ⊆ recommended ∪ auto-added)

### C. RAG retrieval → `graders/retrieval.py` (offline, no agent — cleanest experiment)
Run three configs directly on `Retriever` over `retrieval_gold.jsonl`:
1. FAISS-only (`rerank=False`)
2. FAISS + reranker (`rerank=True`)
3. rewrite (`rewrite_retrieval_queries`) → FAISS + reranker

Compute **Recall@5, Recall@10, MRR, nDCG@10**, bootstrap CIs over queries.
Isolates RAG from LLM noise; directly proves the reranker and rewriting claims.

### D. Recommendation quality → `grade_sheet.py` (human, 1–5)
Emit a **blinded** CSV: one row per (prompt, variant), variant IDs hidden, rows
shuffled, columns for the five axes (relevance, degree usefulness, explanation,
honesty about uncertainty, no hallucinated courses). Two graders on a ~30% overlap
→ Cohen's κ. "No hallucinated courses" is *also* auto-scored via
`_catalog_title_mismatches`, giving a cheap auto-metric the human pass validates.

### E. Agent efficiency → `graders/efficiency.py` (from trace + usage delta)
- tool calls/turn = count of `tool_result` events; ReAct iterations = count of `assistant_step` (max 8).
- latency = wall-clock around `handle_turn`.
- tokens/cost = `get_session_usage()` delta × model price.
- trace completeness / right-tool = from `expected_tools`, precision/recall of tool
  selection (scheduling → `build_timetable`; degree → `get_requirement_state`; RAG → `retrieve_catalog_courses`).
- Cross-mode caveat: pipeline mode has a fixed call structure → compare it to ReAct
  on **LLM calls / tokens / latency**, not "tool calls."

---

## 7. Headline demo metric

Both halves come straight out of the harness:
- **Invalid-recommendation rate** = (recommended courses failing any §B check) /
  (total recommended), for **No-RAG / RAG-only vs Full**, same prompts →
  "reduces invalid recommendations from X% to Y%."
- **Recall@10** (§C) for FAISS-only vs rewrite+rerank → "while improving
  relevant-course Recall@10."

Use **McNemar's test** (paired) for the validity drop and bootstrap CIs for the
recall gain.

---

## 8. Rigor & cost

- **Paired design:** identical prompt set across all variants.
- **Determinism:** planner temperature 0; pin and record the model snapshot
  (e.g. `gpt-4o-2024-xx-xx`). The agent is still stochastic → run **N=3 repeats**
  for LLM-dependent metrics (mean ± std). Deterministic graders (B, C, profile)
  need no repeats.
- **Cost guard:** ~50 prompts × 9 variants × 3 repeats ≈ 1,350 ReAct runs.
  Run deterministic/retrieval metrics at full scale (cheap). For the ablation
  sweep, consider `gpt-4o-mini` and/or N=1 on a 25-prompt subset, keeping full
  `gpt-4o` × N=3 only for **Full** and the headline comparison.

---

## 9. Phasing

1. **Day 1–2:** `run.py` + dump format + synthetic student fixtures + `prompts.jsonl` (start with 20).
2. **Day 2–3:** deterministic graders (validity, profile, efficiency) — no labeling, immediate numbers.
3. **Day 3–4:** `retrieval_gold.jsonl` + retrieval grader (cleanest, most citable result).
4. **Day 4–5:** `AblationConfig` flags + run the matrix; `score.py` aggregation with CIs.
5. **Day 5–6:** blinded human rubric on the best variant + headline comparison; assemble the demo slide.

---

## 10. Open decisions

- **Fixtures source** (see §4.2): synthesize-only (default) vs + real anchor vs user-provided real transcripts.
- **Ablation model**: `gpt-4o` everywhere vs `gpt-4o-mini` for the sweep to control cost.
- **Repeats**: N=3 for all LLM-dependent runs vs N=3 only for Full + headline.
