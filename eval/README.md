# Evaluation suite

Reproducible evaluation for SuSchedule-r. Three of the four runners are fully
local (no API key, no cost); only `run_agent.py` calls OpenAI.

Run everything from the repo root with the project's Python env (3.10+, deps
from `requirements.txt`; the FAISS index in `embeddings/` must be built — see
the main README §6).

## Runners

| Script | What it measures | Needs OpenAI? | Output |
|---|---|---|---|
| `python -m eval.run_retrieval` | Retrieval quality (Recall@k, MRR, nDCG) + cross-encoder reranker ablation, over `retrieval_queries.json` (34 labeled queries) | No | `results_retrieval.json` |
| `python -m eval.run_conflicts` | Time-conflict detector accuracy on labeled meeting pairs + `build_timetable` soundness on real bundles | No | `results_conflicts.json` |
| `python -m eval.run_requirements` | Degree-requirement remaining-set vs the official Degree Evaluation HTML (precision/recall) | No | `results_requirements.json` |
| `python -m eval.run_eligibility` | Prereq engine: in-progress courses satisfy hard prereqs for the future term (regression guard, graph-derived cases) | No | `results_eligibility.json` |
| `python -m eval.run_agent` | End-to-end ReAct agent grounding accuracy, latency, tokens, tool calls, over `agent_questions.json` | **Yes** (`.env` → `OPENAI_API_KEY`) | `results_agent.json`, `agent_transcript.md` |

## Datasets (editable, version-controlled)

- `retrieval_queries.json` — natural-language queries with gold course codes.
- `agent_questions.json` — student questions with checkable gold facts
  (`must_contain` / `any_of` / `must_not`) drawn from the catalog.

To grow the evaluation, add entries to those JSON files and re-run; the metrics
scale automatically.

## Notes

- Gold labels are hand-assigned from the SU catalog titles/descriptions and the
  student's official Degree Evaluation; they are intentionally small and
  high-precision rather than exhaustive.
- `run_agent.py` runs one conversational session with a real transcript loaded,
  so later questions can see earlier context (matching real product use).
- `run_retrieval.py` also reports the **search backend actually used**: stage 1
  queries the FAISS `IndexFlatIP`, restricted to the eligibility-filtered row ids
  via an `IDSelector` (`search_backend: faiss-flat-ip`). The index is exact, so
  results equal the numpy fallback used when faiss/the index is unavailable.
- The cross-encoder reranker is **on by default** (`retriever.RERANK_DEFAULT = True`),
  forming the full two-stage pipeline; if the reranker model is unavailable,
  retrieval transparently falls back to bi-encoder order. `run_retrieval.py`
  reports both ON and OFF as an ablation — note the reranker did not improve these
  short-query metrics, so the ablation table is the place to inspect its effect.
