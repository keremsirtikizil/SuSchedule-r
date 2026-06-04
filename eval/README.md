# Evaluation suite

Reproducible evaluation for SuSchedule-r. Three of the four runners are fully
local (no API key, no cost); only `run_agent.py` calls OpenAI.

Run everything from the repo root with the project's Python env (3.10+, deps
from `requirements.txt`; the FAISS index in `embeddings/` must be built — see
the main README §6).

## Runners

| Script | What it measures | Needs OpenAI? | Output |
|---|---|---|---|
| `python -m eval.run_retrieval` | Retrieval quality (Recall@k, MRR, nDCG) + cross-encoder reranker ablation, over `retrieval_queries.json` (44 labeled queries) | No | `results_retrieval.json` |
| `python -m eval.run_conflicts` | Time-conflict detector accuracy + corequisite-expanded `build_timetable` soundness and feasibility on real bundles | No | `results_conflicts.json` |
| `python -m eval.run_requirements` | Official overall completion and Required-section credit-gap comparison against the graph engine | No | `results_requirements.json` |
| `python -m eval.run_agent` | End-to-end ReAct answer grounding, required-action success, task success, latency, tokens, and tool calls over `agent_questions.json` | **Yes** (`.env` → `OPENAI_API_KEY`) | `results_agent.json`, `agent_transcript.md` |
| `python -m eval.run_agent --rescore` | Re-score saved raw agent answers after label edits without another API call | No | updates `results_agent.json`, `agent_transcript.md` |

## Datasets (editable, version-controlled)

- `retrieval_queries.json` — natural-language queries with gold course codes.
- `agent_questions.json` — student questions with structured, checkable gold
  facts and optional required trace actions.

To grow the evaluation, add entries to those JSON files and re-run; the metrics
scale automatically.

## Notes

- Gold labels are hand-assigned from the SU catalog titles/descriptions and the
  student's official Degree Evaluation; they are intentionally small and
  high-precision rather than exhaustive.
- Retrieval entries marked `review_status: "provisional"` are exploratory
  cross-domain additions. Human reviewers should expand or correct their gold
  sets before those rows are used in final reported metrics.
- Agent entries may also be marked provisional when the accepted answer set is
  not yet exhaustive or the expected scope is ambiguous. The result file
  reports task success separately by review status.
- `run_agent.py` runs one conversational session with a real transcript loaded,
  so later questions can see earlier context (matching real product use).
- Retrieval evaluation requires all four local artifacts under `embeddings/`:
  parquet metadata, ID map, embedding matrix, and FAISS index. Do not compare a
  saved result against a changed query dataset without regenerating it.
- `run_retrieval.py` fails if the FAISS index does not load or if no query uses
  FAISS. The result JSON records index type, vector count, and FAISS search
  count.
- The requirements runner treats the audit's `GENERAL` row as the authoritative
  overall total. Section rows overlap and are not additive.
- The suite does not currently measure runtime LLM query-rewrite quality.
