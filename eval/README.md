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
| `python -m eval.run_agent` | Transcript-backed ReAct answer grounding, required-action success, task success, latency, tokens, and tool calls over `agent_questions.json` | **Yes** (`.env` → `OPENAI_API_KEY`) | `results_agent.json`, `agent_transcript.md` |
| `python -m eval.run_agent --dataset general_conversation_questions.json` | Transcript-free general conversation behavior: RAG use, smooth catalog answers, Check Schedule behavior, follow-up behavior, and no unauthorized planning actions | **Yes** (`.env` → `OPENAI_API_KEY`) | `results_agent_general_conversation.json`, `agent_general_conversation_transcript.md` |
| `python -m eval.run_agent --rescore` | Re-score saved raw agent answers after label edits without another API call | No | updates the result files for the selected dataset |

## Datasets (editable, version-controlled)

- `retrieval_queries.json` — natural-language queries with gold course codes.
- `agent_questions.json` — student questions with structured, checkable gold
  facts and optional required trace actions.
- `general_conversation_questions.json` — transcript-free chat questions with
  human labels for course suggestions, course explanations, follow-up planning
  behavior, and forbidden actions such as committing a plan too early.
- `submission_human_cases.json` — 20 representative human-labeled coverage
  cases used in the final report. This is a reviewer-facing coverage matrix,
  not a separate paid live runner.

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
- Agent labels support both `required_actions` and `forbidden_actions`. This is
  how the general-conversation set checks that the agent uses RAG for open
  exploration but does not call `set_plan` or `validate_plan` when the user has
  not provided enough profile context.
- Agent rows can include `schedule_picks`, which preloads the same kind of
  course selections that the web UI schedule builder sends to the backend.
  Those rows test the Check Schedule button path: the agent should call
  `get_current_schedule`, not rebuild a new timetable from scratch.
- `run_agent.py` runs one conversational session with a real transcript loaded,
  when the selected dataset has a `transcript` field. If `transcript` is null,
  the run starts with no profile context, matching first-contact chat use.
  Later questions still see earlier context in the same run.
- Retrieval evaluation requires all four local artifacts under `embeddings/`:
  parquet metadata, ID map, embedding matrix, and FAISS index. Do not compare a
  saved result against a changed query dataset without regenerating it.
- `run_retrieval.py` fails if the FAISS index does not load or if no query uses
  FAISS. The result JSON records index type, vector count, and FAISS search
  count.
- The requirements runner treats the audit's `GENERAL` row as the authoritative
  overall total. Section rows overlap and are not additive.
- The suite does not currently measure runtime LLM query-rewrite quality.
