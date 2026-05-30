# Evaluation Harness

The evaluation has three layers. Start cheap, then spend API tokens only where
the LLM is actually part of the question.

## 1. Offline deterministic regression suite

```bash
source .venv/bin/activate
python -m eval.run_offline
```

This checks graph aliases, cohort mapping, scoped degree graphs, Degree
Evaluation HTML loading, requirement de-duplication, prerequisite `AND` / `OR`
evaluation, offering likelihood, timetable overlap detection, recitation/lab
corequisites, scoped lexical fallback retrieval, and local RAG-artifact
alignment when generated artifacts are present.

## 2. Retrieval ablations

The lexical baseline runs without loading embedding models:

```bash
python -m eval.run_retrieval --backends lexical
```

Compare retrieval configurations after the BGE models are cached locally:

```bash
python -m eval.run_retrieval \
  --backends lexical faiss faiss_rerank
```

Add the hand-reviewed internal query rewrites:

```bash
python -m eval.run_retrieval \
  --backends lexical faiss faiss_rerank \
  --use-rewrites \
  --json eval/reports/retrieval_with_rewrites.json
```

Reported metrics: `Recall@5`, `Recall@10`, `MRR`, and `nDCG@10`.

## 3. Online ReAct evaluation

This consumes OpenAI API tokens. Start with two prompts:

```bash
python -m eval.run_agent --variant react_full --limit 2
```

Then compare the implemented agent variants:

```bash
python -m eval.run_agent --variant react_full
python -m eval.run_agent --variant pipeline
python -m eval.run_agent --variant react_no_required_injection
```

Each run saves one JSON artifact per prompt plus `summary.json` under
`eval/runs/`. Every artifact contains the answer, raw agent trace, tool calls,
latency, token usage, extracted course codes, and hallucinated-code check.

Compare completed runs:

```bash
python -m eval.score_agent_runs \
  eval/runs/<react-run>/summary.json \
  eval/runs/<pipeline-run>/summary.json
```

The richer ablation design, including future toggles such as no RAG and no
offering filter, lives in [`PLAN.md`](PLAN.md).
