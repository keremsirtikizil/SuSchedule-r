# SuSchedule-r — Demo Script (~3–4 minutes)

A tight, reproducible walkthrough for the live/recorded demo. Have the server
running first:

```bash
uvicorn api.main:app --port 8000     # then open http://localhost:8000
```

Keep `data/transcript_cagan.json` handy for the upload step.

---

## 0. One-line framing (10s)
> "SuSchedule-r is a course advisor that never guesses — every answer is grounded
> in Sabancı's real catalogue, prerequisite graph, degree requirements and weekly
> offerings. Chat on the left, build your schedule on the right."

## 1. Grounded Q&A (40s)
Type into the chat, one at a time:
- **"What are the prerequisites for CS 301?"** → answer names **CS 300** and **MATH 204** (from the prereq graph, not memory).
- **"What does CS 445 cover?"** → natural language processing.
- Point at the **Session & agent trace** drawer: show the tool calls (retrieval, requirement lookups) behind the answer.

> Talking point: "The trace shows the agent calling real tools — this is RAG +
> ReAct, not a chatbot hallucinating course codes."

## 2. Upload transcript → personalised planning (40s)
- Click **Upload transcript**, choose `transcript_cagan.json`.
- Show the student card (program BSCS-DM, completed/in-progress).
- Ask: **"What required courses do I still need?"** → grounded in the degree graph.

## 3. Build a schedule (60s) — the core feature
- In the right pane, add: **CS 201**, then **CS 204**, then **CS 412**.
- Note the **proxy-term banner**: Fall 2026–2027 isn't published, so times come
  from Fall 2025–2026 as a proxy (CRNs will differ) — honest by design.
- Point out the **calendar grid**: colour-coded blocks, room labels, half-hour lines.
- Add a course that **clashes** (e.g. CS 201 + CS 204 overlap on Wednesday) →
  the two blocks render **side-by-side with red conflict rings**, and the footer
  shows "2 with conflicts".
- Toggle **light/dark** in the header to show the themed UI.

## 4. "Check Schedule" — agent reviews your plan (40s)
- Click **Check Schedule** (bottom-right).
- The agent calls `get_current_schedule` and replies in chat with:
  the exact Wednesday conflict, the total SU credit load vs the 12–21 range,
  and (since a transcript is loaded) how the picks fit remaining requirements.

> Talking point: "The conflict the agent reports is computed server-side, not
> by the LLM — so the critique is grounded too."

## 5. Evaluation in one breath (30s)
> "We evaluated four ways, all reproducible in `eval/`:
> retrieval Hit@5 = 1.00 / Recall@10 = 0.96; conflict detection 100% on labelled
> pairs with a provably sound scheduler; 8/8 grounded end-to-end answers.
> And honestly — the cross-encoder re-ranker didn't help, and our requirement
> engine over-counts by 4 credits on a course-substitution case. Both are in the
> report's error analysis."

## Backup / Q&A facts
- Catalogue: **688 courses**; prereq graph **721 nodes / 729 edges**, verified DAG.
- Models: **BGE-base** bi-encoder + **BGE-reranker** cross-encoder; **GPT-4o** agent (**21 tools**).
- Cost/latency: ~**6k tokens**, ~**1–3 s** per question (26 s once, on first retriever load).
- Reproduce results: `python -m eval.run_retrieval | run_conflicts | run_requirements | run_agent`.
