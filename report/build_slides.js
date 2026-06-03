/* CS 455 presentation for SuSchedule-r.
   Run: NODE_PATH=$(npm root -g) node report/build_slides.js  */
const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.defineLayout({ name: "W", width: 13.333, height: 7.5 });
p.layout = "W";

const NAVY = "002776", NAVY2 = "1C47A8", ICE = "E8EEFA", INK = "18203A",
      GREY = "5C657F", WHITE = "FFFFFF", GREEN = "1F8F6A", RED = "C43D36",
      AMBER = "C08A3A", LIGHT = "F4F7FC";
const HFONT = "Georgia", BFONT = "Calibri";

function titleBar(s, t, kicker) {
  s.addText(kicker.toUpperCase(), { x: 0.6, y: 0.45, w: 12, h: 0.3, fontFace: BFONT,
    fontSize: 12, bold: true, color: NAVY2, charSpacing: 2 });
  s.addText(t, { x: 0.6, y: 0.72, w: 12.1, h: 0.8, fontFace: HFONT, fontSize: 30, bold: true, color: NAVY });
}
function statCard(s, x, y, w, big, label, color) {
  s.addShape(p.ShapeType.roundRect, { x, y, w, h: 1.9, fill: { color: LIGHT },
    line: { color: "D8E0F0", width: 1 }, rectRadius: 0.08 });
  s.addText(big, { x, y: y + 0.18, w, h: 0.95, align: "center", fontFace: HFONT,
    fontSize: 40, bold: true, color: color || NAVY });
  s.addText(label, { x: x + 0.15, y: y + 1.12, w: w - 0.3, h: 0.65, align: "center",
    fontFace: BFONT, fontSize: 12.5, color: GREY });
}
function card(s, x, y, w, h, head, body, accent) {
  s.addShape(p.ShapeType.roundRect, { x, y, w, h, fill: { color: WHITE },
    line: { color: "D8E0F0", width: 1 }, rectRadius: 0.06, shadow: { type: "outer", blur: 6, offset: 2, angle: 90, color: "C9D2E5", opacity: 0.5 } });
  s.addShape(p.ShapeType.rect, { x, y, w: 0.09, h, fill: { color: accent || NAVY } });
  s.addText(head, { x: x + 0.25, y: y + 0.16, w: w - 0.4, h: 0.4, fontFace: BFONT, fontSize: 15, bold: true, color: NAVY });
  s.addText(body, { x: x + 0.25, y: y + 0.62, w: w - 0.45, h: h - 0.75, fontFace: BFONT,
    fontSize: 12.5, color: INK, lineSpacingMultiple: 1.05, valign: "top" });
}

/* 1. Title */
let s = p.addSlide(); s.background = { color: NAVY };
s.addShape(p.ShapeType.rect, { x: 0, y: 0, w: 13.333, h: 0.18, fill: { color: AMBER } });
s.addText("SuSchedule-r", { x: 0.8, y: 2.2, w: 11.7, h: 1.1, fontFace: HFONT, fontSize: 60, bold: true, color: WHITE });
s.addText("A retrieval-augmented, tool-using course-planning agent for Sabancı University",
  { x: 0.82, y: 3.35, w: 11.4, h: 0.7, fontFace: BFONT, fontSize: 20, color: ICE });
s.addText("CS 455 / CS 555 — Large Language Models · Final Project Report",
  { x: 0.82, y: 4.5, w: 11.4, h: 0.4, fontFace: BFONT, fontSize: 14, bold: true, color: WHITE });
s.addText("Grounding an LLM in the real catalogue, prerequisite graph, degree requirements and weekly offerings.",
  { x: 0.82, y: 4.95, w: 11.4, h: 0.5, fontFace: BFONT, fontSize: 13, italic: true, color: ICE });

/* 2. Problem */
s = p.addSlide(); s.background = { color: WHITE };
titleBar(s, "Planning a semester is a constraint problem", "The problem");
s.addText([
  { text: "Students must satisfy several things at once:", options: { fontSize: 15, bold: true, color: INK, breakLine: true, paraSpaceAfter: 8 } },
  { text: "Unmet degree requirements  ·  prerequisite chains", options: { fontSize: 14, color: INK, bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 6 } },
  { text: "Credit bounds (12–21 SU)  ·  workload balance", options: { fontSize: 14, color: INK, bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 6 } },
  { text: "Non-overlapping weekly meeting times across sections", options: { fontSize: 14, color: INK, bullet: { code: "2022" }, breakLine: true } },
], { x: 0.6, y: 1.9, w: 6.4, h: 3.0, valign: "top" });
card(s, 7.4, 1.9, 5.3, 3.6,
  "General chatbots fail here",
  "They answer fluently but unreliably: they hallucinate course codes, invent prerequisites, and cannot see your transcript or the term's real offerings.\n\nOur principle: never let the model answer from memory when a data-backed tool or injected context can answer instead.",
  RED);

/* 3. System at a glance */
s = p.addSlide(); s.background = { color: WHITE };
titleBar(s, "Three grounded components", "System architecture");
card(s, 0.6, 1.9, 3.9, 3.9, "1 · Semantic retriever",
  "Two-stage RAG over 688 courses:\n\n• BGE-base bi-encoder + FAISS inner-product index\n• optional BGE cross-encoder re-ranker\n• pre-filters: eligible set, subject, season, level", NAVY);
card(s, 4.7, 1.9, 3.9, 3.9, "2 · Tool-using agent",
  "ReAct loop on GPT-4o with 21 tools:\n\n• profile + requirement lookups\n• retrieval, prereq checks, minor tracking\n• timetable build + get-current-schedule\n• profile/term/date injected into context", NAVY2);
card(s, 8.8, 1.9, 3.9, 3.9, "3 · Web UI + builder",
  "FastAPI + vanilla JS:\n\n• chat advisor (left)\n• weekly schedule builder (right)\n• calendar grid, live conflict detection\n• Check-Schedule -> agent review\n• light/dark, Sabancı-branded", AMBER);

/* 4. Grounded in real data */
s = p.addSlide(); s.background = { color: WHITE };
titleBar(s, "Everything is grounded in university data", "Knowledge base");
statCard(s, 0.6, 2.1, 2.85, "688", "courses in the catalogue", NAVY);
statCard(s, 3.65, 2.1, 2.85, "721 / 729", "prereq graph nodes / edges (a verified DAG)", NAVY2);
statCard(s, 6.7, 2.1, 2.85, "100%", "prerequisite parse coverage (426 expressions, 0 fails)", GREEN);
statCard(s, 9.75, 2.1, 2.95, "21", "agent tools wired to real data", AMBER);
s.addText("Catalogue · prerequisite DAG · per-(program, cohort) degree graphs · per-term section offerings with weekly meeting times · the student's own transcript & official Degree Evaluation.",
  { x: 0.6, y: 4.5, w: 12.1, h: 0.9, fontFace: BFONT, fontSize: 14, italic: true, color: GREY, align: "center" });

/* 5. Schedule builder */
s = p.addSlide(); s.background = { color: WHITE };
titleBar(s, "Build your week, see conflicts instantly", "Feature highlight");
s.addText([
  { text: "Add courses by code → real sections are fetched", options: { fontSize: 14.5, color: INK, bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 8 } },
  { text: "Mon–Fri calendar: colour-coded, lane-split blocks with room labels", options: { fontSize: 14.5, color: INK, bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 8 } },
  { text: "Overlaps render side-by-side with red conflict rings", options: { fontSize: 14.5, color: INK, bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 8 } },
  { text: "“Check Schedule” → agent reviews conflicts, credit load & fit", options: { fontSize: 14.5, color: INK, bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 8 } },
  { text: "Conflicts are computed server-side — so the agent's critique is grounded too", options: { fontSize: 14.5, color: INK, bullet: { code: "2022" }, breakLine: true } },
], { x: 0.6, y: 1.95, w: 7.1, h: 3.6, valign: "top" });
card(s, 8.0, 1.95, 4.7, 3.5, "Honest by design: the proxy term",
  "Fall 2026–2027 has no published offerings yet, so the builder schedules against the most recent same-season term as a proxy.\n\nThe UI states clearly that days/times are indicative and CRNs/section numbers will differ from real registration.", AMBER);

/* 6. Evaluation methodology */
s = p.addSlide(); s.background = { color: WHITE };
titleBar(s, "Four reproducible evaluations", "How we evaluated");
card(s, 0.6, 1.9, 5.9, 1.75, "Retrieval quality",
  "34 labelled queries with gold course codes → Hit@k, Recall@k, MRR, nDCG, plus a re-ranker ablation.", NAVY);
card(s, 6.8, 1.9, 5.9, 1.75, "Time-conflict detection",
  "12 labelled meeting pairs (boundary cases) + soundness of the backtracking scheduler on real bundles.", NAVY2);
card(s, 0.6, 3.85, 5.9, 1.75, "Degree requirements",
  "Official Banner Degree Evaluation (per-section SU credits) as ground truth vs. the graph engine.", GREEN);
card(s, 6.8, 3.85, 5.9, 1.75, "End-to-end agent (live)",
  "8 questions with checkable gold facts through the full ReAct agent; grounding, latency, tokens, tool calls.", AMBER);
s.addText("Test sets are small and high-precision — results are indicative; all numbers reproduce from eval/.",
  { x: 0.6, y: 5.85, w: 12.1, h: 0.4, fontFace: BFONT, fontSize: 12.5, italic: true, color: GREY, align: "center" });

/* 7. Results: retrieval */
s = p.addSlide(); s.background = { color: WHITE };
titleBar(s, "Retrieval: strong — and a surprise", "Results · 1 of 3");
statCard(s, 0.6, 2.0, 3.0, "1.00", "Hit@5 (a gold course in the top 5, every query)", GREEN);
statCard(s, 3.8, 2.0, 3.0, "0.96", "Recall@10 (gold courses recovered)", NAVY);
statCard(s, 7.0, 2.0, 3.0, "0.96", "MRR@10 (rank of first gold)", NAVY2);
card(s, 10.2, 2.0, 2.5, 1.9, "Surprise", "The cross-encoder re-ranker slightly HURT every metric.", RED);
s.addText([
  { text: "Re-ranker ablation:  ", options: { bold: true, color: INK } },
  { text: "Hit@1 0.97 (off) vs 0.91 (on) · MRR 0.985 vs 0.956.  ", options: { color: INK } },
  { text: "On short topical queries the bi-encoder already ranks the exact match first; the re-ranker occasionally demotes it. Bi-encoder-only is the better default.", options: { color: GREY, italic: true } },
], { x: 0.6, y: 4.3, w: 12.1, h: 1.4, fontFace: BFONT, fontSize: 14, valign: "top" });

/* 8. Results: conflicts + agent */
s = p.addSlide(); s.background = { color: WHITE };
titleBar(s, "Conflicts sound, answers grounded", "Results · 2 of 3");
statCard(s, 0.6, 2.1, 2.85, "100%", "conflict-pair accuracy (12 labelled cases)", GREEN);
statCard(s, 3.65, 2.1, 2.85, "0", "unsound schedules (scheduler never overlaps)", GREEN);
statCard(s, 6.7, 2.1, 2.85, "8 / 8", "end-to-end answers correctly grounded", NAVY);
statCard(s, 9.75, 2.1, 2.95, "6.7 s", "avg latency · ~6.2k tokens / question", NAVY2);
s.addText("Prerequisite questions trigger an explicit lookup tool call; the agent's answers contained the correct catalogue facts with no hallucinations.",
  { x: 0.6, y: 4.5, w: 12.1, h: 0.8, fontFace: BFONT, fontSize: 14, italic: true, color: GREY, align: "center" });

/* 9. Error analysis */
s = p.addSlide(); s.background = { color: WHITE };
titleBar(s, "What didn't work (honestly)", "Results · 3 of 3");
card(s, 0.6, 1.9, 3.9, 3.7, "Re-ranker doesn't help",
  "The cross-encoder reduced every retrieval metric on short queries and adds ~0.5 s latency.\n\nFix: bi-encoder-only by default; keep the re-ranker behind a flag for ambiguous queries.", RED);
card(s, 4.7, 1.9, 3.9, 3.7, "Requirement over-count",
  "Engine: 7 SU required left (incl. MATH 212). Official audit: 3 SU left.\n\nCause: an accepted MATH 201+202 substitution the degree graph doesn't treat as equivalent.\n\nFix: add a course-equivalence table.", AMBER);
card(s, 8.8, 1.9, 3.9, 3.7, "Proxy-term caveat",
  "Future term has no offerings, so times come from a proxy term — CRNs won't match real registration, and some bundles look infeasible.\n\nFix: ingest real offerings once published.", NAVY2);

/* 10. Conclusion */
s = p.addSlide(); s.background = { color: NAVY };
s.addShape(p.ShapeType.rect, { x: 0, y: 0, w: 13.333, h: 0.18, fill: { color: AMBER } });
s.addText("Takeaways", { x: 0.8, y: 0.8, w: 11, h: 0.7, fontFace: HFONT, fontSize: 34, bold: true, color: WHITE });
s.addText([
  { text: "Grounding works: Hit@5 = 1.00, a provably sound conflict checker, 8/8 grounded answers.", options: { fontSize: 17, color: ICE, bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 12 } },
  { text: "Negative results matter: drop the re-ranker, add course equivalences, ingest real offerings.", options: { fontSize: 17, color: ICE, bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 12 } },
  { text: "Reproducible end-to-end: README + eval/ harness regenerate every number in the report.", options: { fontSize: 17, color: ICE, bullet: { code: "2022" }, breakLine: true } },
], { x: 0.9, y: 2.1, w: 11.4, h: 3.0, valign: "top" });
s.addText("python -m eval.run_retrieval | run_conflicts | run_requirements | run_agent",
  { x: 0.9, y: 5.5, w: 11.5, h: 0.5, fontFace: "Consolas", fontSize: 15, color: WHITE, fill: { color: NAVY2 }, align: "center" });

p.writeFile({ fileName: __dirname + "/CS455_SuSchedule-r_Slides.pptx" }).then(f => console.log("wrote", f));
