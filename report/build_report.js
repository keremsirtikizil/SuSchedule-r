/* Build the CS 455 final report for SuSchedule-r as a .docx.
   Run: NODE_PATH=$(npm root -g) node report/build_report.js  */
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType, ShadingType,
  TableOfContents, PageNumber, PageBreak, Header, Footer,
} = require("docx");

const ACCENT = "002776";       // Sabanci navy
const GREY = "5C657F";
const HEADFILL = "D8E0F0";
const ZEBRA = "F1F4FA";
const CW = 9360;               // content width (US Letter, 1" margins)

const B = { style: BorderStyle.SINGLE, size: 1, color: "C9D2E5" };
const BORDERS = { top: B, bottom: B, left: B, right: B };

function P(text, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after ?? 120, before: opts.before ?? 0, line: 276 },
    alignment: opts.align,
    children: [new TextRun({ text, bold: opts.bold, italics: opts.italics,
      color: opts.color, size: opts.size })],
  });
}
function H1(text) { return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] }); }
function H2(text) { return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] }); }
function bullet(text) {
  return new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 80, line: 268 },
    children: [new TextRun(text)] });
}
function bulletRich(runs) {
  return new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 80, line: 268 }, children: runs });
}

// Build a table from a header row + body rows (arrays of strings). colW sums to CW.
function tbl(header, rows, colW, opts = {}) {
  const headCells = header.map((h, i) => new TableCell({
    borders: BORDERS, width: { size: colW[i], type: WidthType.DXA },
    shading: { fill: HEADFILL, type: ShadingType.CLEAR },
    margins: { top: 70, bottom: 70, left: 110, right: 110 },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : (opts.numAlign ?? AlignmentType.RIGHT),
      children: [new TextRun({ text: h, bold: true, size: 19, color: ACCENT })] })],
  }));
  const bodyRows = rows.map((r, ri) => new TableRow({
    children: r.map((c, i) => new TableCell({
      borders: BORDERS, width: { size: colW[i], type: WidthType.DXA },
      shading: ri % 2 ? { fill: ZEBRA, type: ShadingType.CLEAR } : undefined,
      margins: { top: 60, bottom: 60, left: 110, right: 110 },
      children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : (opts.numAlign ?? AlignmentType.RIGHT),
        children: [new TextRun({ text: String(c), size: 19,
          bold: opts.boldFirstCol && i === 0 })] })],
    })),
  }));
  return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: colW,
    rows: [new TableRow({ tableHeader: true, children: headCells }), ...bodyRows] });
}
function caption(text) {
  return new Paragraph({ spacing: { before: 60, after: 200 },
    children: [new TextRun({ text, italics: true, size: 17, color: GREY })] });
}

const children = [];

/* ---------------- Title block ---------------- */
children.push(new Paragraph({ spacing: { before: 1600, after: 0 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "SuSchedule-r", bold: true, size: 56, color: ACCENT })] }));
children.push(new Paragraph({ spacing: { after: 240 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "A Retrieval-Augmented, Tool-Using Course-Planning Agent", size: 28 })] }));
children.push(new Paragraph({ spacing: { after: 600 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "for Sabancı University", size: 28 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
  children: [new TextRun({ text: "CS 455 / CS 555 — Large Language Models", size: 24, bold: true })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
  children: [new TextRun({ text: "Final Project Report", size: 24, color: GREY })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
  children: [new TextRun({ text: "Sabancı University", size: 22, color: GREY })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
  children: [new TextRun({ text: "Abstract", bold: true, size: 22 })] }));
children.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 200, line: 276 },
  indent: { left: 720, right: 720 },
  children: [new TextRun({ size: 20, text:
    "SuSchedule-r is a course-advising assistant that grounds a large-language-model agent in Sabancı University’s real catalogue, prerequisite graph, degree requirements and weekly section offerings. It combines a two-stage semantic retriever (BGE bi-encoder + cross-encoder re-ranker over 688 courses) with a tool-using ReAct agent (21 tools, GPT-4o) and an interactive web UI that includes a drag-free weekly schedule builder with live time-conflict detection. We evaluate the system on four axes — retrieval quality, time-conflict detection, degree-requirement accuracy, and end-to-end answer grounding — using labelled, reproducible test sets. Retrieval reaches Hit@5 = 1.00 and Recall@10 = 0.96; conflict detection is 100% accurate on labelled pairs with a provably sound scheduler; and the end-to-end agent answered 8/8 factual questions with correctly grounded facts. We also report, honestly, where the system falls short: the cross-encoder re-ranker does not help (and slightly hurts) on short topical queries, the graph-based requirement engine over-counts remaining required credits by 4 SU relative to the official audit due to a course-substitution gap, and future-term scheduling relies on a proxy term whose CRNs do not match real registration." })] }));
children.push(new Paragraph({ children: [new PageBreak()] }));

/* ---------------- TOC ---------------- */
children.push(new Paragraph({ spacing: { after: 160 }, children: [new TextRun({ text: "Contents", bold: true, size: 28, color: ACCENT })] }));
children.push(new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }));
children.push(new Paragraph({ children: [new PageBreak()] }));

/* ---------------- 1. Introduction ---------------- */
children.push(H1("1. Introduction"));
children.push(P("Every semester, Sabancı University students assemble a course plan under a web of constraints: unmet degree requirements, prerequisite chains, credit bounds, and — once a plan is chosen — non-overlapping weekly meeting times across the chosen sections. General-purpose chatbots answer these questions fluently but unreliably: they hallucinate course codes, invent prerequisites, and cannot see the student’s transcript or the term’s real offerings."));
children.push(P("SuSchedule-r addresses this by grounding every factual claim in university data. The guiding design principle throughout the project is simple: never let the model answer from memory when a data-backed tool or injected context can answer instead. The system retrieves relevant courses semantically, computes eligibility and remaining requirements from a parsed prerequisite graph and the student’s official degree evaluation, and detects time conflicts deterministically from published section meeting times."));
children.push(P("This report documents the system, its evaluation, and an honest account of its limitations. Contributions are: (i) a grounded RAG + ReAct advising agent over the full SU catalogue; (ii) an interactive schedule builder with calendar-style conflict visualisation; and (iii) a reproducible four-part evaluation with labelled test sets.", { after: 200 }));

/* ---------------- 2. Data ---------------- */
children.push(H1("2. Data and Knowledge Base"));
children.push(P("All knowledge is scraped from public SU sources and the student’s own transcript / degree-evaluation export. The scraper pipeline enumerates courses from degree pages, fetches each course-detail page, and parses prerequisite expressions into a directed graph."));
children.push(tbl(
  ["Asset", "Contents", "Scale"],
  [
    ["Course catalogue", "Code, title, description, SU credit, ECTS, prereq text, programs, seasons", "688 courses"],
    ["Prerequisite graph", "Parsed prereq DAG over all courses", "721 nodes, 729 edges"],
    ["Degree graphs", "Per-(program, cohort) requirement sub-graphs", "Required / Core / Area / Free"],
    ["Offerings", "Per-term sections with normalised weekly meeting times", "Terms 202401–202503"],
    ["Student transcript", "Completed / in-progress courses, CGPA, program, admit term", "BSCS-DM sample"],
    ["FAISS index", "BGE-base embeddings of every course (cosine / inner product)", "688 × 768-d vectors"],
  ],
  [2400, 5160, 1800], { numAlign: AlignmentType.LEFT, boldFirstCol: true }));
children.push(caption("Table 1. Data assets underpinning SuSchedule-r."));
children.push(P("Prerequisite parsing was validated to 100% coverage (audit: 426 strict expressions, 0 failures) and the resulting full graph is verified acyclic (a DAG), which is a precondition for sound eligibility reasoning.", { after: 200 }));

/* ---------------- 3. Architecture ---------------- */
children.push(H1("3. System Architecture"));
children.push(H2("3.1 Two-stage semantic retriever"));
children.push(P("Stage 1 is a BGE-base-en-v1.5 bi-encoder: every course is embedded once at build time and stored in a FAISS inner-product index; at query time the query is embedded and the top candidates are retrieved by cosine similarity, with optional pre-filters (eligible set, subject, season, level). Stage 2 is an optional BGE-reranker-base cross-encoder that re-scores each (query, course) pair for higher precision."));
children.push(H2("3.2 Tool-using agent"));
children.push(P("The agent runs in a ReAct loop (GPT-4o) with 21 tools: profile and requirement lookups, semantic retrieval, prerequisite checks, a science/engineering-credit finder, minor-progress tracking, timetable construction, and a get-current-schedule tool that lets the UI ask the agent to review the student’s assembled plan. A legacy fixed 8-stage “pipeline” mode is also available. Student profile, remaining requirements, the current date, and the planning term are injected into context so the agent grounds answers rather than guessing."));
children.push(H2("3.3 Web UI and schedule builder"));
children.push(P("The web UI (FastAPI + vanilla JS) places the AI advisor on the left and a weekly schedule builder on the right. Students add courses by code; the builder fetches the real sections, renders a Monday–Friday calendar with colour-coded, lane-split blocks, and flags time conflicts client-side. A “Check Schedule” button posts the assembled plan to the agent, which calls get-current-schedule and reviews conflicts, credit load, and requirement fit. Because the planning term (Fall 2026–2027) has no published offerings yet, the builder falls back to the most recent same-season term as a proxy and warns that CRNs/section numbers will differ."));
children.push(P("The UI was restyled to a Sabancı-branded design with a light/dark theme toggle; an explicit disclaimer notes that the assistant is an AI model and its results should be double-checked.", { after: 200 }));

/* ---------------- 4. Methodology ---------------- */
children.push(H1("4. Evaluation Methodology"));
children.push(P("We evaluate four components with labelled, version-controlled test sets in the eval/ directory. Three runners are fully local (no API cost); the end-to-end runner makes live GPT-4o calls."));
children.push(bullet("Retrieval. 34 natural-language student queries, each with a hand-assigned gold set of relevant course codes. We report Hit@1/@5, Recall@5/@10, MRR@10 and nDCG@10, with the cross-encoder re-ranker ON (production) and OFF (ablation)."));
children.push(bullet("Time-conflict detection. 12 labelled meeting pairs covering the tricky cases (touching-but-not-overlapping, shared-day, multi-day, one-minute boundaries) for detector accuracy; plus a soundness check that the backtracking scheduler never returns overlapping picks on real course bundles."));
children.push(bullet("Degree requirements. The student’s official Banner Degree Evaluation (per-section SU credits) is the ground truth, compared against the graph-based requirement engine."));
children.push(bullet("End-to-end agent. 8 student questions with checkable gold facts (credits, prerequisites, topics) drawn from the catalogue, run through the full ReAct agent with a real transcript loaded. An answer is “grounded” iff it contains the correct fact(s) and no hallucination markers; we also record latency, token cost and tool-call count."));
children.push(P("Test sets are intentionally small and high-precision rather than exhaustive; results are therefore indicative, not population estimates. All numbers below are reproduced by the scripts in eval/.", { after: 200 }));

/* ---------------- 5. Results ---------------- */
children.push(H1("5. Results"));
children.push(H2("5.1 Retrieval quality"));
children.push(tbl(
  ["Metric", "Re-ranker ON", "Re-ranker OFF", "Δ (ON−OFF)"],
  [
    ["Hit@1", "0.912", "0.971", "−0.059"],
    ["Hit@5", "1.000", "1.000", "+0.000"],
    ["Recall@5", "0.941", "0.966", "−0.025"],
    ["Recall@10", "0.961", "0.975", "−0.015"],
    ["MRR@10", "0.956", "0.985", "−0.029"],
    ["nDCG@10", "0.924", "0.963", "−0.039"],
  ],
  [2760, 2200, 2200, 2200], { boldFirstCol: true }));
children.push(caption("Table 2. Retrieval metrics over 34 labelled queries (k=10). The bi-encoder alone is already strong; the cross-encoder re-ranker slightly reduces every metric on this query distribution."));
children.push(P("The retriever finds at least one gold course in the top 5 for every query (Hit@5 = 1.00) and recovers 96–98% of all gold courses by rank 10. The re-ranker ablation is the headline surprise (see §6)."));
children.push(tbl(
  ["Category", "Recall@10", "n"],
  [
    ["CS core", "1.000", "8"],
    ["CS elective", "0.933", "15"],
    ["MATH", "1.000", "6"],
    ["EE", "0.833", "2"],
    ["IE", "1.000", "1"],
    ["Cross-subject", "1.000", "2"],
  ],
  [4360, 3000, 2000], { boldFirstCol: true }));
children.push(caption("Table 3. Recall@10 by query category (re-ranker ON)."));

children.push(H2("5.2 Time-conflict detection and scheduler soundness"));
children.push(tbl(
  ["Check", "Result"],
  [
    ["Labelled meeting pairs (n=12): accuracy", "1.000"],
    ["Labelled meeting pairs: precision / recall", "1.000 / 1.000"],
    ["Scheduler soundness: unsound results / bundles", "0 / 4"],
    ["Feasible bundles correctly scheduled conflict-free", "3 of 4"],
    ["Infeasible bundle correctly reported (no clash-free combo)", "1 of 4"],
  ],
  [6960, 2400], { numAlign: AlignmentType.LEFT, boldFirstCol: true }));
children.push(caption("Table 4. Conflict detector and backtracking scheduler. The detector classifies all boundary cases correctly; the scheduler never returns overlapping picks."));

children.push(H2("5.3 End-to-end agent grounding"));
children.push(tbl(
  ["Metric", "Value"],
  [
    ["Grounding accuracy", "1.000 (8 / 8)"],
    ["Average latency", "6.7 s / question"],
    ["Average token cost", "6,156 tokens / question"],
    ["Average tool calls", "0.5 / question"],
  ],
  [6360, 3000], { numAlign: AlignmentType.LEFT, boldFirstCol: true }));
children.push(caption("Table 5. End-to-end ReAct agent over 8 factual questions (GPT-4o, real transcript loaded)."));
children.push(P("Every answer contained the correct, catalogue-grounded fact. Prerequisite questions triggered an explicit lookup tool call; some credit/topic questions were answered from injected context without a tool call (discussed in §6). Latency is dominated by the multi-step ReAct loop and a one-time retriever load on the first retrieval-bearing question (the 26 s outlier)."));

children.push(H2("5.4 Degree-requirement accuracy"));
children.push(tbl(
  ["Requirement section (official audit)", "Min SU", "Done SU", "Remaining"],
  [
    ["University courses", "41", "41", "0"],
    ["Core electives", "31", "31", "0"],
    ["Required courses", "29", "26", "3"],
    ["Area electives", "9", "9", "0"],
    ["Free electives", "15", "15", "0"],
    ["Overall (Banner rollup)", "125", "122", "3"],
  ],
  [4360, 1660, 1660, 1680], { boldFirstCol: true }));
children.push(caption("Table 6. Official Degree Evaluation (ground truth): the student needs 3 more SU credits, all in the Required-courses bucket."));
children.push(P("The graph-based engine agrees the student is nearly finished but lists three required items still open — CS 395, ENS 492 and MATH 212 — totalling 7 SU, a +4 SU over-count against the official 3 SU. The cause is analysed in §6.", { after: 200 }));

/* ---------------- 6. Error analysis ---------------- */
children.push(H1("6. Error Analysis: What Did Not Work"));
children.push(P("Per the project brief, we report failures as carefully as successes."));
children.push(H2("6.1 The cross-encoder re-ranker does not help"));
children.push(P("We expected the BGE cross-encoder to improve precision. Instead it reduced every retrieval metric (Table 2): Hit@1 fell from 0.971 to 0.912 and MRR from 0.985 to 0.956. On short topical queries against a small, well-separated catalogue, the bi-encoder already ranks the exact-match course first; the cross-encoder occasionally demotes it in favour of a topically broader course. The re-ranker also adds ~0.5 s of CPU latency per query. Conclusion: for this query distribution the re-ranker is not worth its cost, and bi-encoder-only retrieval is the better default. The re-ranker is retained behind a flag for longer, more ambiguous queries where it may still help."));
children.push(H2("6.2 Requirement engine over-counts required credits"));
children.push(P("The engine reports MATH 212 (Linear Algebra and Differential Equations) as an open requirement, but the official audit does not. The student satisfied this requirement through an accepted substitution (the separate MATH 201 + MATH 202 sequence), which the degree-graph encoding does not treat as equivalent. Together with how internship/graduation-project credits (CS 395, ENS 492) are counted, this produces the +4 SU over-count. The engine is therefore conservative — it will not under-state what is owed — but it needs an explicit course-equivalence/substitution table to match Banner exactly."));
children.push(H2("6.3 Future-term scheduling relies on a proxy term"));
children.push(P("The planning term Fall 2026–2027 (202601) has no published offerings, so the builder schedules against the most recent same-season term (Fall 2025–2026, 202501). Meeting days/times are therefore indicative and the CRNs/section numbers shown will not match actual registration — the UI states this explicitly, and proxy CRNs are scrubbed from agent output. A side effect: some course bundles that will likely be feasible in the real term are reported infeasible in the proxy term (e.g. CS 300 + CS 301 + MATH 201 had no conflict-free combination in 202501 offerings)."));
children.push(H2("6.4 Grounding without explicit retrieval"));
children.push(P("Several credit/topic questions were answered correctly with zero tool calls, relying on context injected into the prompt (and, possibly, the model’s prior). Answers were verified correct, but the system does not yet force a tool call for every factual claim, so grounding is not always mechanically traceable. Requiring a citation-producing tool call for atomic facts would make grounding auditable."));
children.push(H2("6.5 Other limitations"));
children.push(bullet("Latency variance: the ReAct loop plus a lazy retriever load produces a 26 s worst-case on the first retrieval-bearing question; subsequent queries are 1–3 s."));
children.push(bullet("Cost: ~6k tokens per question on GPT-4o; an earlier experiment with smaller/cheaper models produced noticeably weaker scheduling explanations."));
children.push(bullet("UI state: reloading the page resets the client-side builder and starts a new session, so an in-progress schedule is not persisted."));
children.push(bullet("Evaluation scale: 34 retrieval and 8 end-to-end items are high-precision but small; numbers are indicative and one program (BSCS-DM) was tested in depth."));

/* ---------------- 7. Reproducibility ---------------- */
children.push(H1("7. Reproducibility"));
children.push(P("The repository is runnable from a clean clone. Setup, data-pipeline, FAISS build, CLI, web UI and data formats are documented in the top-level README; the evaluation suite has its own eval/README. Secrets are handled safely: .env is git-ignored and only .env.example (a placeholder) is committed."));
children.push(bulletRich([new TextRun({ text: "Install: ", bold: true }), new TextRun("pip install -r requirements.txt (Python 3.10+).")]));
children.push(bulletRich([new TextRun({ text: "Key: ", bold: true }), new TextRun("cp .env.example .env and set OPENAI_API_KEY.")]));
children.push(bulletRich([new TextRun({ text: "Index: ", bold: true }), new TextRun("the FAISS index is prebuilt in embeddings/ (rebuild script documented).")]));
children.push(bulletRich([new TextRun({ text: "Evaluate: ", bold: true }), new TextRun("python -m eval.run_retrieval / run_conflicts / run_requirements / run_agent.")]));
children.push(bulletRich([new TextRun({ text: "Web UI: ", bold: true }), new TextRun("uvicorn api.main:app and open the served page.")]));
children.push(P("Note: a stray local .venv built on Python 3.9 exists in the working tree; the supported interpreter is 3.10+ (the system Python used for all evaluation here is 3.12). The eval test sets are JSON and version-controlled, so any reviewer can extend them and re-run to regenerate the tables in §5.", { after: 200 }));

/* ---------------- 8. Conclusion ---------------- */
children.push(H1("8. Conclusion"));
children.push(P("SuSchedule-r shows that a disciplined “ground everything” approach turns an LLM into a trustworthy course advisor: strong semantic retrieval (Hit@5 = 1.00), a provably sound conflict checker (100% on labelled pairs), and 8/8 grounded end-to-end answers. Equally important are the negative results — the re-ranker that does not earn its keep, the requirement engine’s substitution gap, and the proxy-term caveat — each of which points to a concrete next step: drop the re-ranker by default, add a course-equivalence table, and ingest real offerings once the registrar publishes them. The system is reproducible end-to-end and the evaluation harness makes future regression-checking straightforward."));

/* ================= assemble ================= */
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Calibri", size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Calibri", color: ACCENT },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, font: "Calibri", color: "1C47A8" },
        paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
      alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 600, hanging: 280 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "SuSchedule-r — CS 455 Final Report     ", size: 16, color: GREY }),
        new TextRun({ children: ["Page ", PageNumber.CURRENT], size: 16, color: GREY })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(__dirname + "/CS455_SuSchedule-r_Final_Report.docx", buf);
  console.log("wrote report/CS455_SuSchedule-r_Final_Report.docx");
});
