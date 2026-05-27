"""
Build the prerequisite DAG for the CS catalog.

Inputs:  data/<PROGRAM>_<TERM>_catalog.json
Outputs:
  data/<PROGRAM>_<TERM>_graph.json   -- nodes + edges + per-course prereq expr
  data/<PROGRAM>_<TERM>_graph.gpickle -- NetworkX pickle for downstream code
  data/<PROGRAM>_<TERM>_graph_report.txt -- sanity-check summary

A node = one course code.
Per-course attributes: title, su_credit, ects_text, description,
  prereq_expr (the parsed boolean tree), program_sections, offered_terms,
  coreq_text.
A directed edge u -> v means "course v lists course u as one of its prereqs".
Edge attributes: role ("required"|"alternative"), min_grade, concurrent.
  - "required": u appears under an AND chain only (must take u).
  - "alternative": u appears under at least one OR branch (one of several).

We also include nodes for "dangling" prereq courses that exist as prereqs
but are not in the program-catalog set; these are kept so eligibility checks
can still reason about them.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

sys.path.append(str(ROOT))
from scraper.prereq_parser import (  # noqa: E402
    has_unparsed,
    parse_prereq,
    parse_prereq_lenient,
)


def edge_roles_for_course(expr: dict | None) -> dict[str, dict]:
    """For a course's prereq tree, return {prereq_course: edge_attrs}.

    A course is "required" if every satisfying assignment must include it
    (i.e. it only appears under ANDs from the root). Otherwise it's
    "alternative" (lives under at least one OR fork).

    We compute this by walking the tree with a flag: under an OR branch,
    every leaf is "alternative". Under an AND branch the flag passes through.
    """
    out: dict[str, dict] = {}
    if expr is None:
        return out

    def walk(node: dict, alt_context: bool) -> None:
        if "course" in node:
            code = node["course"]
            prev = out.get(code)
            role = "alternative" if alt_context else "required"
            if prev is None:
                out[code] = {
                    "role": role,
                    "min_grade": node.get("min_grade", "D"),
                    "concurrent": node.get("concurrent", False),
                }
            else:
                # if it appears as required anywhere -> required wins
                if prev["role"] == "alternative" and role == "required":
                    prev["role"] = "required"
                # concurrent collapses to True only if all occurrences are concurrent
                prev["concurrent"] = prev["concurrent"] and node.get("concurrent", False)
            return
        op = node["op"]
        new_alt = alt_context or (op == "or")
        for child in node["operands"]:
            walk(child, new_alt)

    walk(expr, False)
    return out


def _resolve_tree(text: str | None, lenient: bool) -> tuple[dict | None, str | None]:
    """Parse a prereq_text. Returns (tree, strict_error_or_None).

    When lenient=True we fall back to parse_prereq_lenient on strict failure;
    this is needed for the unified SU catalog where non-CS programs surface
    grammar variants the strict parser doesn't accept.
    """
    try:
        return parse_prereq(text), None
    except Exception as exc:
        if not lenient:
            return None, str(exc)
        return parse_prereq_lenient(text), str(exc)


def build(
    term: str | None = None,
    program: str | None = "BSCS-DM",
    catalog_path: Path | None = None,
    out_prefix: str | None = None,
    lenient: bool = False,
) -> tuple[nx.DiGraph, Path]:
    """Build a prereq DAG.

    Two calling conventions:
      - Single-program (back-compat): pass term + program. Reads
        data/<PROGRAM>_<TERM>_catalog.json and writes the matching graph
        files. Strict prereq parser.
      - Full catalog: pass catalog_path=data/SU_full_catalog.json and
        out_prefix='SU_full'. Uses the lenient parser so non-CS grammar
        variants don't drop edges.
    """
    if catalog_path is None:
        catalog_path = DATA_DIR / f"{program}_{term}_catalog.json"
        out_prefix = out_prefix or f"{program}_{term}"
    else:
        out_prefix = out_prefix or catalog_path.stem
    if not catalog_path.exists():
        raise SystemExit(f"Missing catalog: {catalog_path}. Run the scraper first.")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

    graph_attrs: dict = {"source": catalog.get("source", "single_program")}
    if program:
        graph_attrs["program"] = program
    if term:
        graph_attrs["admit_term"] = term
    G = nx.DiGraph(**graph_attrs)

    parse_errors: list[tuple[str, str]] = []
    unparsed_courses: list[str] = []
    for cr in catalog["courses"]:
        code = cr["code"]
        tree, strict_err = _resolve_tree(cr.get("prereq_text"), lenient=lenient)
        if strict_err is not None:
            parse_errors.append((code, strict_err))
        if lenient and has_unparsed(tree):
            unparsed_courses.append(code)
        G.add_node(
            code,
            title=cr.get("title", ""),
            subj=cr.get("subj", ""),
            num=cr.get("num", ""),
            su_credit=cr.get("su_credit"),
            ects_text=cr.get("ects_text", ""),
            description=cr.get("description", ""),
            coreq_text=cr.get("coreq_text", ""),
            prereq_text=cr.get("prereq_text", ""),
            prereq_expr=tree,
            program_sections=cr.get("program_sections", []),
            offered_terms=[o["term"] for o in cr.get("offered_terms", [])],
            in_catalog=True,
        )

    # Edges
    for cr in catalog["courses"]:
        code = cr["code"]
        tree, _ = _resolve_tree(cr.get("prereq_text"), lenient=lenient)
        edges = edge_roles_for_course(tree)
        for prereq_code, attrs in edges.items():
            if not G.has_node(prereq_code):
                # Out-of-catalog prereq (older code, foreign equivalent, etc.).
                G.add_node(prereq_code, title="", in_catalog=False)
            G.add_edge(prereq_code, code, **attrs)

    # DAG check
    try:
        cycles = list(nx.simple_cycles(G))
    except Exception:
        cycles = []

    # Write outputs
    out_json = DATA_DIR / f"{out_prefix}_graph.json"
    out_pickle = DATA_DIR / f"{out_prefix}_graph.gpickle"
    out_report = DATA_DIR / f"{out_prefix}_graph_report.txt"

    out_json.write_text(
        json.dumps(
            {
                "program": program,
                "admit_term": term,
                "source": graph_attrs.get("source"),
                "nodes": [
                    {"code": n, **{k: v for k, v in attrs.items() if k != "prereq_expr"},
                     "prereq_expr": attrs.get("prereq_expr")}
                    for n, attrs in G.nodes(data=True)
                ],
                "edges": [
                    {"prereq": u, "course": v, **attrs}
                    for u, v, attrs in G.edges(data=True)
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with out_pickle.open("wb") as f:
        pickle.dump(G, f)

    # Report
    in_catalog = [n for n, d in G.nodes(data=True) if d.get("in_catalog")]
    out_catalog = [n for n, d in G.nodes(data=True) if not d.get("in_catalog")]
    req_edges = [e for e in G.edges(data=True) if e[2]["role"] == "required"]
    alt_edges = [e for e in G.edges(data=True) if e[2]["role"] == "alternative"]
    isolates = [n for n in G.nodes() if G.degree(n) == 0]
    leaves = [n for n in G.nodes() if G.in_degree(n) == 0]
    roots = [n for n in G.nodes() if G.out_degree(n) == 0]

    lines = [
        f"Source: {graph_attrs.get('source')}",
        f"Program: {program or '(multi)'}",
        f"Admit term: {term or '(multi)'}",
        "",
        f"Nodes (total):           {G.number_of_nodes()}",
        f"  in catalog:            {len(in_catalog)}",
        f"  external prereqs only: {len(out_catalog)}",
        f"Edges (total):           {G.number_of_edges()}",
        f"  required:              {len(req_edges)}",
        f"  alternative:           {len(alt_edges)}",
        "",
        f"Isolates (no edges):           {len(isolates)}",
        f"Source nodes (no prereqs in):  {len(leaves)}",
        f"Sink nodes (no successors):    {len(roots)}",
        "",
        f"Is DAG?                  {nx.is_directed_acyclic_graph(G)}",
        f"Cycle count (simple):    {len(cycles)}",
    ]
    if cycles:
        lines.append("First cycles:")
        for c in cycles[:10]:
            lines.append("  " + " -> ".join(c) + f" -> {c[0]}")
    if out_catalog:
        lines.append("")
        lines.append("External prereq course codes (sample 20):")
        for code in sorted(out_catalog)[:20]:
            lines.append(f"  {code}")
    if parse_errors:
        lines.append("")
        label = "Strict parse errors (recovered by lenient parser):" if lenient else "Parse errors (should be zero):"
        lines.append(label)
        for code, e in parse_errors[:20]:
            lines.append(f"  {code}: {e}")
    if lenient and unparsed_courses:
        lines.append("")
        lines.append(f"Courses with unparsed prereq spans: {len(unparsed_courses)}")
        for code in unparsed_courses[:20]:
            lines.append(f"  {code}")
    report = "\n".join(lines) + "\n"
    out_report.write_text(report, encoding="utf-8")

    print(report)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_pickle}")
    print(f"Wrote {out_report}")
    return G, out_json


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("term", nargs="?", default="202601",
                    help="term code (ignored when --full is used)")
    ap.add_argument("--program", default="BSCS-DM",
                    help="program code (ignored when --full is used)")
    ap.add_argument("--full", action="store_true",
                    help="build the unified DAG from data/SU_full_catalog.json")
    ap.add_argument("--catalog", default=None,
                    help="override catalog path (implies --full mode)")
    args = ap.parse_args()

    if args.full or args.catalog:
        catalog = Path(args.catalog) if args.catalog else DATA_DIR / "SU_full_catalog.json"
        build(catalog_path=catalog, out_prefix="SU_full", lenient=True, program=None, term=None)
    else:
        build(term=args.term, program=args.program)
