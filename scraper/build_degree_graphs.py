"""
Build per-degree, per-term, per-section prereq DAGs from the unified SU_full_catalog.

For each (degree, term, section) triple we emit a slice of the catalog
containing only courses tagged with that combination in program_sections, plus
the dangling prereq nodes those courses reference.

Layout:
  data/degree_graphs/
    BSCS-DM/
      202201/              <- 2022 Fall cohort requirements
        Required.json
        Required.gpickle
        Required_report.txt
        Core_Elective.json
        ...
      202301/              <- 2023 Fall cohort
        ...
      202601/              <- 2025 Fall cohort (latest)
        ...
    BSEE-DM/
      ...

Run:
  python -m scraper.build_degree_graphs                                # all terms
  python -m scraper.build_degree_graphs --terms 202201 202601          # specific terms
  python -m scraper.build_degree_graphs --sections Required "Core Elective"
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import networkx as nx

from scraper.build_graph import _resolve_tree, edge_roles_for_course
from scraper.prereq_parser import has_unparsed

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
GRAPHS_DIR = DATA_DIR / "degree_graphs"

DEFAULT_SECTIONS: tuple[str, ...] = (
    "Required",
    "Core Elective",
    "Area Elective",
    "Free Elective",
)


def _slug(section: str) -> str:
    """Filesystem-safe section name: 'Core Elective' -> 'Core_Elective'."""
    return section.replace(" ", "_")


def _build_one(
    courses: list[dict],
    program: str,
    section: str,
    lenient: bool,
) -> nx.DiGraph:
    G = nx.DiGraph(program=program, section=section, source="SU_full_slice")

    # Add nodes for the in-slice courses.
    for cr in courses:
        code = cr["code"]
        tree, _ = _resolve_tree(cr.get("prereq_text"), lenient=lenient)
        # Collect every (program, term, section) tag for this course so the
        # node knows which other sections/degrees it also lives in.
        all_tags = cr.get("program_sections", [])
        terms_in_section = sorted({
            t["term"] for t in all_tags
            if t["program"] == program and t["section"] == section
        })
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
            offered_terms=[o["term"] for o in cr.get("offered_terms", [])],
            terms_in_section=terms_in_section,
            in_slice=True,
            in_catalog=True,
            program_sections=all_tags,
        )

    # Add edges; introduce upstream prereq nodes (not in slice) as in_slice=False.
    for cr in courses:
        code = cr["code"]
        tree, _ = _resolve_tree(cr.get("prereq_text"), lenient=lenient)
        edges = edge_roles_for_course(tree)
        for prereq_code, attrs in edges.items():
            if not G.has_node(prereq_code):
                G.add_node(prereq_code, title="", in_slice=False, in_catalog=False)
            G.add_edge(prereq_code, code, **attrs)

    return G


def _write_outputs(G: nx.DiGraph, program: str, term: str, section: str) -> None:
    out_dir = GRAPHS_DIR / program / term
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(section)
    out_json = out_dir / f"{slug}.json"
    out_pickle = out_dir / f"{slug}.gpickle"
    out_report = out_dir / f"{slug}_report.txt"

    out_json.write_text(
        json.dumps(
            {
                "program": program,
                "term": term,
                "section": section,
                "source": "SU_full_slice",
                "nodes": [
                    {
                        "code": n,
                        **{k: v for k, v in attrs.items() if k != "prereq_expr"},
                        "prereq_expr": attrs.get("prereq_expr"),
                    }
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
    in_slice = [n for n, d in G.nodes(data=True) if d.get("in_slice")]
    out_slice = [n for n, d in G.nodes(data=True) if not d.get("in_slice")]
    req_edges = [e for e in G.edges(data=True) if e[2].get("role") == "required"]
    alt_edges = [e for e in G.edges(data=True) if e[2].get("role") == "alternative"]
    isolates = [n for n in G.nodes() if G.degree(n) == 0]
    sources = [n for n in G.nodes() if G.in_degree(n) == 0]
    sinks = [n for n in G.nodes() if G.out_degree(n) == 0]
    is_dag = nx.is_directed_acyclic_graph(G)
    unparsed = [n for n, d in G.nodes(data=True) if has_unparsed(d.get("prereq_expr"))]

    lines = [
        f"Program: {program}",
        f"Term:    {term}",
        f"Section: {section}",
        f"Source:  SU_full_slice",
        "",
        f"Nodes (total):              {G.number_of_nodes()}",
        f"  in this section:          {len(in_slice)}",
        f"  upstream prereqs only:    {len(out_slice)}",
        f"Edges (total):              {G.number_of_edges()}",
        f"  required:                 {len(req_edges)}",
        f"  alternative:              {len(alt_edges)}",
        "",
        f"Isolates (no edges):              {len(isolates)}",
        f"Source nodes (no prereqs in):     {len(sources)}",
        f"Sink nodes (no successors):       {len(sinks)}",
        "",
        f"Is DAG?                     {is_dag}",
        f"Courses with unparsed text: {len(unparsed)}",
    ]
    if unparsed:
        lines.append("Sample (first 10):")
        for c in unparsed[:10]:
            lines.append(f"  {c}")
    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_all(
    catalog_path: Path,
    sections: tuple[str, ...] = DEFAULT_SECTIONS,
    terms: tuple[str, ...] | None = None,
    lenient: bool = True,
    verbose: bool = True,
) -> dict[tuple[str, str, str], int]:
    """Build every (degree, term, section) graph. Returns counts."""
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    all_courses = catalog["courses"]

    # Group courses by (program, term, section); a course can fall into multiple.
    bucket: dict[tuple[str, str, str], list[dict]] = {}
    for cr in all_courses:
        seen_keys: set[tuple[str, str, str]] = set()
        for tag in cr.get("program_sections", []):
            if tag["section"] not in sections:
                continue
            if terms is not None and tag["term"] not in terms:
                continue
            key = (tag["program"], tag["term"], tag["section"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            bucket.setdefault(key, []).append(cr)

    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    counts: dict[tuple[str, str, str], int] = {}
    for (program, term, section), courses in sorted(bucket.items()):
        G = _build_one(courses, program, section, lenient=lenient)
        _write_outputs(G, program, term, section)
        counts[(program, term, section)] = len(courses)
        if verbose:
            print(f"  {program:<10} {term}  {section:<18} -> {len(courses):>4} courses, "
                  f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Derive the Required-section either/or choice groups from the degree-page
    # footnotes now that the Required.json slices exist (the extractor validates
    # each group against them). Keeps requirement_choice_groups.json in sync.
    from scraper.extract_choice_groups import build_choice_groups
    build_choice_groups(verbose=verbose)

    # Summary
    if verbose:
        per_program: Counter = Counter()
        for (program, _term, _sec), n in counts.items():
            per_program[program] += n
        print()
        print("=== summary (course tags per program) ===")
        for program, n in sorted(per_program.items()):
            print(f"  {program:<10} {n}")
        print()
        print(f"wrote {len(counts)} graphs under {GRAPHS_DIR}/")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--catalog",
        default=str(DATA_DIR / "SU_full_catalog.json"),
    )
    ap.add_argument(
        "--sections",
        nargs="+",
        default=list(DEFAULT_SECTIONS),
        help="which sections to slice (default: the four canonical ones)",
    )
    ap.add_argument(
        "--terms",
        nargs="+",
        default=None,
        help="which admit terms to build (default: all terms found in catalog)",
    )
    args = ap.parse_args()
    build_all(
        Path(args.catalog),
        sections=tuple(args.sections),
        terms=tuple(args.terms) if args.terms else None,
    )


if __name__ == "__main__":
    main()
