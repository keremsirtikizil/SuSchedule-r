"""Scoped degree/cohort graph selection.

Normal advising should not use the global prerequisite graph.  This module
loads only the degree requirement graphs for a student's program/cohort and
keeps requirement sections separate for targeted retrieval and validation.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
DEGREE_GRAPHS_DIR = ROOT / "data" / "degree_graphs"

DEFAULT_SECTIONS: tuple[str, ...] = (
    "Required",
    "Core Elective",
    "Area Elective",
    "Free Elective",
)

PROGRAM_ALIASES: dict[str, str] = {
    "CS": "BSCS-DM",
    "CSE": "BSCS-DM",
    "COMPUTER SCIENCE": "BSCS-DM",
    "COMPUTER SCIENCE AND ENGINEERING": "BSCS-DM",
    "IE": "BSIE-DM",
    "INDUSTRIAL ENGINEERING": "BSIE-DM",
    "EE": "BSEE-DM",
    "EEE": "BSEE-DM",
    "ELECTRONICS ENGINEERING": "BSEE-DM",
    "ELECTRICAL ENGINEERING": "BSEE-DM",
    "DSA": "BSDSA-DM",
    "DATA SCIENCE": "BSDSA-DM",
    "DATA SCIENCE AND ANALYTICS": "BSDSA-DM",
    "ME": "BSME-DM",
    "MECHATRONICS": "BSME-DM",
    "MECHATRONICS ENGINEERING": "BSME-DM",
    "MAT": "BSMAT-DM",
    "MATERIALS": "BSMAT-DM",
    "MATERIALS SCIENCE": "BSMAT-DM",
    "BIO": "BSBIO-DM",
    "MOLECULAR BIOLOGY": "BSBIO-DM",
    "MANAGEMENT": "BAMAN-DM",
    "BUSINESS": "BAMAN-DM",
    "ECON": "BAECON-DM",
    "ECONOMICS": "BAECON-DM",
    "POLS": "BAPOLS-DM",
    "POLITICAL SCIENCE": "BAPOLS-DM",
    "PSIR": "BAPSIR-DM",
    "POLITICAL SCIENCE AND INTERNATIONAL RELATIONS": "BAPSIR-DM",
    "PSY": "BAPSY-DM",
    "PSYCH": "BAPSY-DM",
    "PSYCHOLOGY": "BAPSY-DM",
    "VACD": "BAVACD-DM",
    "VISUAL ARTS": "BAVACD-DM",
    "VISUAL ARTS AND VISUAL COMMUNICATIONS DESIGN": "BAVACD-DM",
    "IS": "BAIS-DM",
    "INTERNATIONAL STUDIES": "BAIS-DM",
}


def normalize_program_code(program: str | None) -> str | None:
    if not program:
        return None
    raw = str(program).strip()
    if not raw:
        return None
    key = raw.upper().replace("_", "-")
    if key in PROGRAM_ALIASES:
        return PROGRAM_ALIASES[key]
    if key.endswith("-DM"):
        return key
    if f"{key}-DM" in PROGRAM_ALIASES.values():
        return f"{key}-DM"
    if key.startswith(("BS", "BA")):
        return key
    return PROGRAM_ALIASES.get(key, key)


def section_slug(section: str) -> str:
    return section.replace(" ", "_")


def available_cohort_terms(program: str, data_dir: Path = DEGREE_GRAPHS_DIR) -> list[str]:
    program = normalize_program_code(program) or program
    prog_dir = data_dir / program
    if not prog_dir.exists():
        return []
    return sorted(p.name for p in prog_dir.iterdir() if p.is_dir() and p.name.isdigit())


def cohort_for_admit(
    program: str,
    admit_term: str | None,
    data_dir: Path = DEGREE_GRAPHS_DIR,
) -> str:
    """Map an admit term to the closest available fall-like cohort graph."""
    program = normalize_program_code(program) or program
    cohorts = available_cohort_terms(program, data_dir)
    if not cohorts:
        raise FileNotFoundError(f"No degree graphs found for program {program}.")

    term = str(admit_term or cohorts[-1])
    if term.endswith("02") or term.endswith("03"):
        term = term[:4] + "01"

    best = cohorts[0]
    for cohort in cohorts:
        if cohort <= term:
            best = cohort
    return best


@dataclass
class SelectedGraphs:
    program: str
    cohort_term: str
    sections: tuple[str, ...]
    graphs: dict[str, nx.DiGraph] = field(default_factory=dict)
    section_courses: dict[str, set[str]] = field(default_factory=dict)
    upstream_only: dict[str, set[str]] = field(default_factory=dict)
    merged: nx.DiGraph = field(default_factory=nx.DiGraph)

    def all_requirement_courses(self) -> set[str]:
        out: set[str] = set()
        for courses in self.section_courses.values():
            out.update(courses)
        return out

    def candidate_courses(self, section: str | None = None) -> set[str]:
        if section:
            return set(self.section_courses.get(section, set()))
        return self.all_requirement_courses()

    def to_dict(self) -> dict:
        return {
            "program": self.program,
            "cohort_term": self.cohort_term,
            "sections": list(self.sections),
            "section_counts": {
                section: len(courses)
                for section, courses in self.section_courses.items()
            },
            "upstream_only_counts": {
                section: len(courses)
                for section, courses in self.upstream_only.items()
            },
            "merged_nodes": self.merged.number_of_nodes(),
            "merged_edges": self.merged.number_of_edges(),
        }


def _load_section_json(path: Path) -> tuple[set[str], set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    in_slice: set[str] = set()
    upstream: set[str] = set()
    for node in data.get("nodes", []):
        code = node.get("code")
        if not code:
            continue
        if node.get("in_slice"):
            in_slice.add(code)
        else:
            upstream.add(code)
    return in_slice, upstream


def _merge_graphs(graphs: dict[str, nx.DiGraph]) -> nx.DiGraph:
    merged = nx.DiGraph()
    for section, graph in graphs.items():
        for node, attrs in graph.nodes(data=True):
            attrs = dict(attrs)
            sections = set(attrs.get("degree_sections", []))
            if attrs.get("in_slice"):
                sections.add(section)
            attrs["degree_sections"] = sorted(sections)
            if not merged.has_node(node):
                merged.add_node(node, **attrs)
            else:
                existing = merged.nodes[node]
                if attrs.get("in_slice"):
                    existing["in_slice"] = True
                existing_sections = set(existing.get("degree_sections", []))
                existing_sections.update(sections)
                existing["degree_sections"] = sorted(existing_sections)
                for key, value in attrs.items():
                    if key not in existing or existing.get(key) in (None, "", []):
                        existing[key] = value
        for u, v, attrs in graph.edges(data=True):
            if not merged.has_edge(u, v):
                merged.add_edge(u, v, **attrs)
    return merged


def select_graphs(
    program: str,
    admit_term: str | None,
    sections: tuple[str, ...] = DEFAULT_SECTIONS,
    data_dir: Path = DEGREE_GRAPHS_DIR,
) -> SelectedGraphs:
    program = normalize_program_code(program) or program
    cohort = cohort_for_admit(program, admit_term, data_dir)
    base = data_dir / program / cohort

    graphs: dict[str, nx.DiGraph] = {}
    section_courses: dict[str, set[str]] = {}
    upstream_only: dict[str, set[str]] = {}

    for section in sections:
        slug = section_slug(section)
        pickle_path = base / f"{slug}.gpickle"
        json_path = base / f"{slug}.json"
        if not pickle_path.exists() or not json_path.exists():
            continue
        with pickle_path.open("rb") as f:
            graphs[section] = pickle.load(f)
        in_slice, upstream = _load_section_json(json_path)
        section_courses[section] = in_slice
        upstream_only[section] = upstream

    if not graphs:
        raise FileNotFoundError(f"No section graphs found under {base}.")

    return SelectedGraphs(
        program=program,
        cohort_term=cohort,
        sections=tuple(graphs.keys()),
        graphs=graphs,
        section_courses=section_courses,
        upstream_only=upstream_only,
        merged=_merge_graphs(graphs),
    )
