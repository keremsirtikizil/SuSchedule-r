"""Run deterministic regression checks without an API key.

Usage:
    python -m eval.run_offline
    python -m eval.run_offline --json eval/reports/offline.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from eval.common import REPORTS_DIR, ROOT, write_json
from scheduler.catalog import Catalog, corequisite_codes
from scheduler.graph_selector import cohort_for_admit, normalize_program_code, select_graphs
from scheduler.offerings import likely_offered_in, load_offerings
from scheduler.requirements import compute_remaining
from scheduler.schemas import PlannerRequest
from scheduler.session import PlannerSession
from scheduler.timetable import meetings_overlap
from scraper.prereq_parser import expr_satisfied, parse_prereq


@dataclass
class Check:
    name: str
    ok: bool
    details: Any


def _check(name: str, fn: Callable[[], Any]) -> Check:
    try:
        details = fn()
        if details is False:
            raise AssertionError("returned False")
        return Check(name=name, ok=True, details=details if details is not None else "ok")
    except Exception as exc:
        return Check(name=name, ok=False, details=f"{type(exc).__name__}: {exc}")


def _expect(condition: bool, details: Any) -> Any:
    if not condition:
        raise AssertionError(details)
    return details


def _profile() -> dict:
    return json.loads((ROOT / "data" / "transcript_cagan.json").read_text(encoding="utf-8"))


def _degree_eval_session() -> PlannerSession:
    return PlannerSession.from_request(PlannerRequest(
        target_term="202601",
        user_request="",
        transcript_html=ROOT / "DEGREE EVALUATION.html",
        mode="react",
    ))


def run() -> dict:
    profile = _profile()
    catalog = Catalog.load()
    offerings = load_offerings()

    def aliases() -> dict:
        got = {
            "CS": normalize_program_code("CS"),
            "Industrial Engineering": normalize_program_code("Industrial Engineering"),
            "EE": normalize_program_code("EE"),
        }
        return _expect(got == {
            "CS": "BSCS-DM",
            "Industrial Engineering": "BSIE-DM",
            "EE": "BSEE-DM",
        }, got)

    def cohorts() -> dict:
        got = {
            "spring_maps_to_fall": cohort_for_admit("BSCS-DM", "202202"),
            "exact": cohort_for_admit("BSCS-DM", "202501"),
            "older_available": cohort_for_admit("BSCS-DM", "202101"),
        }
        return _expect(got == {
            "spring_maps_to_fall": "202201",
            "exact": "202501",
            "older_available": "202201",
        }, got)

    def scoped_graph() -> dict:
        selected = select_graphs("CS", "202202")
        counts = selected.to_dict()
        _expect(counts["program"] == "BSCS-DM", counts)
        _expect(counts["cohort_term"] == "202201", counts)
        _expect(set(counts["sections"]) == {"Required", "Core Elective", "Area Elective", "Free Elective"}, counts)
        _expect(counts["section_counts"]["Required"] > 0, counts)
        _expect(counts["upstream_only_counts"]["Required"] > 0, counts)
        return counts

    def html_profile_autoload() -> dict:
        session = _degree_eval_session()
        _expect(session.transcript_loaded, "Degree Evaluation HTML did not auto-load")
        _expect(session.student is not None, "student missing")
        details = {
            "program": session.student.program,
            "admit_term": session.student.admit_term,
            "completed_count": len(session.student.completed),
            "in_progress": sorted(session.student.in_progress),
            "section_requirement_keys": sorted(session.section_requirements),
        }
        _expect(details["program"] == "BSCS-DM", details)
        _expect(details["admit_term"] == "202201", details)
        _expect("ENGINEERING" in session.section_requirements, details)
        _expect("BASIC SCIENCE" in session.section_requirements, details)
        return details

    def remaining_requirements() -> dict:
        report = compute_remaining(
            profile["program"],
            set(profile["completed_for_eligibility"]),
            set(profile["in_progress"]),
            profile["admit_term"],
        )
        all_pools = report.required_left + report.core_left + report.area_left + report.free_left
        details = report.to_dict()
        details["duplicate_count_across_sections"] = len(all_pools) - len(set(all_pools))
        _expect(report.cohort_term == "202201", details)
        _expect(details["duplicate_count_across_sections"] == 0, details)
        _expect("ENS 492" in report.required_left, details)
        return details

    def prereq_boolean_tree() -> dict:
        text = (
            "(MATH 201 - Undergraduate - Min Grade D or "
            "MATH 212 - Undergraduate - Min Grade D) and "
            "MATH 203 - Undergraduate - Min Grade D"
        )
        expr = parse_prereq(text)
        details = {
            "expr": expr,
            "math_201_and_203": expr_satisfied(expr, {"MATH 201", "MATH 203"}),
            "math_212_and_203": expr_satisfied(expr, {"MATH 212", "MATH 203"}),
            "missing_second_condition": expr_satisfied(expr, {"MATH 201"}),
        }
        _expect(details["math_201_and_203"], details)
        _expect(details["math_212_and_203"], details)
        _expect(not details["missing_second_condition"], details)
        return details

    def offering_pattern() -> dict:
        likely = likely_offered_in(offerings, "202601", same_season_only=True)
        details = {
            "likely_fall_count": len(likely),
            "CS 412": "CS 412" in likely,
            "CS 408": "CS 408" in likely,
        }
        _expect(details["likely_fall_count"] > 0, details)
        _expect(details["CS 412"], details)
        return details

    def timetable_overlap() -> dict:
        a = {"days": [0, 2], "start": 600, "end": 690}
        b = {"days": [2], "start": 660, "end": 720}
        c = {"days": [1], "start": 660, "end": 720}
        details = {
            "same_day_overlap": meetings_overlap(a, b),
            "different_day_no_overlap": meetings_overlap(a, c),
        }
        _expect(details["same_day_overlap"], details)
        _expect(not details["different_day_no_overlap"], details)
        return details

    def corequisites() -> dict:
        details = {
            "CS 412": corequisite_codes(catalog, "CS 412"),
            "CS 308": corequisite_codes(catalog, "CS 308"),
            "CS 302": corequisite_codes(catalog, "CS 302"),
        }
        _expect("CS 412R" in details["CS 412"], details)
        _expect("CS 308L" in details["CS 308"], details)
        _expect("CS 302R" in details["CS 302"], details)
        return details

    def lexical_scope() -> dict:
        selected = select_graphs("BSCS-DM", "202201")
        pool = selected.candidate_courses("Core Elective")
        rows = catalog.search("network security", k=8, candidate_pool=pool, subj=["CS"])
        codes = [row["code"] for row in rows]
        details = {"pool_size": len(pool), "results": codes}
        _expect(bool(codes), details)
        _expect(all(code in pool for code in codes), details)
        _expect("CS 432" in codes, details)
        return details

    def rag_artifact_alignment() -> dict:
        embeddings_dir = ROOT / "embeddings"
        parquet = embeddings_dir / "su_courses.parquet"
        matrix = embeddings_dir / "su_courses_embeddings.npy"
        id_map_path = embeddings_dir / "id_map.json"
        index = embeddings_dir / "su_courses.index"
        missing = [str(path.relative_to(ROOT)) for path in (parquet, matrix, id_map_path, index) if not path.exists()]
        if missing:
            return {
                "skipped": True,
                "reason": "generated RAG artifacts are not built locally",
                "missing": missing,
                "rebuild": "python -m scheduler.build_faiss_index",
            }
        import numpy as np
        import pandas as pd

        df = pd.read_parquet(parquet)
        embeddings = np.load(matrix)
        id_map = {int(k): v for k, v in json.loads(id_map_path.read_text(encoding="utf-8")).items()}
        expected_codes = df["code"].astype(str).tolist()
        mapped_codes = [id_map.get(i) for i in range(len(expected_codes))]
        details = {
            "metadata_rows": len(df),
            "embedding_rows": int(embeddings.shape[0]),
            "id_map_rows": len(id_map),
            "row_order_matches": mapped_codes == expected_codes,
        }
        _expect(len(df) == embeddings.shape[0] == len(id_map), details)
        _expect(details["row_order_matches"], details)
        return details

    checks = [
        _check("program aliases normalize to graph directory names", aliases),
        _check("admit terms map to the correct cohort graph", cohorts),
        _check("scoped degree graphs load with in-slice and upstream nodes", scoped_graph),
        _check("Degree Evaluation HTML auto-loads into PlannerSession", html_profile_autoload),
        _check("remaining requirements de-duplicate overlapping section pools", remaining_requirements),
        _check("AND/OR prerequisite expression trees evaluate correctly", prereq_boolean_tree),
        _check("same-season offering heuristic returns known Fall courses", offering_pattern),
        _check("meeting overlap checker detects only real overlaps", timetable_overlap),
        _check("catalog corequisite lookup finds recitations and labs", corequisites),
        _check("lexical fallback retrieval respects scoped graph filters", lexical_scope),
        _check("local RAG artifacts align with current catalog metadata", rag_artifact_alignment),
    ]
    passed = sum(check.ok for check in checks)
    return {
        "suite": "offline_deterministic",
        "passed": passed,
        "failed": len(checks) - passed,
        "total": len(checks),
        "checks": [asdict(check) for check in checks],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json",
        default=str(REPORTS_DIR / "offline.json"),
        help="Output report path (default: eval/reports/offline.json)",
    )
    args = ap.parse_args()

    report = run()
    write_json(args.json, report)
    for check in report["checks"]:
        mark = "PASS" if check["ok"] else "FAIL"
        print(f"[{mark}] {check['name']}")
        if not check["ok"]:
            print(f"       {check['details']}")
    print(f"\nOffline deterministic suite: {report['passed']}/{report['total']} passed")
    print(f"Report: {Path(args.json)}")
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
