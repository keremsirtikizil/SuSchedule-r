"""Lightweight catalog lookup and lexical fallback retrieval."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG_PATH = ROOT / "data" / "SU_full_catalog.json"


@dataclass
class CatalogCourse:
    code: str
    title: str
    description: str
    subj: str
    num: str
    su_credit: float | None
    prereq_text: str
    coreq_text: str
    offered_terms: list
    program_sections: list

    def to_result(self, rank: int, score: float) -> dict:
        return {
            "rank": rank,
            "score": round(score, 4),
            "code": self.code,
            "title": self.title,
            "credits": self.su_credit,
            "description": (self.description or "")[:500],
            "prereq_text": self.prereq_text,
            "coreq_text": self.coreq_text,
            "offered_terms": self.offered_terms[:8],
            "program_sections": self.program_sections[:12],
        }


class Catalog:
    def __init__(self, courses: list[CatalogCourse]) -> None:
        self.courses = courses
        self.by_code = {c.code: c for c in courses}

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CATALOG_PATH) -> "Catalog":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        items = raw.get("courses", raw) if isinstance(raw, dict) else raw
        courses: list[CatalogCourse] = []
        for item in items:
            su_credit = item.get("su_credit")
            try:
                su_credit = float(su_credit) if su_credit is not None else None
            except (TypeError, ValueError):
                su_credit = None
            courses.append(
                CatalogCourse(
                    code=item.get("code", ""),
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    subj=item.get("subj", ""),
                    num=str(item.get("num", "")),
                    su_credit=su_credit,
                    prereq_text=item.get("prereq_text", ""),
                    coreq_text=item.get("coreq_text", ""),
                    offered_terms=item.get("offered_terms", []),
                    program_sections=item.get("program_sections", []),
                )
            )
        return cls([c for c in courses if c.code])

    def get(self, code: str) -> CatalogCourse | None:
        return self.by_code.get(code.upper().strip())

    def search(
        self,
        query: str,
        k: int = 8,
        candidate_pool: set[str] | None = None,
        subj: list[str] | None = None,
    ) -> list[dict]:
        terms = _tokens(query)
        if not terms:
            return []
        subj_set = {s.upper() for s in subj or []}
        inferred_subjects = _infer_subject_boosts(query)
        scored: list[tuple[float, CatalogCourse]] = []
        for course in self.courses:
            if candidate_pool is not None and course.code not in candidate_pool:
                continue
            if subj_set and course.subj.upper() not in subj_set:
                continue
            text = _tokens(
                f"{course.code} {course.title} {course.description} {course.prereq_text}"
            )
            if not text:
                continue
            title_tokens = _tokens(course.title)
            text_set = set(text)
            score = sum(2.0 if t in title_tokens else 1.0 for t in terms if t in text_set)
            phrase_bonus = 2.0 if query.lower() in f"{course.title} {course.description}".lower() else 0.0
            subj_bonus = 2.5 if inferred_subjects and course.subj.upper() in inferred_subjects else 0.0
            title_bonus = _title_bonus(query, course.title)
            total = score + phrase_bonus + subj_bonus + title_bonus
            if total:
                scored.append((total, course))
        scored.sort(key=lambda x: (-x[0], x[1].code))
        return [course.to_result(i + 1, score) for i, (score, course) in enumerate(scored[:k])]


def _tokens(text: str) -> list[str]:
    return [
        tok
        for tok in re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
        if len(tok) > 1
    ]


def _title_bonus(query: str, title: str) -> float:
    low_q = query.lower()
    low_t = title.lower()
    bonus = 0.0
    if any(term in low_q for term in ("optimization", "operations", "operational", "supply chain", "logistics")):
        for phrase in (
            "operations research",
            "optimization",
            "decision analysis",
            "simulation",
            "stochastic",
            "production and service systems",
            "supply chain",
            "logistics",
            "quality planning",
            "project scheduling",
            "manufacturing",
        ):
            if phrase in low_t:
                bonus += 2.0
                break
    return bonus


def _infer_subject_boosts(query: str) -> set[str]:
    low = query.lower()
    boosts: set[str] = set()
    if any(term in low for term in (
        "industrial engineering", "operations research", "optimization",
        "operations", "operational", "supply chain", "logistics",
        "production", "inventory",
    )):
        boosts.update({"IE", "ENS", "OPIM"})
    if any(term in low for term in ("computer science", "algorithms", "theoretical", "llm", "machine learning")):
        boosts.update({"CS", "DSA", "MATH"})
    if any(term in low for term in ("network", "communication", "wireless")):
        boosts.update({"CS", "EE"})
    return boosts
