"""
Term-offering filter.

Reads data/offerings.json (produced by ``scraper.scrape_offerings``) and
exposes helpers for asking:

  * Which courses are likely to be offered in a given target season?
  * Was course X offered in any recent term of season S?

The proposal's filter rule:

    Keep a course if it was offered in its usual season at least once in the
    last 4 terms (= 2 academic years for a given season). Drop otherwise.

We default to checking only ``Fall`` or ``Spring`` matches because Summer
historically has a very narrow catalog (≈9 CS courses per summer in our data).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OFFERINGS = ROOT / "data" / "offerings.json"

# Season suffix in our 6-digit term codes.
SEASON_SUFFIX = {"Fall": "01", "Spring": "02", "Summer": "03"}
SUFFIX_SEASON = {v: k for k, v in SEASON_SUFFIX.items()}


def load_offerings(path: str | Path = DEFAULT_OFFERINGS) -> dict[str, dict[str, list[dict]]]:
    """Return the on-disk offerings map: ``{term_code: {course_code: [sections...]}}``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run scraper/scrape_offerings.py first."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def season_of(term_code: str) -> str:
    return SUFFIX_SEASON[term_code[-2:]]


def terms_of_season(
    offerings: dict[str, dict],
    season: str,
    include: Iterable[str] | None = None,
) -> list[str]:
    """Return every scraped term code matching ``season`` (newest first)."""
    if season not in SEASON_SUFFIX:
        raise ValueError(f"Unknown season: {season!r}")
    suffix = SEASON_SUFFIX[season]
    terms = sorted((t for t in offerings if t.endswith(suffix)), reverse=True)
    if include is not None:
        allowed = set(include)
        terms = [t for t in terms if t in allowed]
    return terms


def likely_offered_in(
    offerings: dict[str, dict],
    target_term: str,
    same_season_only: bool = True,
) -> set[str]:
    """Return the set of course codes likely to be offered in ``target_term``.

    Heuristic: a course is "likely" if it was offered in the same season at
    least once within the scraped window. When ``same_season_only`` is False
    we accept any term in the window (useful for very small Summer terms).
    """
    season = season_of(target_term)
    if same_season_only:
        terms = terms_of_season(offerings, season)
    else:
        terms = sorted(offerings.keys(), reverse=True)
    likely: set[str] = set()
    for t in terms:
        likely.update(offerings[t].keys())
    return likely


def offered_history(
    offerings: dict[str, dict],
    course_code: str,
) -> dict[str, int]:
    """Return ``{term_code: section_count}`` for every term in which the course ran."""
    out: dict[str, int] = {}
    for term, codes in offerings.items():
        sections = codes.get(course_code)
        if sections:
            out[term] = len(sections)
    return dict(sorted(out.items(), reverse=True))


# --------------------------------------------------------------------------- #
# CLI: quick inspection
# --------------------------------------------------------------------------- #


def _cli() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="202601",
                    help="Target term code (default 202601 = Fall 2026-2027)")
    ap.add_argument("--any-season", action="store_true",
                    help="Don't restrict to same-season terms")
    ap.add_argument("--course", help="Show offered history for one course")
    args = ap.parse_args()

    off = load_offerings()

    if args.course:
        hist = offered_history(off, args.course)
        print(f"{args.course} offered in:")
        for t, n in hist.items():
            print(f"  {t} ({SUFFIX_SEASON[t[-2:]]}): {n} sections")
        return

    likely = likely_offered_in(off, args.target, same_season_only=not args.any_season)
    season = season_of(args.target)
    print(f"Target term {args.target} (season={season})")
    print(f"Window terms considered: {terms_of_season(off, season) if not args.any_season else sorted(off, reverse=True)}")
    print(f"Likely-offered courses: {len(likely)}")
    cs = sorted(c for c in likely if c.startswith('CS '))
    print(f"  CS subset ({len(cs)}): {cs}")


if __name__ == "__main__":
    _cli()
