"""Accepted course equivalences / substitutions.

Some degree audits accept a *combination* of courses in place of a single
required course. The degree-graph requirement engine only knows the literal
required code, so without this table it keeps reporting the required course as
"still needed" — and over-counts the remaining required credits — even when the
student already completed an accepted substitute.

Each rule says: completing ALL of ``requires`` grants credit for ``grants``.

Seeded with the one substitution surfaced by the requirements evaluation
(``eval/run_requirements.py``), where the engine reported MATH 212 as still
needed while the official Banner audit had accepted MATH 201 + MATH 202 in its
place. Add further rules here as registrar-accepted substitutions are confirmed;
this is intentionally a small, auditable allow-list rather than a guess.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Equivalence:
    """Completing every code in ``requires`` grants credit for ``grants``."""

    grants: str
    requires: tuple[str, ...]


# Registrar-accepted substitutions. Keep codes in canonical "SUBJ NUM" form.
COURSE_EQUIVALENCES: tuple[Equivalence, ...] = (
    Equivalence(grants="MATH 212", requires=("MATH 201", "MATH 202")),
)


def _norm(codes) -> set[str]:
    return {c.upper().strip() for c in codes}


def granted_codes(completed) -> set[str]:
    """Return the codes a student is *granted* purely via equivalences.

    Resolved to a fixed point so chained substitutions (a grant that in turn
    completes another rule) settle correctly.
    """
    done = _norm(completed)
    granted: set[str] = set()
    changed = True
    while changed:
        changed = False
        pool = done | granted
        for eq in COURSE_EQUIVALENCES:
            if eq.grants in pool:
                continue
            if all(req in pool for req in eq.requires):
                granted.add(eq.grants)
                changed = True
    return granted


def apply_equivalences(completed) -> set[str]:
    """Return ``completed`` (normalised) augmented with equivalence-granted codes."""
    done = _norm(completed)
    return done | granted_codes(done)
