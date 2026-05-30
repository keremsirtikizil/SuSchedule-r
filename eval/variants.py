"""Named configurations for online agent comparisons.

The retrieval-specific ablations live in ``eval.run_retrieval`` because those
can be measured offline without paying for LLM calls.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentVariant:
    name: str
    mode: str = "react"
    include_required: bool = True
    description: str = ""


VARIANTS: dict[str, AgentVariant] = {
    "react_full": AgentVariant(
        name="react_full",
        mode="react",
        include_required=True,
        description="Production ReAct agent with scoped graphs, RAG prefetch, requirements, and guardrails.",
    ),
    "pipeline": AgentVariant(
        name="pipeline",
        mode="pipeline",
        include_required=True,
        description="Legacy fixed pipeline for a ReAct-vs-pipeline comparison.",
    ),
    "react_no_required_injection": AgentVariant(
        name="react_no_required_injection",
        mode="react",
        include_required=False,
        description="ReAct agent without automatic required-course injection.",
    ),
}


def get_variant(name: str) -> AgentVariant:
    try:
        return VARIANTS[name]
    except KeyError as exc:
        available = ", ".join(sorted(VARIANTS))
        raise ValueError(f"Unknown variant {name!r}. Choose one of: {available}") from exc
