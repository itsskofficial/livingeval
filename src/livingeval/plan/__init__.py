"""Deciding which evals a codebase needs.

`taxonomy` is the catalogue of what can be measured; `planner` turns a scanned
inventory into a concrete set of pipelines. The split matters: the catalogue is
reviewable domain knowledge, the planner is mechanical.
"""

from livingeval.plan.planner import Pipeline, Plan, Question, build_plan
from livingeval.plan.taxonomy import (
    CATALOGUE,
    Archetype,
    Level,
    Mechanism,
    Method,
    Metric,
    Reference,
    Risk,
    for_target,
    lookup,
)

__all__ = [
    "CATALOGUE",
    "Archetype",
    "Level",
    "Mechanism",
    "Method",
    "Metric",
    "Pipeline",
    "Plan",
    "Question",
    "Reference",
    "Risk",
    "build_plan",
    "for_target",
    "lookup",
]
