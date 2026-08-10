"""What regression are you asking the suite to catch?

Power is not a property of a suite. It is a property of a suite *against a specific
alternative*, and a tool that reports "your power is 0.4" without naming the
alternative has reported nothing. So the alternative is an object you construct and
that goes into the record.

A `Regression` is a predicate over cases plus an effect size: among the cases the
predicate selects, a currently-passing case fails with probability `effect`.

Three constructors cover the cases worth asking about:

- `uniform(effect)` - quality drops everywhere. The easy case, and the one every
  eval suite handles.
- `on_meta(key, value, effect)` - the drop is confined to traces whose metadata
  matches, which is what a regression in one intent, one language or one tool
  actually looks like.
- `on_cluster(cluster_ids, effect, assignment)` - the drop is confined to clusters
  of the current traffic. This is the one that exposes staleness, because a frozen
  suite has no cases in the cluster that appeared last month.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from livingeval.suite.case import EvalCase

__all__ = ["Regression", "on_cluster", "on_meta", "on_tag", "uniform"]


@dataclass
class Regression:
    """A named alternative hypothesis."""

    name: str
    predicate: Callable[[EvalCase], bool]
    effect: float
    description: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.effect <= 1.0:
            raise ValueError(f"effect must be a probability in [0, 1], got {self.effect}")

    def affected(self, cases: Iterable[EvalCase]) -> np.ndarray:
        return np.asarray([bool(self.predicate(c)) for c in cases], dtype=bool)

    def as_dict(self) -> dict:
        return {"name": self.name, "effect": self.effect, "description": self.description}


def uniform(effect: float = 0.30) -> Regression:
    """Quality drops across the board."""
    return Regression(
        name=f"uniform(effect={effect:g})",
        predicate=lambda case: True,
        effect=effect,
        description="every case is affected",
    )


def on_meta(key: str, value, effect: float = 0.30) -> Regression:
    """The drop is confined to traces whose `meta[key] == value`."""
    return Regression(
        name=f"{key}={value}(effect={effect:g})",
        predicate=lambda case: case.trace.meta.get(key) == value,
        effect=effect,
        description=f"only traces with meta[{key!r}] == {value!r}",
    )


def on_tag(tag: str, effect: float = 0.30) -> Regression:
    """The drop is confined to cases carrying a tag."""
    return Regression(
        name=f"tag:{tag}(effect={effect:g})",
        predicate=lambda case: tag in case.tags,
        effect=effect,
        description=f"only cases tagged {tag!r}",
    )


def on_cluster(
    clusters: Iterable[int], assignment: dict[str, int], effect: float = 0.30
) -> Regression:
    """The drop is confined to named clusters of the current traffic.

    `assignment` maps `case_id -> cluster`, produced by
    `power.assign_cases_to_clusters`. Cases in no cluster are unaffected, which is
    the honest reading: a suite case that sits nowhere near current traffic is not
    going to see a regression in current traffic.
    """
    wanted = set(int(c) for c in clusters)
    return Regression(
        name=f"clusters{sorted(wanted)}(effect={effect:g})",
        predicate=lambda case: assignment.get(case.case_id, -1) in wanted,
        effect=effect,
        description=f"only cases assigned to clusters {sorted(wanted)}",
    )
