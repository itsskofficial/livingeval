"""Does this suite still represent the traffic?

A suite score is a score on the distribution the suite was drawn from. When that
stops being the distribution you serve, the score keeps being reported and stops
being about anything. Coverage is the quantity that notices.

**The radius is calibrated, not chosen.** "A trace is covered if it is within tau of
some case" needs tau to mean something, and a fixed cosine threshold does not - it is
unitless, corpus-dependent and unfalsifiable. So:

    tau = the q-th percentile of nearest-neighbour distance *within the suite itself*

Read it as: a production trace is covered when it is no further from the suite than
the suite's own members are from each other. That has three properties worth having.
It is scale-free, so it survives a change of representation. It is falsifiable, since
`q` is a stated percentile and every headline number's sensitivity to it is published.
And it degrades in the right direction: a tightly clustered suite earns a small radius
and correctly reports that it covers very little of diverse traffic.

The failure mode to know about: a suite of one case has no internal distances, so tau
is undefined. That returns coverage `None` and status `UNCALIBRATED` rather than a
made-up number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from livingeval.mine.cluster import Clustering
from livingeval.mine.space import Space
from livingeval.stats.tests import Interval, wilson_ci
from livingeval.suite.suite import EvalSuite
from livingeval.trace.types import TraceSet

__all__ = ["CoverageResult", "coverage"]


@dataclass
class CoverageResult:
    """Suite coverage of a trace stream."""

    suite_name: str
    n_cases: int
    n_traces: int
    coverage: float | None
    ci: Interval | None
    radius: float | None
    q: float
    per_cluster: dict[int, float] = field(default_factory=dict)
    cluster_shares: dict[int, float] = field(default_factory=dict)
    cluster_terms: dict[int, str] = field(default_factory=dict)
    sensitivity: dict[float, float] = field(default_factory=dict)
    status: str = "OK"
    representation: dict = field(default_factory=dict)

    @property
    def uncovered(self) -> float | None:
        return None if self.coverage is None else 1.0 - self.coverage

    def summary(self) -> str:  # pragma: no cover - display only
        if self.coverage is None:
            return (
                f"coverage  {self.suite_name}  [{self.status}]\n"
                f"  {self.n_cases} case(s) is too few to calibrate a radius from, so "
                "coverage is undefined rather than guessed."
            )
        lines = [
            f"coverage  {self.suite_name}  [{self.status}]",
            f"  suite n={self.n_cases}   traffic n={self.n_traces}   "
            f"radius=q{self.q:.0f} within-suite NN = {self.radius:.4f}",
            f"  covered   {self.ci}   ({self.uncovered:.1%} of traffic is unrepresented)",
        ]
        if self.sensitivity:
            span = ", ".join(f"q{int(q)}={v:.3f}" for q, v in sorted(self.sensitivity.items()))
            lines.append(f"  sensitivity to the radius percentile: {span}")
        if self.per_cluster:
            lines.append("  per cluster:")
            for c in sorted(self.per_cluster, key=lambda c: -self.cluster_shares.get(c, 0)):
                terms = self.cluster_terms.get(c, "")
                lines.append(
                    f"    cluster {c:<3} share {self.cluster_shares.get(c, 0):>6.1%}  "
                    f"covered {self.per_cluster[c]:>6.1%}   {terms}"
                )
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "kind": "coverage",
            "suite": self.suite_name,
            "status": self.status,
            "n_cases": self.n_cases,
            "n_traces": self.n_traces,
            "coverage": self.coverage,
            "ci": None if self.ci is None else self.ci.as_dict(),
            "radius": self.radius,
            "q": self.q,
            "per_cluster": {str(k): v for k, v in self.per_cluster.items()},
            "cluster_shares": {str(k): v for k, v in self.cluster_shares.items()},
            "cluster_terms": {str(k): v for k, v in self.cluster_terms.items()},
            "sensitivity": {str(k): v for k, v in self.sensitivity.items()},
            **self.representation,
        }


def calibrate_radius(suite_coords: np.ndarray, q: float = 50.0) -> float | None:
    """The q-th percentile of within-suite nearest-neighbour distance."""
    if suite_coords.shape[0] < 2:
        return None
    d = Space.cosine_distance(suite_coords, suite_coords)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    return float(np.percentile(nn, q))


def coverage(
    suite: EvalSuite,
    traces: TraceSet,
    space: Space | None = None,
    clustering: Clustering | None = None,
    q: float = 50.0,
    sensitivity_qs: tuple[float, ...] = (25.0, 50.0, 75.0),
    view: str = "request",
    seed: int = 0,
) -> CoverageResult:
    """Fraction of `traces` represented by `suite`, at a radius calibrated from the
    suite's own geometry.

    Pass a `clustering` (or let one be built) to get the per-cluster breakdown, which
    is the actionable half - a global 0.61 tells you to do something, and the cluster
    table tells you what.
    """
    space = space or Space.fit(traces, view=view, seed=seed)
    if len(suite) == 0:
        return CoverageResult(suite.name, 0, len(traces), None, None, None, q, status="EMPTY_SUITE",
                              representation=space.as_dict())

    suite_coords = space.transform(suite.traces)
    traffic_coords = space.transform(traces)

    radius = calibrate_radius(suite_coords, q)
    if radius is None:
        return CoverageResult(suite.name, len(suite), len(traces), None, None, None, q,
                              status="UNCALIBRATED", representation=space.as_dict())

    nearest = Space.cosine_distance(traffic_coords, suite_coords).min(axis=1)
    covered = nearest <= radius
    point = float(np.mean(covered))
    ci = wilson_ci(int(covered.sum()), len(covered))

    sensitivity: dict[float, float] = {}
    for alt_q in sensitivity_qs:
        alt_r = calibrate_radius(suite_coords, alt_q)
        if alt_r is not None:
            sensitivity[alt_q] = float(np.mean(nearest <= alt_r))

    per_cluster: dict[int, float] = {}
    shares: dict[int, float] = {}
    terms: dict[int, str] = {}
    if clustering is not None:
        labels = clustering.labels
        shares = clustering.shares()
        terms = clustering.describe(traces)
        for c in sorted(set(labels.tolist())):
            rows = labels == c
            per_cluster[int(c)] = float(np.mean(covered[rows])) if rows.any() else 0.0

    return CoverageResult(
        suite_name=suite.name,
        n_cases=len(suite),
        n_traces=len(traces),
        coverage=point,
        ci=ci,
        radius=radius,
        q=q,
        per_cluster=per_cluster,
        cluster_shares=shares,
        cluster_terms=terms,
        sensitivity=sensitivity,
        status="OK",
        representation=space.as_dict(),
    )
