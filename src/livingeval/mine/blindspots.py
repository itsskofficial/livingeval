"""Where the suite has no power at all.

Coverage is one number; this is the list you act on. A blind spot is a cluster that
carries real traffic and has little or no representation in the suite, which means a
regression confined to it cannot move the suite score - not "is harder to see",
cannot move it.

Ranked by `traffic_share * (1 - coverage)`: the share of all production traffic that
this cluster contributes *and* the suite cannot see. That product is the right
ordering because neither factor alone is: a completely uncovered cluster carrying 0.4%
of traffic is not the problem, and a cluster carrying 30% of traffic that is already
well covered is not either.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from livingeval.mine.cluster import Clustering
from livingeval.mine.cluster import cluster as _cluster
from livingeval.mine.coverage import CoverageResult
from livingeval.mine.coverage import coverage as _coverage
from livingeval.mine.space import Space
from livingeval.suite.suite import EvalSuite
from livingeval.trace.types import TraceSet

__all__ = ["BlindSpot", "BlindSpotReport", "blindspots"]


@dataclass
class BlindSpot:
    cluster: int
    n_traces: int
    traffic_share: float
    coverage: float
    n_suite_cases: int
    terms: str
    severity: float
    example_trace_id: str | None = None

    def as_dict(self) -> dict:
        return {
            "cluster": self.cluster,
            "n_traces": self.n_traces,
            "traffic_share": self.traffic_share,
            "coverage": self.coverage,
            "n_suite_cases": self.n_suite_cases,
            "terms": self.terms,
            "severity": self.severity,
            "example_trace_id": self.example_trace_id,
        }


@dataclass
class BlindSpotReport:
    suite_name: str
    spots: list[BlindSpot]
    coverage: CoverageResult
    clustering: Clustering
    threshold: float

    def __iter__(self):
        return iter(self.spots)

    def __getitem__(self, i):
        return self.spots[i]

    def __len__(self) -> int:
        return len(self.spots)

    @property
    def blind_clusters(self) -> list[int]:
        """Clusters below the coverage threshold - the ones that make a gate BLIND."""
        return [s.cluster for s in self.spots if s.coverage < self.threshold]

    def summary(self) -> str:  # pragma: no cover - display only
        if not self.spots:
            return f"blind spots  {self.suite_name}\n  none: every cluster is covered."
        lines = [
            f"blind spots  {self.suite_name}   "
            f"(overall coverage {self.coverage.coverage:.1%}, threshold {self.threshold:.0%})",
            f"{'cluster':<9}{'traffic':>9}{'covered':>9}{'cases':>7}{'severity':>10}  terms",
        ]
        for s in self.spots:
            lines.append(
                f"{s.cluster:<9}{s.traffic_share:>8.1%}{s.coverage:>9.1%}{s.n_suite_cases:>7}"
                f"{s.severity:>10.3f}  {s.terms}"
            )
        worst = self.spots[0]
        lines.append("")
        lines.append(
            f"  cluster {worst.cluster} carries {worst.traffic_share:.1%} of traffic with "
            f"{worst.n_suite_cases} suite case(s). A regression confined to it cannot move "
            f"the suite score."
        )
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "kind": "blindspots",
            "suite": self.suite_name,
            "threshold": self.threshold,
            "blind_clusters": self.blind_clusters,
            "spots": [s.as_dict() for s in self.spots],
        }


def blindspots(
    suite: EvalSuite,
    traces: TraceSet,
    top: int | None = None,
    threshold: float = 0.30,
    space: Space | None = None,
    clustering: Clustering | None = None,
    q: float = 50.0,
    seed: int = 0,
    view: str = "request",
) -> BlindSpotReport:
    """Rank traffic clusters by how much of your traffic the suite cannot see."""
    space = space or Space.fit(traces, view=view, seed=seed)
    clustering = clustering or _cluster(traces, space=space, seed=seed)
    cov = _coverage(suite, traces, space=space, clustering=clustering, q=q)

    # Which cluster does each suite case sit in? Assigning cases to the traffic's
    # clusters (rather than clustering the suite separately) is what makes "this
    # cluster has two cases" a statement about the same partition as the coverage row.
    suite_cluster_counts: dict[int, int] = dict.fromkeys(clustering.sizes, 0)
    if len(suite) and clustering.centroids is not None:
        suite_coords = space.transform(suite.traces)
        assign = np.argmin(Space.cosine_distance(suite_coords, clustering.centroids), axis=1)
        keys = sorted(clustering.sizes)
        for a in assign:
            suite_cluster_counts[keys[int(a)]] += 1

    shares = clustering.shares()
    terms = clustering.describe(traces)
    spots: list[BlindSpot] = []
    for c in sorted(clustering.sizes):
        cvg = cov.per_cluster.get(c, 0.0)
        share = shares.get(c, 0.0)
        rows = np.flatnonzero(clustering.labels == c)
        spots.append(
            BlindSpot(
                cluster=int(c),
                n_traces=int(clustering.sizes[c]),
                traffic_share=share,
                coverage=cvg,
                n_suite_cases=suite_cluster_counts.get(c, 0),
                terms=terms.get(c, ""),
                severity=float(share * (1.0 - cvg)),
                example_trace_id=traces[int(rows[0])].trace_id if rows.size else None,
            )
        )

    spots.sort(key=lambda s: -s.severity)
    if top is not None:
        spots = spots[:top]
    return BlindSpotReport(suite.name, spots, cov, clustering, threshold)
