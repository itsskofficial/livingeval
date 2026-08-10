"""Staleness, as a curve.

This is the measurement the library is named for. Walk a trace stream forward in time
window by window. In each window, ask the same question of the same frozen suite:

> If the system regressed on the part of traffic that is currently growing, would this
> suite fire?

A suite that was assembled before that part of traffic existed has no cases there, so
the answer stops being yes. Nothing in a dashboard changes at the moment this happens.
The suite keeps returning a score, CI keeps going green, and the score stops being
about the traffic being served.

Run it again with a suite that is refreshed from each window's traffic and you get the
comparison that makes the point: same generator, same gate, same regression, two
curves.

The `crossover` is the reportable number - the first window in which the frozen
suite's detection power falls below one half. Before it the gate is a measurement;
after it the gate is a coin flip that nobody has told you about.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from livingeval.gate.run import SuiteRun, run_suite
from livingeval.judge.base import Judge
from livingeval.power.estimate import PowerResult, estimate
from livingeval.power.regression import Regression
from livingeval.suite.suite import EvalSuite
from livingeval.trace.types import TraceSet

__all__ = ["DecayCurve", "DecayPoint", "decay", "staleness_curve"]


@dataclass
class DecayPoint:
    window: int
    t_start: float
    t_end: float
    n_traces: int
    coverage: float | None
    power: float
    power_lo: float
    power_hi: float
    n_affected_cases: int
    affected_traffic_share: float
    #: Set by `staleness_curve`: the single worst cluster's power, and which it was.
    #: The average across clusters is the number a dashboard would show; this is the
    #: number that matters, and the gap between them is the point.
    worst_power: float | None = None
    worst_cluster: int | None = None

    def as_dict(self) -> dict:
        return {
            "window": self.window,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "n_traces": self.n_traces,
            "coverage": self.coverage,
            "power": self.power,
            "power_lo": self.power_lo,
            "power_hi": self.power_hi,
            "n_affected_cases": self.n_affected_cases,
            "affected_traffic_share": self.affected_traffic_share,
            "worst_power": self.worst_power,
            "worst_cluster": self.worst_cluster,
        }


@dataclass
class DecayCurve:
    """Detection power over time for one suite strategy."""

    label: str
    regression: str
    points: list[DecayPoint]
    meta: dict = field(default_factory=dict)

    @property
    def powers(self) -> np.ndarray:
        return np.asarray([p.power for p in self.points], dtype=float)

    @property
    def coverages(self) -> np.ndarray:
        return np.asarray(
            [np.nan if p.coverage is None else p.coverage for p in self.points], dtype=float
        )

    @property
    def worst_powers(self) -> np.ndarray:
        return np.asarray(
            [np.nan if p.worst_power is None else p.worst_power for p in self.points], dtype=float
        )

    def crossover(self, level: float = 0.5) -> int | None:
        """First window whose power is below `level`, or None if it never is."""
        for p in self.points:
            if p.power < level:
                return p.window
        return None

    def worst_crossover(self, level: float = 0.5) -> int | None:
        """First window in which *some* region of traffic drops below `level`.

        This fires long before `crossover` does, and it is the honest alarm: an
        average over clusters stays respectable while one cluster goes to zero, which
        is the same aggregation failure this library exists to point at.
        """
        for p in self.points:
            if p.worst_power is not None and p.worst_power < level:
                return p.window
        return None

    def table(self) -> str:
        has_worst = any(p.worst_power is not None for p in self.points)
        head = f"{'window':<8}{'traces':>8}{'coverage':>10}{'power':>22}"
        head += f"{'worst cluster':>16}" if has_worst else ""
        lines = [head, "-" * len(head)]
        for p in self.points:
            cov = "n/a" if p.coverage is None else f"{p.coverage:.1%}"
            row = (
                f"{p.window:<8}{p.n_traces:>8}{cov:>10}"
                f"{f'{p.power:.3f} [{p.power_lo:.3f}, {p.power_hi:.3f}]':>22}"
            )
            if has_worst:
                row += f"{f'{p.worst_power:.3f} (c{p.worst_cluster})':>16}"
            lines.append(row)
        return "\n".join(lines)

    def summary(self) -> str:  # pragma: no cover - display only
        cross = self.crossover()
        worst = self.worst_crossover()
        tail = [
            f"  average power never falls below 0.5 across {len(self.points)} windows"
            if cross is None
            else f"  average power falls below 0.5 at window {cross}"
        ]
        if any(p.worst_power is not None for p in self.points):
            tail.append(
                "  no single region drops below 0.5"
                if worst is None
                else f"  some region of traffic drops below 0.5 at window {worst} - the average "
                "above is hiding it"
            )
        return f"decay  {self.label}  vs  {self.regression}\n{self.table()}\n" + "\n".join(tail)

    def as_dict(self) -> dict:
        return {
            "kind": "decay",
            "label": self.label,
            "regression": self.regression,
            "crossover": self.crossover(),
            "worst_crossover": self.worst_crossover(),
            "points": [p.as_dict() for p in self.points],
            "meta": self.meta,
        }


def decay(
    traces: TraceSet,
    judge: Judge,
    make_regression: Callable[[TraceSet], Regression],
    suite: EvalSuite | None = None,
    make_suite: Callable[[TraceSet, int], EvalSuite] | None = None,
    n_windows: int = 8,
    label: str = "frozen",
    n_sim: int = 300,
    seed: int = 0,
    with_coverage: bool = True,
    **power_kwargs,
) -> DecayCurve:
    """Detection power window by window.

    Supply exactly one of:

    - `suite` - a fixed suite, held constant across every window. This is the frozen
      arm, and the one that decays.
    - `make_suite(window_traces, window_index)` - rebuilt from each window's own
      traffic. This is the mined arm, and the comparison.

    `make_regression(window_traces)` builds the alternative from the window's traffic,
    which is what lets "the regression hits whatever is currently growing" be asked
    consistently at every point on the curve.
    """
    if (suite is None) == (make_suite is None):
        raise ValueError("pass exactly one of `suite` or `make_suite`")

    windows = traces.sorted_by_time().windows(n_windows)
    points: list[DecayPoint] = []
    regression_name = ""
    baseline_cache: SuiteRun | None = None

    for i, window in enumerate(windows):
        if len(window) == 0:
            continue
        active = suite if suite is not None else make_suite(window, i)  # type: ignore[misc]
        if len(active) == 0:
            continue

        regression = make_regression(window)
        regression_name = regression.name

        if suite is not None and baseline_cache is not None:
            baseline = baseline_cache
        else:
            baseline = run_suite(active, judge, label=f"w{i}")
            if suite is not None:
                baseline_cache = baseline

        result: PowerResult = estimate(
            active, baseline, regression, n_sim=n_sim, seed=seed + i, **power_kwargs
        )

        cov = None
        if with_coverage:
            from livingeval.mine.coverage import coverage as _coverage

            cov_res = _coverage(active, window, seed=seed)
            cov = cov_res.coverage

        affected_traffic = float(
            np.mean([regression.predicate(_as_case(t)) for t in window])
        ) if len(window) else 0.0

        points.append(
            DecayPoint(
                window=i,
                t_start=window.tmin,
                t_end=window.tmax,
                n_traces=len(window),
                coverage=cov,
                power=result.power.point,
                power_lo=result.power.lo,
                power_hi=result.power.hi,
                n_affected_cases=result.n_affected_cases,
                affected_traffic_share=affected_traffic,
            )
        )

    return DecayCurve(label, regression_name, points, {"n_windows": n_windows, "seed": seed})


def staleness_curve(
    traces: TraceSet,
    judge: Judge,
    suite: EvalSuite | None = None,
    make_suite: Callable[[TraceSet, int], EvalSuite] | None = None,
    n_windows: int = 8,
    label: str = "frozen",
    effect: float = 0.40,
    top_clusters: int = 8,
    n_sim: int = 200,
    seed: int = 0,
    **gate_kwargs,
) -> DecayCurve:
    """Expected power over time, against a regression of unknown location.

    In every window the traffic is re-clustered, the suite's cases are assigned to
    that window's clusters, and detection power is averaged over which cluster the
    regression lands in, weighted by traffic share.

    This is the curve that decays, and the mechanism is worth stating plainly: the
    suite does not get worse. The traffic it never covered gets bigger.
    """
    from livingeval.mine.cluster import cluster as _cluster
    from livingeval.mine.coverage import coverage as _coverage
    from livingeval.mine.space import Space
    from livingeval.power.estimate import expected_power

    if (suite is None) == (make_suite is None):
        raise ValueError("pass exactly one of `suite` or `make_suite`")

    windows = traces.sorted_by_time().windows(n_windows)
    points: list[DecayPoint] = []
    baseline_cache: SuiteRun | None = None

    for i, window in enumerate(windows):
        if len(window) < 20:
            continue
        active = suite if suite is not None else make_suite(window, i)  # type: ignore[misc]
        if len(active) == 0:
            continue

        space = Space.fit(window, seed=seed)
        clustering = _cluster(window, space=space, seed=seed)

        if suite is not None and baseline_cache is not None:
            baseline = baseline_cache
        else:
            baseline = run_suite(active, judge, label=f"w{i}")
            if suite is not None:
                baseline_cache = baseline

        exp = expected_power(
            active, baseline, clustering, effect=effect, top=top_clusters,
            n_sim=n_sim, seed=seed + i * 97, **gate_kwargs
        )
        cov = _coverage(active, window, space=space, seed=seed).coverage
        worst = exp.worst_cluster

        points.append(
            DecayPoint(
                window=i,
                t_start=window.tmin,
                t_end=window.tmax,
                n_traces=len(window),
                coverage=cov,
                power=exp.expected,
                power_lo=min(exp.per_cluster.values()) if exp.per_cluster else float("nan"),
                power_hi=max(exp.per_cluster.values()) if exp.per_cluster else float("nan"),
                n_affected_cases=0 if worst is None else round(exp.weights.get(worst, 0.0) * len(active)),
                affected_traffic_share=exp.weight_covered,
                worst_power=None if worst is None else exp.per_cluster[worst],
                worst_cluster=worst,
            )
        )

    return DecayCurve(
        label,
        f"random cluster (effect={effect:g})",
        points,
        {"n_windows": n_windows, "seed": seed, "kind": "expected", "top_clusters": top_clusters},
    )


def _as_case(trace):
    """Wrap a trace so a `Regression` predicate written against cases can be applied
    to raw traffic. Predicates read `case.trace` and `case.tags`, both of which this
    provides; nothing else is needed and nothing else is faked."""
    from livingeval.suite.case import EvalCase

    return EvalCase(case_id=trace.trace_id, trace=trace, expected=1, provenance="curated")
