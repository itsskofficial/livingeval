"""How often would this suite actually catch that regression?

The estimate is a Monte Carlo simulation, and the docstring says so in every place
the number surfaces. It answers: *given this suite's composition and this gate's
statistics, if the named regression were present, what fraction of the time would the
gate fire?* It is a statement about the measuring instrument. It is not a prediction
about your model, and reporting it as one would be the fastest way to lose the
audience that can tell the difference.

Two numbers come out and they belong together:

- **power** - the fire rate under the regression, at `effect > 0`.
- **false alarm** - the fire rate under *no* regression at all, which is the same
  procedure at `effect = 0`. It is the gate's realised Type-I error, and running it
  is the cheapest way to find out that a fixed threshold on a small suite is firing
  at random.

The simulation resamples the *regression*, not the suite. The question is about this
suite, so its composition is held fixed; what varies is which of the affected cases
happen to break. Add `resample_cases=True` to fold in the suite-composition
uncertainty as well, which widens the interval and answers the slightly different
question "a suite like this one".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from livingeval.gate.decide import evaluate
from livingeval.gate.run import SuiteRun
from livingeval.power.regression import Regression
from livingeval.stats.tests import Interval, wilson_ci
from livingeval.suite.suite import EvalSuite

__all__ = [
    "ExpectedPower",
    "PowerResult",
    "assign_cases_to_clusters",
    "estimate",
    "expected_power",
    "expected_rerun_score",
    "false_alarm_rate",
    "measure_flake",
]


@dataclass
class PowerResult:
    """Simulated detection power for one (suite, gate, regression) triple."""

    suite_name: str
    regression: str
    effect: float
    n_sim: int
    power: Interval
    n_affected_cases: int
    n_gateable: int
    affected_fraction: float
    mode: str
    blind_rate: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def is_blind(self) -> bool:
        """No case in the suite is affected, so the regression is invisible by
        construction and 'power' equals the false-alarm rate."""
        return self.n_affected_cases == 0

    def summary(self) -> str:  # pragma: no cover - display only
        lines = [
            f"power (simulated)  {self.suite_name}  vs  {self.regression}",
            f"  {self.n_affected_cases}/{self.n_gateable} gateable case(s) affected "
            f"({self.affected_fraction:.1%})   gate mode={self.mode}   {self.n_sim} simulations",
            f"  power  {self.power}",
        ]
        if self.is_blind:
            lines.append(
                "  NO case in this suite is affected by this regression. The number above "
                "is the gate's false-alarm rate, not its detection rate."
            )
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "kind": "power",
            "suite": self.suite_name,
            "regression": self.regression,
            "effect": self.effect,
            "n_sim": self.n_sim,
            "power": self.power.as_dict(),
            "n_affected_cases": self.n_affected_cases,
            "n_gateable": self.n_gateable,
            "affected_fraction": self.affected_fraction,
            "mode": self.mode,
            "is_blind": self.is_blind,
            "blind_rate": self.blind_rate,
            "meta": self.meta,
        }


def assign_cases_to_clusters(suite: EvalSuite, clustering, space=None) -> dict[str, int]:
    """Map each suite case onto the nearest cluster centroid of current traffic.

    Cases are assigned to the *traffic's* partition rather than clustered separately,
    so "this cluster has two cases" refers to the same clusters the coverage table and
    the blind-spot report use.
    """
    from livingeval.mine.space import Space

    space = space or clustering.space
    if not len(suite):
        return {}
    if clustering.centroids is None:
        raise ValueError("clustering has no centroids; pass one produced by mine.cluster")
    keys = sorted(clustering.sizes)
    coords = space.transform(suite.traces)
    assign = np.argmin(Space.cosine_distance(coords, clustering.centroids), axis=1)
    return {c.case_id: keys[int(a)] for c, a in zip(suite, assign, strict=False)}


def _build(baseline: SuiteRun, ids: list[str], idx: np.ndarray, outcomes: np.ndarray, label: str) -> SuiteRun:
    return SuiteRun(
        baseline.suite_name, baseline.suite_hash, baseline.judge_name, ids,
        outcomes, baseline.expected[idx], baseline.observed[idx], baseline.groups[idx],
        baseline.gateable[idx], label, 0.0,
    )


def _simulate(
    suite: EvalSuite,
    baseline: SuiteRun,
    regression: Regression,
    n_sim: int,
    seed: int,
    gate_kwargs: dict,
    resample_cases: bool,
    flake: float,
) -> tuple[int, int, int, int]:
    """Return `(n_fired, n_blind, n_affected_cases, n_gateable)`.

    Both arms are redrawn every replicate. This matters: if the baseline were held at
    its observed value and only the current arm were perturbed, the gate would face a
    noise-free reference it never has in practice, and the false-alarm rate would come
    out at zero for every gate design - which would make the comparison between them
    meaningless.

    `flake` is the per-case probability that a rerun of the *same* system flips the
    outcome: the judge is stochastic, decoding is stochastic, retrieval reorders. Each
    case is modelled as a Bernoulli with `p = 1 - flake` where it passed and
    `p = flake` where it failed, so neither arm drifts systematically and the only
    asymmetry between them is the regression.
    """
    rng = np.random.default_rng(seed)
    affected = regression.affected(suite)
    gateable = baseline.gateable
    n_affected = int((affected & gateable).sum())
    n = len(baseline.outcomes)
    p = np.where(baseline.outcomes == 1, 1.0 - flake, flake)
    identity = np.arange(n)

    fired = blind = 0
    for _ in range(n_sim):
        if resample_cases:
            idx = rng.integers(0, n, size=n)
            ids = [f"{baseline.case_ids[i]}#{k}" for k, i in enumerate(idx)]
        else:
            idx = identity
            ids = list(baseline.case_ids)

        before = (rng.random(n) < p[idx]).astype(int)
        after = (rng.random(n) < p[idx]).astype(int)
        if regression.effect > 0:
            flip = affected[idx] & (after == 1) & (rng.random(n) < regression.effect)
            after[flip] = 0

        result = evaluate(
            _build(baseline, ids, idx, after, "current"),
            _build(baseline, ids, idx, before, "baseline"),
            **gate_kwargs,
        )
        if result.decision == "FAIL":
            fired += 1
        elif result.decision == "BLIND":
            blind += 1

    return fired, blind, n_affected, int(gateable.sum())


def estimate(
    suite: EvalSuite,
    baseline: SuiteRun,
    regression: Regression,
    n_sim: int = 400,
    seed: int = 0,
    resample_cases: bool = False,
    flake: float = 0.05,
    **gate_kwargs,
) -> PowerResult:
    """Simulate the gate under `regression` and report how often it fires.

    `gate_kwargs` are forwarded to `gate.evaluate`, so the power you measure is the
    power of the gate you actually ship, not of an idealised one. Coverage- and
    power-based BLIND checks are switched off inside the simulation, since including
    them here would be circular.

    `flake` is the per-case rerun instability, and it is the parameter to think
    hardest about: at `flake=0` every gate has a false-alarm rate of zero and every
    comparison between gate designs is vacuous. The 5% default is a placeholder, not
    a measurement of your system - `power.measure_flake` estimates it from two real
    runs of the same suite, and that is the number to use.
    """
    kwargs = {"coverage": None, "power": None, "min_cases": 0, "n_boot": 0, **gate_kwargs}
    fired, blind, n_affected, n_gateable = _simulate(
        suite, baseline, regression, n_sim, seed, kwargs, resample_cases, flake
    )
    return PowerResult(
        suite_name=suite.name,
        regression=regression.name,
        effect=regression.effect,
        n_sim=n_sim,
        power=wilson_ci(fired, n_sim),
        n_affected_cases=n_affected,
        n_gateable=n_gateable,
        affected_fraction=(n_affected / n_gateable) if n_gateable else 0.0,
        mode=str(kwargs.get("mode", "paired")),
        blind_rate=blind / n_sim if n_sim else 0.0,
        meta={
            "resample_cases": resample_cases,
            "flake": flake,
            "description": regression.description,
        },
    )


@dataclass
class ExpectedPower:
    """Power against a regression whose location is unknown.

    Asking "would the suite catch a regression in cluster 6" needs you to already
    know where the regression will be. The decision-relevant version averages over
    that: a regression lands in one region of traffic, each region is as likely as
    its share of traffic, and this is the resulting detection rate.

    It is the single number that genuinely *decays*. A suite loses expected power not
    because it gets worse, but because the traffic it does not cover grows.
    """

    suite_name: str
    expected: float
    effect: float
    per_cluster: dict[int, float]
    weights: dict[int, float]
    n_clusters_considered: int
    n_clusters_total: int
    weight_covered: float
    n_sim: int

    @property
    def worst_cluster(self) -> int | None:
        return min(self.per_cluster, key=lambda c: self.per_cluster[c]) if self.per_cluster else None

    def summary(self) -> str:  # pragma: no cover - display only
        lines = [
            f"expected power  {self.suite_name}   effect={self.effect:g}",
            f"  {self.expected:.3f}  averaged over {self.n_clusters_considered} of "
            f"{self.n_clusters_total} clusters carrying {self.weight_covered:.0%} of traffic",
        ]
        for c in sorted(self.per_cluster, key=lambda c: -self.weights[c]):
            lines.append(
                f"    cluster {c:<3} weight {self.weights[c]:>6.1%}   power {self.per_cluster[c]:.3f}"
            )
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "kind": "expected_power",
            "suite": self.suite_name,
            "expected": self.expected,
            "effect": self.effect,
            "per_cluster": {str(k): v for k, v in self.per_cluster.items()},
            "weights": {str(k): v for k, v in self.weights.items()},
            "n_clusters_considered": self.n_clusters_considered,
            "n_clusters_total": self.n_clusters_total,
            "weight_covered": self.weight_covered,
            "n_sim": self.n_sim,
        }


def expected_power(
    suite: EvalSuite,
    baseline: SuiteRun,
    clustering,
    effect: float = 0.40,
    top: int = 6,
    n_sim: int = 200,
    seed: int = 0,
    **kwargs,
) -> ExpectedPower:
    """Detection power averaged over where the regression might land.

    Only the `top` clusters by traffic share are simulated, and the weights are
    renormalised over those - simulating a long tail of 0.5% clusters costs a lot and
    moves the average very little. The share of traffic actually covered is reported
    as `weight_covered` rather than left implicit, because a truncation you cannot see
    is indistinguishable from a claim of completeness.
    """
    from livingeval.power.regression import on_cluster

    assignment = assign_cases_to_clusters(suite, clustering)
    shares = clustering.shares()
    ranked = sorted(shares, key=lambda c: -shares[c])[:top]
    total_weight = sum(shares[c] for c in ranked) or 1.0

    per_cluster: dict[int, float] = {}
    weights: dict[int, float] = {}
    for i, c in enumerate(ranked):
        reg = on_cluster([c], assignment, effect)
        res = estimate(suite, baseline, reg, n_sim=n_sim, seed=seed + i, **kwargs)
        per_cluster[int(c)] = res.power.point
        weights[int(c)] = shares[c] / total_weight

    exp = float(sum(weights[c] * per_cluster[c] for c in per_cluster)) if per_cluster else float("nan")
    return ExpectedPower(
        suite_name=suite.name,
        expected=exp,
        effect=effect,
        per_cluster=per_cluster,
        weights=weights,
        n_clusters_considered=len(ranked),
        n_clusters_total=len(shares),
        weight_covered=float(sum(shares[c] for c in ranked)),
        n_sim=n_sim,
    )


def expected_rerun_score(baseline: SuiteRun, flake: float = 0.05) -> float:
    """The score you should *expect* from rerunning an unchanged system.

    It is not the score you observed. A single run is one draw, and under per-case
    instability the mean of future runs sits at `(1-f)*s + f*(1-s)`, pulled toward
    one half. Setting a gate threshold a couple of points under the *observed* score
    therefore puts it above the expected rerun score, and the gate fires almost every
    time for a reason that has nothing to do with the model.

    That mistake is easy to make and hard to see, so this is the number to set a
    fixed threshold against - and the fact that it exists at all is one more argument
    for not using a fixed threshold.
    """
    s = baseline.score
    return float((1.0 - flake) * s + flake * (1.0 - s))


def measure_flake(run_a: SuiteRun, run_b: SuiteRun) -> float:
    """Estimate per-case rerun instability from two runs of the same suite.

    Run your suite twice against an unchanged system and pass both. The fraction of
    cases whose outcome differs is `2 * flake * (1 - flake)`; this inverts it. Doing
    this once and feeding the result to `estimate(flake=...)` turns every power and
    false-alarm number from an illustration into a measurement of your setup.
    """
    shared = [c for c in run_a.case_ids if c in set(run_b.case_ids)]
    if not shared:
        raise ValueError("the two runs share no cases")
    ia = {c: i for i, c in enumerate(run_a.case_ids)}
    ib = {c: i for i, c in enumerate(run_b.case_ids)}
    disagree = float(
        np.mean([run_a.outcomes[ia[c]] != run_b.outcomes[ib[c]] for c in shared])
    )
    # d = 2f(1-f)  ->  f = (1 - sqrt(1 - 2d)) / 2, valid for d <= 0.5
    if disagree >= 0.5:
        return 0.5
    return float((1.0 - np.sqrt(1.0 - 2.0 * disagree)) / 2.0)


def false_alarm_rate(
    suite: EvalSuite,
    baseline: SuiteRun,
    n_sim: int = 400,
    seed: int = 0,
    **gate_kwargs,
) -> PowerResult:
    """The gate's realised Type-I error: how often it fires with nothing wrong.

    Implemented as `estimate` with a zero-effect regression, so it is the same code
    path the power estimate uses - which is the only way the two numbers are
    comparable.
    """
    from livingeval.power.regression import uniform

    null = uniform(0.0)
    null.name = "null (no regression)"
    null.description = "nothing is wrong; this is the gate's false-alarm rate"
    return estimate(suite, baseline, null, n_sim=n_sim, seed=seed, **gate_kwargs)
