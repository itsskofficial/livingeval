"""Statistics for eval decisions.

Four jobs, and the reason each one is here rather than being a one-liner:

- **Agreement between two labellers** must be chance-corrected. Raw agreement on a
  task with a 90% pass rate starts at 0.90 before either labeller has done anything.
  Cohen's kappa is the correction; a bootstrap gives it an interval.
- **Before/after comparisons of an eval suite are paired.** The same cases run in both
  arms. Treating the two arms as independent samples discards the pairing and inflates
  the variance of the difference, which is how a real regression gets waved through.
- **Binary paired comparisons have an exact test.** McNemar conditions on the number of
  discordant pairs and reduces to a binomial, so there is no reason to use the
  chi-square approximation on the small counts an eval suite actually produces.
- **A layerwise or per-cluster sweep is a family of tests.** Reporting the clusters that
  "came out significant" without correction manufactures findings.

Everything here is pure numpy/scipy and seeded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as _sps

__all__ = [
    "Interval",
    "bh_fdr",
    "bootstrap_ci",
    "cohens_kappa",
    "kappa_ci",
    "mcnemar_exact",
    "paired_bootstrap_diff",
    "wilson_ci",
]


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval."""

    point: float
    lo: float
    hi: float
    level: float = 0.95

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}]"

    def excludes(self, value: float) -> bool:
        """True when `value` lies outside the interval."""
        return value < self.lo or value > self.hi

    def as_dict(self) -> dict:
        return {"point": self.point, "lo": self.lo, "hi": self.hi, "level": self.level}


def _rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def bootstrap_ci(
    values: np.ndarray,
    statistic=np.mean,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int | np.random.Generator | None = 0,
    groups: np.ndarray | None = None,
) -> Interval:
    """Percentile bootstrap CI for `statistic(values)`.

    `groups` switches to a **cluster bootstrap**: whole groups are resampled rather
    than individual observations. Eval cases derived from the same session are not
    independent, and an observation-level bootstrap on clustered data reports an
    interval that is too narrow by roughly the square root of the design effect.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), level)
    rng = _rng(seed)
    point = float(statistic(values))

    if groups is None:
        idx = rng.integers(0, values.size, size=(n_boot, values.size))
        reps = np.asarray([statistic(values[i]) for i in idx], dtype=float)
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        members = [np.flatnonzero(groups == g) for g in uniq]
        reps = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            pick = rng.integers(0, len(uniq), size=len(uniq))
            sel = np.concatenate([members[p] for p in pick])
            reps[b] = statistic(values[sel])

    alpha = 1.0 - level
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(point, float(lo), float(hi), level)


def paired_bootstrap_diff(
    before: np.ndarray,
    after: np.ndarray,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int | np.random.Generator | None = 0,
    groups: np.ndarray | None = None,
) -> Interval:
    """CI for `mean(after) - mean(before)` over **paired** observations.

    Resamples pair indices, not the two arms independently. On an eval suite the
    per-case outcomes are strongly correlated across runs, so the paired interval is
    typically several times narrower than the unpaired one at the same n. That width
    is the whole reason a 40-case suite can detect anything at all.
    """
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    if before.shape != after.shape:
        raise ValueError(f"paired arrays must match: {before.shape} vs {after.shape}")
    return bootstrap_ci(after - before, np.mean, n_boot, level, seed, groups)


def cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa between two labellings of the same items.

    Returns 1.0 for identical labellings, ~0.0 for independent ones. When both
    labellers emit a single constant label the observed and expected agreement are
    both 1.0 and kappa is 0/0; that is reported as 0.0, because two labellers who
    never disagree because neither ever varies have demonstrated nothing.
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.shape != b.shape:
        raise ValueError(f"label arrays must match: {a.shape} vs {b.shape}")
    if a.size == 0:
        return float("nan")

    classes = np.unique(np.concatenate([a, b]))
    index = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    cm = np.zeros((k, k), dtype=float)
    for x, y in zip(a, b, strict=False):
        cm[index[x], index[y]] += 1.0
    n = cm.sum()
    po = np.trace(cm) / n
    pe = float((cm.sum(axis=0) * cm.sum(axis=1)).sum()) / (n * n)
    if np.isclose(pe, 1.0):
        return 0.0
    return float((po - pe) / (1.0 - pe))


def kappa_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int | np.random.Generator | None = 0,
    groups: np.ndarray | None = None,
) -> Interval:
    """Bootstrap CI for Cohen's kappa, optionally clustered by `groups`."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    n = a.size
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), level)
    rng = _rng(seed)
    point = cohens_kappa(a, b)

    if groups is None:
        draws = rng.integers(0, n, size=(n_boot, n))
        reps = np.asarray([cohens_kappa(a[i], b[i]) for i in draws], dtype=float)
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        members = [np.flatnonzero(groups == g) for g in uniq]
        reps = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            pick = rng.integers(0, len(uniq), size=len(uniq))
            sel = np.concatenate([members[p] for p in pick])
            reps[i] = cohens_kappa(a[sel], b[sel])

    reps = reps[np.isfinite(reps)]
    if reps.size == 0:
        return Interval(point, float("nan"), float("nan"), level)
    alpha = 1.0 - level
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(point, float(lo), float(hi), level)


def mcnemar_exact(before: np.ndarray, after: np.ndarray) -> tuple[float, int, int]:
    """Exact McNemar test on paired binary outcomes.

    Returns `(p_value, n01, n10)` where `n01` is the count of cases that passed
    before and failed after. The test conditions on `n01 + n10` and is a two-sided
    binomial test at p=0.5, which is exact for the small discordant counts an eval
    suite produces. With zero discordant pairs the p-value is 1.0.
    """
    before = np.asarray(before).astype(int).ravel()
    after = np.asarray(after).astype(int).ravel()
    if before.shape != after.shape:
        raise ValueError("paired arrays must match")
    n01 = int(np.sum((before == 1) & (after == 0)))
    n10 = int(np.sum((before == 0) & (after == 1)))
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    p = float(_sps.binomtest(min(n01, n10), n, 0.5).pvalue)
    return min(1.0, p), n01, n10


def wilson_ci(successes: int, n: int, level: float = 0.95) -> Interval:
    """Wilson score interval for a proportion.

    Used for false-alarm and power estimates, where the point estimate sits near 0 or
    1 and the normal-approximation interval runs off the end of the scale.
    """
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), level)
    z = float(_sps.norm.ppf(1 - (1 - level) / 2))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Interval(float(p), float(max(0.0, centre - half)), float(min(1.0, centre + half)), level)


def bh_fdr(pvalues: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg step-up procedure.

    Returns `(rejected, qvalues)`. A per-cluster or per-layer sweep is a family of
    tests; reporting the ones that "came out significant" uncorrected manufactures
    findings at a rate of roughly alpha per test.
    """
    p = np.asarray(pvalues, dtype=float).ravel()
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    out_q = np.empty(m, dtype=float)
    out_q[order] = q
    return out_q <= alpha, out_q
