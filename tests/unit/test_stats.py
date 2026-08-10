"""Layer A - analytical invariants for the statistics.

No model, no network, no reference numbers. These hold or the maths is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from livingeval.stats import (
    bh_fdr,
    bootstrap_ci,
    cohens_kappa,
    kappa_ci,
    mcnemar_exact,
    paired_bootstrap_diff,
    wilson_ci,
)

# -- Cohen's kappa ----------------------------------------------------------


def test_kappa_of_identical_labellings_is_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    assert cohens_kappa(y, y) == pytest.approx(1.0)


def test_kappa_of_independent_labellings_is_about_zero():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 2, 4000)
    b = rng.integers(0, 2, 4000)
    assert abs(cohens_kappa(a, b)) < 0.05


def test_kappa_corrects_for_an_imbalanced_marginal():
    """Two independent labellers with a 90% pass rate agree 82% of the time. Raw
    agreement calls that excellent; kappa calls it nothing, which is the point."""
    rng = np.random.default_rng(2)
    a = (rng.random(8000) < 0.9).astype(int)
    b = (rng.random(8000) < 0.9).astype(int)
    assert np.mean(a == b) > 0.78
    assert abs(cohens_kappa(a, b)) < 0.05


def test_kappa_of_two_constant_labellers_is_zero_not_nan():
    a = np.ones(50, dtype=int)
    assert cohens_kappa(a, a.copy()) == 0.0


def test_kappa_is_symmetric():
    rng = np.random.default_rng(3)
    a, b = rng.integers(0, 3, 300), rng.integers(0, 3, 300)
    assert cohens_kappa(a, b) == pytest.approx(cohens_kappa(b, a))


def test_kappa_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(4)
    a = rng.integers(0, 2, 400)
    b = np.where(rng.random(400) < 0.2, 1 - a, a)
    ci = kappa_ci(a, b, n_boot=400)
    assert ci.lo <= ci.point <= ci.hi
    assert ci.point > 0.5


# -- McNemar ----------------------------------------------------------------


def test_mcnemar_reduces_to_the_binomial_on_discordant_pairs():
    before = np.array([1] * 12 + [0] * 4 + [1] * 30)
    after = np.array([0] * 12 + [1] * 4 + [1] * 30)
    p, n01, n10 = mcnemar_exact(before, after)
    assert (n01, n10) == (12, 4)
    assert p == pytest.approx(float(sps.binomtest(4, 16, 0.5).pvalue))


def test_mcnemar_with_no_discordant_pairs_is_one():
    y = np.array([1, 0, 1, 1, 0])
    p, n01, n10 = mcnemar_exact(y, y.copy())
    assert (p, n01, n10) == (1.0, 0, 0)


def test_mcnemar_ignores_concordant_pairs():
    """Adding cases that behave identically in both arms changes nothing. That is
    the property that makes the paired test powerful on a small suite."""
    before = np.array([1] * 8 + [0] * 2)
    after = np.array([0] * 8 + [1] * 2)
    p1, *_ = mcnemar_exact(before, after)
    pad = np.ones(500, dtype=int)
    p2, *_ = mcnemar_exact(np.concatenate([before, pad]), np.concatenate([after, pad]))
    assert p1 == pytest.approx(p2)


def test_mcnemar_type_one_error_is_controlled():
    """Under the null the test fires at most alpha of the time."""
    rng = np.random.default_rng(5)
    fires = 0
    trials = 600
    for _ in range(trials):
        p_true = 0.9
        before = (rng.random(120) < p_true).astype(int)
        after = (rng.random(120) < p_true).astype(int)
        p, *_ = mcnemar_exact(before, after)
        fires += p < 0.05
    assert fires / trials < 0.08


# -- bootstrap --------------------------------------------------------------


def test_bootstrap_ci_covers_the_truth_about_95_percent_of_the_time():
    rng = np.random.default_rng(6)
    covered = 0
    trials = 300
    for _ in range(trials):
        sample = rng.normal(0.0, 1.0, 200)
        ci = bootstrap_ci(sample, n_boot=400, seed=int(rng.integers(1 << 30)))
        covered += ci.lo <= 0.0 <= ci.hi
    assert 0.88 < covered / trials < 1.0


def test_paired_bootstrap_is_narrower_than_the_unpaired_difference():
    """The reason gates must be paired: correlated arms make the interval on the
    difference much tighter at the same n."""
    rng = np.random.default_rng(7)
    base = rng.normal(0, 1, 300)
    after = base + rng.normal(0, 0.05, 300)
    paired = paired_bootstrap_diff(base, after, n_boot=600)
    unpaired_width = (
        bootstrap_ci(after, n_boot=600).hi - bootstrap_ci(after, n_boot=600).lo
    ) * 2
    assert (paired.hi - paired.lo) < unpaired_width / 2


def test_cluster_bootstrap_is_wider_than_the_observation_bootstrap():
    """Correlated observations within a group carry less information than their
    count suggests, and the interval has to say so."""
    rng = np.random.default_rng(8)
    groups = np.repeat(np.arange(40), 10)
    group_means = rng.normal(0, 1, 40)
    values = group_means[groups] + rng.normal(0, 0.05, 400)
    naive = bootstrap_ci(values, n_boot=500)
    clustered = bootstrap_ci(values, n_boot=500, groups=groups)
    assert (clustered.hi - clustered.lo) > 2 * (naive.hi - naive.lo)


def test_wilson_interval_stays_inside_the_unit_scale():
    for successes, n in ((0, 20), (20, 20), (1, 500)):
        ci = wilson_ci(successes, n)
        assert 0.0 <= ci.lo <= ci.point <= ci.hi <= 1.0


# -- multiplicity -----------------------------------------------------------


def test_bh_fdr_rejects_nothing_under_the_null_most_of_the_time():
    rng = np.random.default_rng(9)
    false_discoveries = 0
    for _ in range(200):
        p = rng.random(20)
        rejected, _ = bh_fdr(p, alpha=0.05)
        false_discoveries += rejected.any()
    assert false_discoveries / 200 < 0.15


def test_bh_fdr_is_monotone_and_bounded():
    p = np.array([0.001, 0.01, 0.02, 0.2, 0.9])
    rejected, q = bh_fdr(p, alpha=0.05)
    assert np.all(q >= p - 1e-12)
    assert np.all(q <= 1.0)
    assert rejected[0]
    assert not rejected[-1]


def test_bh_fdr_handles_an_empty_family():
    rejected, q = bh_fdr(np.array([]))
    assert rejected.size == 0 and q.size == 0
