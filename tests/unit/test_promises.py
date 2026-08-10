"""Layer A - the library's promises, as tests.

If any of these fail, the README is wrong. They are the reason the synthetic
generators exist: each claim is checked against a case whose answer is known by
construction, with no model, no network and no API key.
"""

from __future__ import annotations

import numpy as np
import pytest

import livingeval as le
from livingeval import synthetic as syn

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


# ---------------------------------------------------------------------------
# Promise 1 - the ladder detects a shallow judge
# ---------------------------------------------------------------------------


def test_keyword_rung_reproduces_a_one_phrase_judge():
    """`shortcut`: the label is the presence of one fixed phrase, so rung 2 must
    reach kappa 1.0 and `judge_depth` must be `keyword`."""
    traces = syn.shortcut(n=600, seed=1)
    result = le.scorer.ladder(traces, le.judge.oracle(), n_boot=200)
    keyword = next(r for r in result.rungs if r.name == "keyword")
    assert keyword.kappa.point >= 0.95
    assert result.depth == "keyword"


def test_bow_is_needed_when_twelve_phrases_share_the_work():
    """`lexical`: twelve unrelated hedges. One keyword reaches a twelfth of them;
    a bag of words reaches all of them. The ladder has to separate those."""
    traces = syn.lexical(n=900, seed=1)
    result = le.scorer.ladder(traces, le.judge.oracle(), n_boot=200)
    by_name = {r.name: r.kappa.point for r in result.rungs}
    assert by_name["keyword"] < 0.6
    assert by_name["bow"] >= 0.95
    assert result.depth == "bow"


def test_character_ngrams_are_needed_for_a_morphological_rule():
    """`morphology`: identifiers are unique per trace, so only the character-level
    casing signal generalises across a group-aware split."""
    traces = syn.morphology(n=900, seed=1)
    result = le.scorer.ladder(traces, le.judge.oracle(), n_boot=200)
    by_name = {r.name: r.kappa.point for r in result.rungs}
    assert by_name["bow"] < 0.4
    assert by_name["charngram"] >= 0.8
    assert result.depth == "charngram"


# ---------------------------------------------------------------------------
# Promise 2 - the ladder does not cry wolf
# ---------------------------------------------------------------------------


def test_no_rung_reproduces_a_judge_that_compares_two_spans():
    """`deep`: the label is the agreement between the tool result and the claim.
    Both are marginally balanced, so every bag-of-features model is at chance and
    `judge_depth` must be None.

    This is the test that keeps the ladder honest. Without it, "your judge is
    shallow" would be a conclusion the instrument reaches about everything."""
    traces = syn.deep(n=900, seed=1)
    result = le.scorer.ladder(traces, le.judge.oracle(), n_boot=200)
    assert result.depth is None
    assert all(r.kappa.point < 0.35 for r in result.rungs)


# ---------------------------------------------------------------------------
# Promise 3 - coverage notices drift
# ---------------------------------------------------------------------------


def test_a_suite_covers_itself_completely():
    traces = syn.stable(n=300, seed=0)
    suite = le.EvalSuite.from_traces(traces, name="self")
    assert le.mine.coverage(suite, traces).coverage == pytest.approx(1.0)


def test_coverage_of_disjoint_traffic_is_low():
    """Two corpora with no shared vocabulary: coverage must not be high just
    because both are text."""
    from livingeval.synthetic import DriftSchedule, TrafficSpec, generate

    old = generate(TrafficSpec(n=300, seed=0, drift=DriftSchedule(
        base_intents=("billing", "shipping"), new_intents=(), onset=1.0)))
    new = generate(TrafficSpec(n=300, seed=1, drift=DriftSchedule(
        base_intents=("crypto_payouts", "voice_agent"), new_intents=(), onset=1.0)))
    suite = le.EvalSuite.from_traces(old, name="old")
    assert le.mine.coverage(suite, new).coverage < 0.15


def test_a_frozen_suite_loses_coverage_as_traffic_drifts():
    traces = syn.drifting(n=1600, seed=0)
    windows = traces.sorted_by_time().windows(6)
    frozen = le.EvalSuite.from_traces(windows[0].sample(150, seed=0), name="frozen")
    first = le.mine.coverage(frozen, windows[0]).coverage
    last = le.mine.coverage(frozen, windows[-1]).coverage
    assert first - last > 0.15


def test_a_suite_of_one_case_reports_uncalibrated_rather_than_a_number():
    traces = syn.stable(n=200, seed=0)
    suite = le.EvalSuite.from_traces(traces[:1], name="tiny")
    result = le.mine.coverage(suite, traces)
    assert result.status == "UNCALIBRATED"
    assert result.coverage is None


# ---------------------------------------------------------------------------
# Promise 4 - blind spots name the region that appeared
# ---------------------------------------------------------------------------


def test_the_emerging_intent_is_the_top_blind_spot():
    traces = syn.drifting(n=1600, seed=0)
    windows = traces.sorted_by_time().windows(6)
    frozen = le.EvalSuite.from_traces(windows[0].sample(150, seed=0), name="frozen")
    recent = windows[-1]
    report = le.mine.blindspots(frozen, recent, seed=0)

    worst = report.spots[0]
    members = [t for t, lab in zip(recent, report.clustering.labels, strict=False) if lab == worst.cluster]
    share_new = np.mean([t.meta["is_new_intent"] for t in members])
    assert share_new > 0.8, "the worst blind spot should be the intent that just appeared"
    assert worst.n_suite_cases == 0
    assert worst.coverage < 0.2


# ---------------------------------------------------------------------------
# Promise 5 - power distinguishes a suite that can see from one that cannot
# ---------------------------------------------------------------------------


def test_a_frozen_suite_has_no_power_against_a_regression_it_cannot_see():
    traces = syn.drifting(n=1600, seed=0)
    windows = traces.sorted_by_time().windows(6)
    judge = le.judge.oracle(noise=0.05, seed=7)

    frozen = le.EvalSuite.from_traces(windows[0].sample(150, seed=0), name="frozen")
    mined = le.EvalSuite.from_traces(windows[-1].sample(150, seed=1), name="mined")
    regression = le.power.on_meta("is_new_intent", True, effect=0.5)

    p_frozen = le.power.estimate(frozen, le.run_suite(frozen, judge), regression, n_sim=200)
    p_mined = le.power.estimate(mined, le.run_suite(mined, judge), regression, n_sim=200)

    assert p_frozen.is_blind
    assert p_frozen.power.point < 0.15
    assert p_mined.power.point > 0.70


def test_power_against_a_zero_effect_regression_equals_the_false_alarm_rate():
    """Definitional, and worth pinning: the two numbers must come from the same code
    path or they are not comparable."""
    traces = syn.stable(n=600, seed=0)
    judge = le.judge.oracle(noise=0.05, seed=3)
    suite = le.EvalSuite.from_traces(traces.sample(120, seed=0), name="s")
    base = le.run_suite(suite, judge)

    null = le.power.uniform(0.0)
    a = le.power.estimate(suite, base, null, n_sim=300, seed=11)
    b = le.power.false_alarm_rate(suite, base, n_sim=300, seed=11)
    assert a.power.point == pytest.approx(b.power.point)


# ---------------------------------------------------------------------------
# Promise 6 - a fixed threshold fires far more often than a paired test
# ---------------------------------------------------------------------------


def test_the_legacy_threshold_gate_false_alarms_far_more_than_the_paired_gate():
    traces = syn.stable(n=800, seed=1)
    judge = le.judge.oracle(noise=0.05, seed=11)
    suite = le.EvalSuite.from_traces(traces.sample(100, seed=5), name="s")
    base = le.run_suite(suite, judge)
    threshold = round(base.score - 0.02, 3)

    paired = le.power.false_alarm_rate(suite, base, n_sim=400, seed=2)
    legacy = le.power.false_alarm_rate(
        suite, base, n_sim=400, seed=2, mode="threshold", threshold=threshold
    )
    assert paired.power.point < 0.10, "a paired test must control its Type-I error"
    assert legacy.power.point > 0.30
    assert legacy.power.point > paired.power.point * 3


def test_zero_flake_means_zero_false_alarms_for_every_gate():
    """The parameter that makes the comparison meaningful: with a perfectly stable
    system, no gate design can false-alarm, and the audit's headline would be empty."""
    traces = syn.stable(n=400, seed=1)
    judge = le.judge.oracle()
    suite = le.EvalSuite.from_traces(traces.sample(80, seed=0), name="s")
    base = le.run_suite(suite, judge)
    result = le.power.false_alarm_rate(suite, base, n_sim=100, flake=0.0,
                                       mode="threshold", threshold=base.score - 0.02)
    assert result.power.point == 0.0
