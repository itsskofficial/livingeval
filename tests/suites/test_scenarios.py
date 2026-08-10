"""Layer C - scenario reproductions.

`pytest -m reproduction`, off by default. These re-run each bundled scenario end to
end and assert that the ladder recovers the depth the scenario was designed to have.
That is the check that licenses the whole instrument: if the ladder cannot recover a
depth that is known by construction, no conclusion it reaches about a real judge is
worth anything.

They are off by default because they are slow, not because they are optional.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import livingeval as le
import suites

pytestmark = pytest.mark.reproduction


@pytest.mark.parametrize("name", sorted(suites.SCENARIOS))
def test_the_ladder_recovers_each_scenario_designed_depth(name):
    scenario = suites.load(name)
    traces = scenario.traces(n=900, seed=2)
    result = le.scorer.ladder(traces, scenario.judge(noise=0.0), n_boot=300, seed=0)
    assert result.depth == scenario.designed_depth, (
        f"{name}: designed {scenario.designed_depth}, measured {result.depth}\n{result.table()}"
    )


@pytest.mark.parametrize("name", sorted(suites.SCENARIOS))
def test_each_scenario_runs_the_whole_pipeline(name):
    scenario = suites.load(name)
    traces = scenario.traces(n=600, seed=3)
    judge = scenario.judge()
    windows = traces.sorted_by_time().windows(4)

    suite = le.EvalSuite.from_traces(windows[0].sample(100, seed=0), name=name)
    report = le.mine.blindspots(suite, windows[-1], seed=0)
    assert report.coverage.coverage is not None

    proposals = le.mine.propose(windows[-1], suite, n=10, judge=judge, seed=0)
    assert all(p.as_case().provenance == "mined" for p in proposals)
    grown = suite.extend(proposals.confirm(reviewer="test"))
    assert len(grown.gateable()) >= len(suite)

    base = le.run_suite(grown, judge, "baseline")
    result = le.gate.evaluate(
        base.with_outcomes(base.outcomes, "current"), base,
        coverage=report.coverage.coverage, min_coverage=0.0,
    )
    assert result.decision in ("PASS", "BLIND")


def test_the_audit_findings_hold_at_reduced_scale():
    """A compressed version of the three headline claims, so a change that breaks one
    of them fails a test rather than being noticed at launch."""
    traces = le.synthetic.drifting(n=1200, seed=0)
    judge = le.judge.oracle(noise=0.05, seed=7)
    windows = traces.sorted_by_time().windows(5)

    frozen = le.EvalSuite.from_traces(windows[0].sample(120, seed=0), name="frozen")
    mined = le.EvalSuite.from_traces(windows[-1].sample(120, seed=1), name="mined")

    # 1: the frozen suite is blind to the regression the mined one catches.
    regression = le.power.on_meta("is_new_intent", True, effect=0.5)
    p_frozen = le.power.estimate(frozen, le.run_suite(frozen, judge), regression, n_sim=200)
    p_mined = le.power.estimate(mined, le.run_suite(mined, judge), regression, n_sim=200)
    assert p_frozen.power.point < 0.10 < 0.70 < p_mined.power.point

    # 2: the legacy gate false-alarms an order of magnitude more often.
    base = le.run_suite(frozen, judge)
    thresh = le.power.expected_rerun_score(base, 0.05) - 0.02
    legacy = le.power.false_alarm_rate(frozen, base, n_sim=400, mode="threshold", threshold=thresh)
    paired = le.power.false_alarm_rate(frozen, base, n_sim=400)
    assert legacy.power.point > 5 * max(paired.power.point, 0.005)

    # 3: the ladder recovers a known depth.
    assert le.scorer.ladder(le.synthetic.deep(n=600, seed=1), le.judge.oracle(),
                            n_boot=200).depth is None
