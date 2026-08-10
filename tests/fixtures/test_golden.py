"""Layer B - fixed-fixture goldens.

A pinned deterministic config whose outputs were recorded once by
`scripts/record_goldens.py`. These do not prove anything is *correct* - Layer A does
that. They catch numerical drift: a refactor that quietly changes a default, a
scikit-learn release that reorders a tie, a change to the canonical rendering that
moves every downstream number by a hair.

Regenerating them is a separate, visible act. There is no `--update` flag, because a
suite that rewrites its own expectations on failure verifies nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import livingeval as le
from livingeval import synthetic as syn

GOLDEN = json.loads((Path(__file__).parent / "golden.json").read_text(encoding="utf-8"))
TOL = 1e-6


@pytest.fixture(scope="module")
def fixture():
    traces = syn.shortcut(n=300, seed=42)
    judge = le.judge.oracle(noise=0.1, seed=42)
    suite = le.EvalSuite.from_traces(traces.sample(80, seed=42), name="golden")
    return traces, judge, suite


def test_generator_output_is_byte_stable(fixture):
    """If this moves, every other golden in the file is meaningless."""
    traces, _, suite = fixture
    assert traces[0].content_hash() == GOLDEN["trace_content_hash"]
    assert suite.content_hash() == GOLDEN["suite_content_hash"]


def test_ladder_is_stable(fixture):
    traces, judge, _ = fixture
    result = le.scorer.ladder(traces, judge, n_boot=200, seed=42)
    assert result.depth == GOLDEN["ladder_depth"]
    for rung in result.rungs:
        assert rung.kappa.point == pytest.approx(GOLDEN["ladder_kappa"][rung.name], abs=TOL)


def test_judge_validation_is_stable(fixture):
    traces, judge, _ = fixture
    result = le.judge.validate(judge, traces, n_boot=200, seed=42)
    assert result.status == GOLDEN["judge_status"]
    assert result.kappa.point == pytest.approx(GOLDEN["judge_kappa"], abs=TOL)


def test_clustering_and_coverage_are_stable(fixture):
    traces, _, suite = fixture
    space = le.mine.Space.fit(traces, seed=42)
    clustering = le.mine.cluster(traces, space=space, k=5, seed=42)
    coverage = le.mine.coverage(suite, traces, space=space, clustering=clustering)
    assert {str(k): v for k, v in clustering.sizes.items()} == GOLDEN["clustering_sizes"]
    assert coverage.coverage == pytest.approx(GOLDEN["coverage"], abs=TOL)
    assert coverage.radius == pytest.approx(GOLDEN["coverage_radius"], abs=TOL)


def test_suite_run_and_false_alarm_are_stable(fixture):
    _, judge, suite = fixture
    run = le.run_suite(suite, judge)
    assert run.score == pytest.approx(GOLDEN["suite_score"], abs=TOL)
    fa = le.power.false_alarm_rate(suite, run, n_sim=200, seed=42)
    assert fa.power.point == pytest.approx(GOLDEN["false_alarm_paired"], abs=TOL)


def test_the_golden_ladder_records_an_honest_near_miss():
    """The pinned fixture uses a 10%-noisy judge, so no rung clears the 0.80 bar even
    though the underlying rule is a single phrase: judge noise caps how well anything
    can reproduce the judge. That `ladder_depth` is null here is the recorded truth,
    not an oversight - and it is a useful reminder that `judge_depth` measures
    reproducibility of the judge's *labels*, noise included."""
    assert GOLDEN["ladder_depth"] is None
    assert GOLDEN["ladder_kappa"]["keyword"] > 0.7
