"""Comparing two runs of a generated suite.

No network: the runs are dictionaries of the shape the generated suite writes,
which is the whole point of that shape being a plain artifact.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from livingeval.baseline import MIN_RUNS, Measurement, update_registry
from livingeval.gate.regress import compare


def registry(**overrides) -> dict:
    base = {
        "workflow.faithfulness": {"direction": "higher", "noise": 0.02,
                                  "measured": True, "catches": "ungrounded claims"},
        "application.toxicity": {"direction": "lower", "noise": 0.02,
                                 "measured": True, "catches": "hostile output"},
        "application.latency_p95": {"direction": "lower", "noise": 0.10,
                                    "measured": True, "catches": "tail latency"},
    }
    for key, patch in overrides.items():
        base[key.replace("__", ".")].update(patch)
    return base


def run(metrics: dict, cases: dict | None = None) -> dict:
    return {"metrics": metrics, "cases": cases or {}}


def cases(key: str, before: list[bool]) -> dict:
    return {key: [{"id": f"c{i}", "passed": p, "score": 1.0 if p else 0.0}
                  for i, p in enumerate(before)]}


# ---------------------------------------------------------------------------
# direction
# ---------------------------------------------------------------------------


def test_a_fall_in_a_higher_is_better_metric_is_a_regression():
    report = compare(run({"workflow.faithfulness": 0.90}),
                     run({"workflow.faithfulness": 0.70}), registry())
    assert report.verdict == "FAIL"
    assert report.regressed[0].key == "workflow.faithfulness"


def test_a_fall_in_a_lower_is_better_metric_is_an_improvement():
    """Getting this backwards reports every safety fix as a regression."""
    report = compare(run({"application.toxicity": 0.30}),
                     run({"application.toxicity": 0.05}), registry())
    assert report.verdict != "FAIL"
    assert report.improved[0].key == "application.toxicity"


def test_a_rise_in_latency_is_a_regression():
    report = compare(run({"application.latency_p95": 1.0}),
                     run({"application.latency_p95": 3.0}), registry())
    assert report.verdict == "FAIL"


# ---------------------------------------------------------------------------
# noise
# ---------------------------------------------------------------------------


def test_movement_inside_the_noise_threshold_is_not_a_regression():
    report = compare(run({"workflow.faithfulness": 0.90}),
                     run({"workflow.faithfulness": 0.89}), registry())
    assert report.verdict == "PASS"


def test_unmeasured_thresholds_return_blind_not_pass():
    """A comparison against a guess is not evidence of stability."""
    report = compare(run({"workflow.faithfulness": 0.90}),
                     run({"workflow.faithfulness": 0.90}),
                     registry(workflow__faithfulness={"measured": False}))
    assert report.verdict == "BLIND"
    assert any("estimated noise" in reason for reason in report.blind)


def test_a_real_regression_beats_blind():
    """FAIL first: a suite that caught something has done its job, and
    downgrading that because the thresholds are shaky suppresses a real signal."""
    report = compare(run({"workflow.faithfulness": 0.90}),
                     run({"workflow.faithfulness": 0.40}),
                     registry(workflow__faithfulness={"measured": False}))
    assert report.verdict == "FAIL"


# ---------------------------------------------------------------------------
# the paired test
# ---------------------------------------------------------------------------


def test_per_case_outcomes_produce_a_p_value():
    key = "workflow.faithfulness"
    before = cases(key, [True] * 20)
    after = cases(key, [True] * 8 + [False] * 12)
    report = compare(run({key: 1.0}, before), run({key: 0.4}, after), registry())
    delta = report.deltas[0]
    assert delta.p is not None and delta.q is not None
    assert delta.verdict == "regressed"


def test_a_small_consistent_wobble_is_not_significant():
    key = "workflow.faithfulness"
    before = cases(key, [True] * 19 + [False])
    after = cases(key, [True] * 18 + [False] * 2)
    report = compare(run({key: 0.95}, before), run({key: 0.90}, after), registry())
    assert report.deltas[0].verdict != "regressed"


def test_cases_are_matched_by_id_not_position():
    """A suite that grew between runs would otherwise pair unrelated cases."""
    key = "workflow.faithfulness"
    before = {key: [{"id": "a", "passed": True}, {"id": "b", "passed": True},
                    {"id": "c", "passed": True}, {"id": "d", "passed": True},
                    {"id": "e", "passed": True}, {"id": "f", "passed": True}]}
    after = {key: [{"id": "z", "passed": False}] +
                  [{"id": i, "passed": True} for i in "abcdef"]}
    report = compare(run({key: 1.0}, before), run({key: 0.85}, after), registry())
    assert report.deltas[0].verdict != "regressed"


def test_multiplicity_is_corrected_across_the_family():
    """Twenty-two metrics at alpha=0.05 produces a false alarm most runs, which
    is how teams learn to ignore their own gate."""
    reg, before, after, metrics_b, metrics_c = {}, {}, {}, {}, {}
    for i in range(20):
        key = f"workflow.m{i}"
        reg[key] = {"direction": "higher", "noise": 0.02, "measured": True}
        # One case flips in each metric: individually marginal, jointly noise.
        before.update(cases(key, [True] * 10))
        after.update(cases(key, [True] * 9 + [False]))
        metrics_b[key], metrics_c[key] = 1.0, 0.9
    report = compare(run(metrics_b, before), run(metrics_c, after), reg)
    assert report.verdict != "FAIL", "one flipped case in each of 20 metrics is noise"
    assert all(d.q is not None and d.q >= d.p for d in report.deltas)


# ---------------------------------------------------------------------------
# degenerate input
# ---------------------------------------------------------------------------


def test_runs_sharing_no_metrics_are_blind_not_pass():
    report = compare(run({"a": 1.0}), run({"b": 1.0}), {})
    assert report.verdict == "BLIND"


def test_metrics_present_in_only_one_run_are_reported():
    reg = registry()
    report = compare(run({"workflow.faithfulness": 0.9, "application.toxicity": 0.0}),
                     run({"workflow.faithfulness": 0.9}), reg)
    assert any("only one run" in reason for reason in report.blind)


def test_report_renders_without_raising():
    report = compare(run({"workflow.faithfulness": 0.9}),
                     run({"workflow.faithfulness": 0.5}), registry())
    text = report.render()
    assert "FAIL" in text and "workflow.faithfulness" in text


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------


def test_too_few_runs_is_refused_rather_than_guessed():
    from livingeval.baseline import measure
    with pytest.raises(ValueError, match="cannot produce a usable"):
        measure(Path("."), runs=MIN_RUNS - 1)


def test_update_registry_marks_metrics_measured(tmp_path):
    path = tmp_path / "metric_registry.py"
    path.write_text(
        '"""doc that must survive."""\n\n'
        'REGISTRY = {\n    "workflow.faithfulness": {"direction": "higher", '
        '"noise": 0.04, "measured": False},\n}\n', encoding="utf-8")
    updated = update_registry(path, {
        "workflow.faithfulness": Measurement(
            key="workflow.faithfulness", mean=0.9, stdev=0.01,
            noise=0.02, values=[0.9] * 6)})
    assert updated == 1
    source = path.read_text(encoding="utf-8")
    assert "doc that must survive" in source
    namespace: dict = {}
    exec(compile(source, "r.py", "exec"), namespace)
    entry = namespace["REGISTRY"]["workflow.faithfulness"]
    assert entry["measured"] is True and entry["noise"] == 0.02 and entry["runs"] == 6


def test_measured_thresholds_turn_blind_into_pass(tmp_path):
    """The whole point of `livingeval baseline`."""
    unmeasured = registry(workflow__faithfulness={"measured": False})
    assert compare(run({"workflow.faithfulness": 0.9}),
                   run({"workflow.faithfulness": 0.9}), unmeasured).verdict == "BLIND"
    assert compare(run({"workflow.faithfulness": 0.9}),
                   run({"workflow.faithfulness": 0.9}), registry()).verdict == "PASS"


def test_a_metric_that_produced_no_number_in_either_run_is_blind_not_silent():
    """It appears in neither run, so it is in neither the shared set nor the
    symmetric difference -- the quietest way for a suite to stop measuring
    something. The runner already knows why; the gate repeats it."""
    base = {"metrics": {"application.toxicity": 0.9},
            "unmeasured": {"application.correctness": "awaiting reference answers"}}
    cand = {"metrics": {"application.toxicity": 0.9},
            "unmeasured": {"application.correctness": "awaiting reference answers"}}
    report = compare(base, cand, registry())
    assert report.verdict == "BLIND"
    assert any("application.correctness" in note for note in report.blind)
    assert any("awaiting reference answers" in note for note in report.blind)


def test_a_test_that_could_not_have_fired_is_not_reported_as_reassurance():
    """Exact McNemar is a sign test over the cases that changed. With three of
    them the smallest two-sided p available is 0.25, so no effect of any size
    makes that test fire -- and printing "within noise" says the opposite of
    what happened.

    Measured live: a prompt edit dropped scope adherence from 0.883 to 0.550 on
    a six-case set and the gate called it noise at p=0.5."""
    base = run({"workflow.faithfulness": 0.90},
               cases("workflow.faithfulness", [True] * 8 + [False] * 2))
    cand = run({"workflow.faithfulness": 0.60},
               {"workflow.faithfulness": [
                   {"id": f"c{i}", "passed": i >= 3} for i in range(10)]})
    report = compare(base, cand, registry())
    delta = report.deltas[0]
    assert delta.floor_p is not None
    assert delta.underpowered, f"floor {delta.floor_p} vs q {delta.q}"
    assert report.verdict == "BLIND"
    assert any("too few cases" in note for note in report.blind)


def test_a_well_powered_test_is_not_flagged_underpowered():
    before = [True] * 40
    after = [{"id": f"c{i}", "passed": i >= 20} for i in range(40)]
    base = run({"workflow.faithfulness": 1.0}, cases("workflow.faithfulness", before))
    cand = run({"workflow.faithfulness": 0.5}, {"workflow.faithfulness": after})
    report = compare(base, cand, registry())
    delta = report.deltas[0]
    assert not delta.underpowered
    assert delta.verdict == "regressed"
