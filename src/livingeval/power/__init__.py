"""Simulated detection power, false-alarm rates, and staleness curves."""

from livingeval.power.decay import DecayCurve, DecayPoint, decay, staleness_curve
from livingeval.power.estimate import (
    ExpectedPower,
    PowerResult,
    assign_cases_to_clusters,
    estimate,
    expected_power,
    expected_rerun_score,
    false_alarm_rate,
    measure_flake,
)
from livingeval.power.regression import Regression, on_cluster, on_meta, on_tag, uniform

__all__ = [
    "DecayCurve",
    "DecayPoint",
    "ExpectedPower",
    "PowerResult",
    "Regression",
    "assign_cases_to_clusters",
    "decay",
    "estimate",
    "expected_power",
    "expected_rerun_score",
    "false_alarm_rate",
    "measure_flake",
    "on_cluster",
    "on_meta",
    "on_tag",
    "staleness_curve",
    "uniform",
]
