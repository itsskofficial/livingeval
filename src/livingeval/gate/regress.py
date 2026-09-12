"""Comparing two runs of a generated suite.

`gate.decide` answers "did this one suite's pass rate move", case by case,
against one baseline. A generated suite asks a different question: twenty-odd
metrics across several pipelines moved by various amounts, in various
directions, some of which are improvements. Which of those movements are real?

The usual answer -- run the baseline ten times, take twice the standard
deviation as a noise threshold, call anything smaller noise -- is the right
instinct implemented as a rule of thumb, and it is wrong in three specific ways
this module fixes.

**It has no false-alarm rate.** Two sigma is a number, not a test. Where per-case
outcomes exist this runs exact McNemar on them and reports a p-value, which is
what "is this real" actually means.

**It ignores multiplicity.** Twenty-two metrics compared at alpha=0.05 produces
roughly one false alarm per run, every run, which is precisely how teams learn
to ignore their own gate. Benjamini-Hochberg across the family fixes it, and the
report shows both p and q so the correction is visible rather than assumed.

**It cannot tell "no regression" from "could not have seen one".** A comparison
against unmeasured noise thresholds is not evidence of stability; it is the
absence of evidence. That returns BLIND, the same as everywhere else in this
library, because CI should be able to treat the two differently.

Direction matters and is easy to get backwards: a fall in toxicity is an
improvement and a fall in faithfulness is not. The registry carries it, and
`_moved_badly` is the only place that reads it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from livingeval.stats.tests import bh_fdr, mcnemar_exact

__all__ = ["MetricDelta", "RegressionReport", "compare", "load_run"]

EXIT_CODES = {"PASS": 0, "FAIL": 1, "BLIND": 2}


@dataclass
class MetricDelta:
    """One metric's movement, and how much to believe it."""

    key: str
    baseline: float
    candidate: float
    direction: str                 # "higher" or "lower" is better
    noise: float
    measured_noise: bool
    verdict: str                   # improved | regressed | flat | noise
    p: float | None = None
    q: float | None = None
    discordant: tuple[int, int] | None = None
    catches: str = ""
    #: The smallest p this metric's paired test could have produced, given how
    #: many cases disagreed between the runs. Not the p it did produce -- the
    #: best it was capable of. When that best is above the threshold, "within
    #: noise" means "this test cannot speak", which is a different sentence.
    floor_p: float | None = None
    alpha: float = 0.05

    @property
    def underpowered(self) -> bool:
        """Whether this test could have rejected at all.

        Compared against alpha rather than against the corrected q, because the
        question is what the test was capable of, not what it happened to
        return. Benjamini-Hochberg only ever makes the bar stricter, so a floor
        above alpha is a floor above the corrected threshold too -- this is the
        conservative half of the claim.
        """
        return self.floor_p is not None and self.floor_p > self.alpha

    @property
    def delta(self) -> float:
        return self.candidate - self.baseline

    @property
    def better(self) -> float:
        """Signed movement in the direction that counts as improvement."""
        return self.delta if self.direction == "higher" else -self.delta

    def line(self) -> str:
        arrow = "+" if self.better > 0 else ("-" if self.better < 0 else " ")
        mark = {"regressed": "REGRESSED", "improved": "improved",
                "noise": "within noise", "flat": "flat"}[self.verdict]
        stat = ""
        if self.q is not None:
            stat = f"  p={self.p:.3f} q={self.q:.3f}"
            if self.underpowered:
                stat += "  (too few cases to reach significance)"
        elif not self.measured_noise:
            stat = "  (noise threshold is an estimate)"
        return (f"  {arrow} {self.key:<38} {self.baseline:6.3f} -> "
                f"{self.candidate:6.3f}  {mark}{stat}")


@dataclass
class RegressionReport:
    verdict: str
    deltas: list[MetricDelta] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    blind: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.verdict]

    @property
    def regressed(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.verdict == "regressed"]

    @property
    def improved(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.verdict == "improved"]

    def render(self) -> str:
        out = [f"-> {self.verdict}  (exit {self.exit_code})"]
        for reason in self.reasons:
            out.append(f"   {reason}")
        for reason in self.blind:
            out.append(f"   {reason}")
        out.append("")
        for delta in sorted(self.deltas, key=lambda d: d.better):
            out.append(delta.line())
        counts = (f"{len(self.regressed)} regressed, {len(self.improved)} improved, "
                  f"{len(self.deltas) - len(self.regressed) - len(self.improved)} unchanged")
        out += ["", f"  {counts}"]
        return "\n".join(out)

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "reasons": self.reasons,
            "blind": self.blind,
            "metrics": [
                {"key": d.key, "baseline": d.baseline, "candidate": d.candidate,
                 "delta": d.delta, "verdict": d.verdict, "p": d.p, "q": d.q,
                 "floor_p": d.floor_p, "underpowered": d.underpowered}
                for d in self.deltas
            ],
        }


def load_run(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _paired(baseline: dict, candidate: dict, key: str) -> tuple[float, tuple[int, int]] | None:
    """McNemar on per-case pass/fail, when both runs recorded it.

    Cases are matched by id rather than by position: a suite that grew between
    runs would otherwise pair unrelated cases and report nonsense with
    confidence.
    """
    before = {c["id"]: c for c in baseline.get("cases", {}).get(key, [])}
    after = {c["id"]: c for c in candidate.get("cases", {}).get(key, [])}
    shared = sorted(set(before) & set(after))
    if len(shared) < 5:
        return None
    b = np.array([1 if before[i].get("passed") else 0 for i in shared], dtype=int)
    a = np.array([1 if after[i].get("passed") else 0 for i in shared], dtype=int)
    p, n01, n10 = mcnemar_exact(b, a)
    return p, (n01, n10)


def _floor_p(discordant: int) -> float:
    """The smallest two-sided p the exact test could return for this many
    disagreements.

    Exact McNemar is a two-sided binomial sign test over the cases that changed.
    With `d` of them, the most extreme outcome available is all `d` falling the
    same way, giving p = 2 * 0.5**d. So d=4 cannot go below 0.125 however large
    the effect, and no amount of regression will make that test fire.

    This is detection power stated exactly rather than simulated, and it is the
    number that separates "nothing happened" from "this suite is too small to
    tell". A generated suite ships five to seven cases per metric, so it is
    usually the second.
    """
    if discordant <= 0:
        return 1.0
    return min(1.0, 2.0 * 0.5 ** discordant)


def _moved_badly(delta: float, direction: str, noise: float) -> bool:
    """Whether the movement is in the wrong direction and larger than noise."""
    signed = delta if direction == "higher" else -delta
    return signed < -abs(noise)


def compare(baseline: dict, candidate: dict, registry: dict,
            alpha: float = 0.05) -> RegressionReport:
    """Decide PASS, FAIL or BLIND across every metric in the two runs."""
    base_metrics = baseline.get("metrics", {})
    cand_metrics = candidate.get("metrics", {})
    shared = sorted(set(base_metrics) & set(cand_metrics))

    if not shared:
        return RegressionReport(
            verdict="BLIND",
            blind=["the two runs share no metrics, so nothing was compared"])

    deltas: list[MetricDelta] = []
    pvalues: list[float] = []
    indexed: list[int] = []

    for key in shared:
        entry = registry.get(key, {})
        direction = entry.get("direction", "higher")
        noise = float(entry.get("noise", 0.0))
        measured = bool(entry.get("measured", False))

        delta = MetricDelta(
            key=key, baseline=float(base_metrics[key]),
            candidate=float(cand_metrics[key]), direction=direction,
            noise=noise, measured_noise=measured, verdict="flat",
            catches=entry.get("catches", ""), alpha=alpha)

        paired = _paired(baseline, candidate, key)
        if paired is not None:
            delta.p, delta.discordant = paired[0], paired[1]
            delta.floor_p = _floor_p(sum(paired[1]))
            indexed.append(len(deltas))
            pvalues.append(delta.p)
        deltas.append(delta)

    # One family of tests, corrected once. Reporting the metrics that "came out
    # significant" uncorrected manufactures a finding roughly every run.
    if pvalues:
        rejected, qvalues = bh_fdr(np.array(pvalues), alpha=alpha)
        for position, index in enumerate(indexed):
            deltas[index].q = float(qvalues[position])
            significant = bool(rejected[position])
            wrong_way = deltas[index].better < 0
            if significant and wrong_way:
                deltas[index].verdict = "regressed"
            elif significant:
                deltas[index].verdict = "improved"
            else:
                deltas[index].verdict = "noise"

    # Metrics with no per-case data fall back to the noise threshold. It is a
    # weaker claim and the report says so.
    for delta in deltas:
        if delta.q is not None:
            continue
        if _moved_badly(delta.delta, delta.direction, delta.noise):
            delta.verdict = "regressed"
        elif abs(delta.better) > abs(delta.noise):
            delta.verdict = "improved"
        else:
            delta.verdict = "noise" if delta.delta else "flat"

    reasons: list[str] = []
    blind: list[str] = []

    regressed = [d for d in deltas if d.verdict == "regressed"]
    for delta in regressed:
        detail = (f"p={delta.p:.3f}, q={delta.q:.3f}" if delta.q is not None
                  else f"beyond a noise threshold of {delta.noise:.3f}")
        reasons.append(f"{delta.key} fell {abs(delta.better):.4f} ({detail})"
                       + (f": {delta.catches}" if delta.catches else ""))

    # Unmeasured thresholds are the common case on a fresh suite, and comparing
    # against a guess is not evidence of stability.
    estimated = [d for d in deltas if not d.measured_noise and d.q is None]
    if estimated:
        blind.append(
            f"{len(estimated)} of {len(deltas)} metrics were compared against "
            f"estimated noise thresholds with no per-case data: run "
            f"`livingeval baseline --runs 10` to measure them")

    # A paired test that could not have rejected at any effect size did not
    # find nothing; it was never able to. Reporting that as "within noise" is
    # the specific way a small suite reassures people -- and after correcting
    # across a family of sixteen, the bar a single metric must clear is far
    # below what five cases can produce.
    underpowered = [d for d in deltas if d.underpowered]
    if underpowered:
        worst = max(underpowered, key=lambda d: abs(d.better) if d.better < 0 else 0)
        detail = ""
        if worst.better < 0:
            detail = (f" -- {worst.key} moved {abs(worst.better):.3f} the wrong way "
                      f"and still could not be called")
        blind.append(
            f"{len(underpowered)} of {len(deltas)} metrics have too few cases for "
            f"the paired test to reach significance at any effect size{detail}. "
            f"Add cases to those golden sets, or read their movement against the "
            f"measured noise threshold instead")

    skipped = set(base_metrics) ^ set(cand_metrics)
    if skipped:
        blind.append(f"{len(skipped)} metrics appear in only one run and were not "
                     f"compared ({', '.join(sorted(skipped)[:3])}"
                     f"{'...' if len(skipped) > 3 else ''})")

    # A metric that produced no number in *either* run is in neither set above
    # and would otherwise pass unremarked -- which is the quietest way for a
    # suite to stop measuring something. The runs say why each one is missing;
    # the gate repeats it rather than deciding without it.
    never = {**baseline.get("unmeasured", {}), **candidate.get("unmeasured", {})}
    never = {k: v for k, v in never.items() if k not in shared}
    if never:
        listed = "; ".join(f"{k} ({v})" for k, v in sorted(never.items())[:3])
        blind.append(f"{len(never)} planned metric(s) produced no number in "
                     f"either run: {listed}"
                     f"{'; ...' if len(never) > 3 else ''}")

    # FAIL first: a suite that caught a regression has done its job, and
    # downgrading that to BLIND because the thresholds are shaky would suppress
    # a real signal.
    if regressed:
        verdict = "FAIL"
    elif blind:
        verdict = "BLIND"
    else:
        verdict = "PASS"
        reasons.append(f"{len(deltas)} metrics compared, none regressed")

    return RegressionReport(verdict=verdict, deltas=deltas,
                            reasons=reasons, blind=blind)
