"""The three-valued CI gate.

Every eval gate in production is some form of `if score < threshold: fail`. Two
things are wrong with it and both are fixable.

**It is not a test.** The same cases run before and after, so the comparison is
paired, and a fixed threshold on a small suite is a hypothesis test with an
uncontrolled Type-I error rate. On a fifty-case suite, run-to-run noise crosses a
fixed line often enough that the team stops believing the gate - which is the actual
mechanism by which eval suites get abandoned. `mode="paired"` uses exact McNemar on
the discordant cases and reports the effect with a paired-bootstrap interval.
`mode="threshold"` reproduces the naive gate on purpose, because the gap between
their false-alarm rates is worth measuring rather than asserting.

**It has two outcomes when the situation has three.** "The suite passed" and "the
suite is no longer capable of failing" are different states, and folding the second
into the first is the specific failure this library exists to prevent. So:

    PASS   exit 0   no regression detected, and the suite could have detected one
    FAIL   exit 1   a regression was detected
    BLIND  exit 2   no regression detected, but the suite has lost the coverage or
                    the power to justify saying so

Precedence is FAIL first: a suite that catches a regression has done its job, and
downgrading that to BLIND because coverage slipped would suppress a real signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from livingeval.gate.run import SuiteRun
from livingeval.stats.tests import Interval, mcnemar_exact, paired_bootstrap_diff

__all__ = ["EXIT_CODES", "GateResult", "evaluate"]

EXIT_CODES = {"PASS": 0, "FAIL": 1, "BLIND": 2}


@dataclass
class GateResult:
    """A gate decision, and everything that went into it."""

    decision: str
    mode: str
    score: float
    baseline_score: float | None
    effect: Interval | None
    p_value: float | None
    n_discordant: tuple[int, int] | None
    n_gateable: int
    coverage: float | None
    power: float | None
    reasons: list[str] = field(default_factory=list)
    blind_reasons: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.decision]

    def summary(self) -> str:  # pragma: no cover - display only
        lines = [f"gate  {self.decision}  (exit {self.exit_code})   mode={self.mode}"]
        if self.baseline_score is None:
            lines.append(f"  score {self.score:.4f}  (no baseline supplied)")
        else:
            lines.append(f"  score {self.score:.4f}  baseline {self.baseline_score:.4f}")
        if self.effect is not None:
            lines.append(f"  effect {self.effect}   (current minus baseline)")
        if self.p_value is not None and self.n_discordant is not None:
            n01, n10 = self.n_discordant
            lines.append(f"  McNemar p={self.p_value:.4g}   {n01} regressed, {n10} improved")
        lines.append(f"  gateable cases {self.n_gateable}")
        if self.coverage is not None:
            lines.append(f"  coverage {self.coverage:.1%}")
        if self.power is not None:
            lines.append(f"  power    {self.power:.1%}")
        for r in self.reasons:
            lines.append(f"  - {r}")
        for r in self.blind_reasons:
            lines.append(f"  ! {r}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "kind": "gate",
            "decision": self.decision,
            "exit_code": self.exit_code,
            "mode": self.mode,
            "score": self.score,
            "baseline_score": self.baseline_score,
            "effect": None if self.effect is None else self.effect.as_dict(),
            "p_value": self.p_value,
            "n_discordant": list(self.n_discordant) if self.n_discordant else None,
            "n_gateable": self.n_gateable,
            "coverage": self.coverage,
            "power": self.power,
            "reasons": self.reasons,
            "blind_reasons": self.blind_reasons,
        }


def evaluate(
    current: SuiteRun,
    baseline: SuiteRun | None = None,
    mode: str = "paired",
    alpha: float = 0.05,
    min_effect: float = 0.0,
    threshold: float = 0.90,
    coverage: float | None = None,
    power: float | None = None,
    min_coverage: float = 0.70,
    min_power: float = 0.80,
    min_cases: int = 20,
    n_boot: int = 2000,
    seed: int = 0,
) -> GateResult:
    """Decide PASS, FAIL or BLIND.

    `mode="paired"` needs a baseline and compares case by case. `mode="threshold"` is
    the legacy gate and ignores the baseline entirely; it is here so the audit can
    measure what it costs.
    """
    outcomes, groups = current.gate_view()
    score = current.score
    reasons: list[str] = []
    blind: list[str] = []

    effect: Interval | None = None
    p: float | None = None
    discordant: tuple[int, int] | None = None
    regressed = False

    if mode == "threshold" or baseline is None:
        used_mode = "threshold"
        if baseline is not None:
            reasons.append("threshold mode ignores the baseline; the comparison is unpaired")
        if mode == "paired" and baseline is None:
            reasons.append(
                "no baseline supplied, so the gate fell back to an absolute threshold - "
                "an uncontrolled test rather than a paired one"
            )
        regressed = bool(np.isfinite(score) and score < threshold)
        if regressed:
            reasons.append(f"score {score:.4f} is below the fixed threshold {threshold:.4f}")
    else:
        used_mode = "paired"
        shared = [c for c in current.case_ids if c in set(baseline.case_ids)]
        if not shared:
            raise ValueError("baseline and current runs share no cases; a paired gate needs both")
        cur_index = {c: i for i, c in enumerate(current.case_ids)}
        base_index = {c: i for i, c in enumerate(baseline.case_ids)}
        gate_ok = [
            c for c in shared if current.gateable[cur_index[c]] and baseline.gateable[base_index[c]]
        ]
        b = np.asarray([baseline.outcomes[base_index[c]] for c in gate_ok], dtype=int)
        a = np.asarray([current.outcomes[cur_index[c]] for c in gate_ok], dtype=int)
        g = np.asarray([current.groups[cur_index[c]] for c in gate_ok], dtype=object)

        # The decision comes from McNemar; the bootstrap interval is for reporting the
        # effect size. `n_boot=0` skips it, which is what the power simulation does -
        # it runs this function thousands of times and never looks at the interval.
        effect = (
            paired_bootstrap_diff(b, a, n_boot=n_boot, seed=seed, groups=g) if n_boot > 0 else None
        )
        p, n01, n10 = mcnemar_exact(b, a)
        discordant = (n01, n10)
        drop = float(np.mean(b) - np.mean(a))
        regressed = bool(p < alpha and n01 > n10 and drop >= min_effect)
        if regressed:
            reasons.append(
                f"{n01} case(s) regressed against {n10} recovered (exact McNemar p={p:.4g}), "
                f"a drop of {drop:.4f}"
            )
        elif p < alpha and n10 > n01:
            reasons.append(f"significant *improvement*: {n10} recovered against {n01} regressed")
        elif drop >= min_effect and p >= alpha:
            reasons.append(
                f"score fell by {drop:.4f} but the paired test does not separate it from "
                f"noise at n={len(gate_ok)} (p={p:.4g})"
            )

    if current.n_gateable < min_cases:
        blind.append(
            f"only {current.n_gateable} gateable case(s); below {min_cases} the gate cannot "
            "resolve a realistic regression"
        )
    if coverage is not None and coverage < min_coverage:
        blind.append(
            f"coverage {coverage:.1%} is below {min_coverage:.0%}: "
            f"{1 - coverage:.0%} of traffic is unrepresented by this suite"
        )
    if power is not None and power < min_power:
        blind.append(
            f"simulated detection power {power:.1%} is below {min_power:.0%} for the "
            "regression tested"
        )

    if regressed:
        decision = "FAIL"
    elif blind:
        decision = "BLIND"
    else:
        decision = "PASS"

    return GateResult(
        decision=decision,
        mode=used_mode,
        score=score,
        baseline_score=None if baseline is None else baseline.score,
        effect=effect,
        p_value=p,
        n_discordant=discordant,
        n_gateable=current.n_gateable,
        coverage=coverage,
        power=power,
        reasons=reasons,
        blind_reasons=blind,
    )
