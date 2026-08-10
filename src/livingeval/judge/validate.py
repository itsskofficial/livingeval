"""Is the judge a measurement?

Almost every published agent-eval number is a judge's opinion reported as though it
were ground truth. Two questions make it a measurement instead:

1. **Does it agree with a human, after correcting for chance?** Raw agreement on a
   task with an 80% pass rate starts at 0.68 before the judge has done anything.
   Cohen's kappa removes that floor.
2. **Which way does it fail?** A judge with kappa 0.6 that never flags a real failure
   is useless for a gate, and one that flags everything is useless for triage. The
   two error rates are reported separately, because the single number hides which
   one you have.

When there are no human labels, `validate` returns `UNVALIDATED` rather than
skipping quietly, and that string lands in the result record and in every table that
record renders. You can ship an unvalidated judge. You cannot ship one and have the
output pretend otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from livingeval.judge.base import Judge, JudgeRun, run_judge
from livingeval.stats.tests import Interval, kappa_ci, wilson_ci
from livingeval.trace.types import TraceSet

__all__ = ["JudgeValidation", "validate"]

#: Landis & Koch's bands, which are conventional rather than principled, and are
#: reported as a word next to the number rather than in place of it.
_BANDS = ((0.81, "almost perfect"), (0.61, "substantial"), (0.41, "moderate"),
          (0.21, "fair"), (0.0, "slight"), (-1.0, "worse than chance"))


@dataclass
class JudgeValidation:
    """A judge measured against human labels."""

    judge_name: str
    n: int
    kappa: Interval | None
    raw_agreement: float | None
    false_pass_rate: Interval | None  # judge says pass, human says fail
    false_fail_rate: Interval | None  # judge says fail, human says pass
    human_pass_rate: float | None
    judge_pass_rate: float | None
    status: str  # "VALIDATED" | "UNVALIDATED" | "UNDERPOWERED"
    cost_usd: float = 0.0
    latency_s: float = 0.0

    @property
    def band(self) -> str:
        if self.kappa is None:
            return "unknown"
        return next(word for cut, word in _BANDS if self.kappa.point >= cut)

    def summary(self) -> str:  # pragma: no cover - display only
        head = f"judge validation  {self.judge_name}  [{self.status}]"
        if self.kappa is None:
            return (
                f"{head}\n"
                "  no human-labelled traces were supplied, so this judge has never been\n"
                "  checked against anything. Every number downstream of it inherits that."
            )
        return (
            f"{head}\n"
            f"  n={self.n} labelled  human_pass={self.human_pass_rate:.4f}  judge_pass={self.judge_pass_rate:.4f}\n"
            f"  kappa            {self.kappa}   ({self.band})\n"
            f"  raw agreement    {self.raw_agreement:.4f}\n"
            f"  false pass rate  {self.false_pass_rate}   (judge said pass, human said fail)\n"
            f"  false fail rate  {self.false_fail_rate}   (judge said fail, human said pass)"
        )

    def as_dict(self) -> dict:
        return {
            "kind": "judge_validation",
            "judge": self.judge_name,
            "status": self.status,
            "n": self.n,
            "kappa": None if self.kappa is None else self.kappa.as_dict(),
            "band": self.band,
            "raw_agreement": self.raw_agreement,
            "false_pass_rate": None if self.false_pass_rate is None else self.false_pass_rate.as_dict(),
            "false_fail_rate": None if self.false_fail_rate is None else self.false_fail_rate.as_dict(),
            "human_pass_rate": self.human_pass_rate,
            "judge_pass_rate": self.judge_pass_rate,
            "cost_usd": self.cost_usd,
            "latency_s": self.latency_s,
        }


def validate(
    judge: Judge | JudgeRun,
    traces: TraceSet | None = None,
    min_n: int = 30,
    n_boot: int = 2000,
    seed: int = 0,
) -> JudgeValidation:
    """Measure a judge against human labels.

    Pass a `JudgeRun` to reuse verdicts you already have, or a judge plus traces to
    run it. Only traces carrying `label` are used; if none do, the result is
    `UNVALIDATED`.

    `min_n` guards the interpretation, not the computation. Below it the status is
    `UNDERPOWERED` and the kappa is still reported with its (wide) interval, because
    hiding a number is a worse failure mode than labelling it.
    """
    if isinstance(judge, JudgeRun):
        run = judge
        if traces is None:
            raise ValueError("pass the same traces used to produce the JudgeRun")
        human = np.asarray([t.label for t in traces], dtype=object)
        judged = run.labels
        groups = run.groups
    else:
        if traces is None:
            raise ValueError("validate(judge, traces) needs traces")
        run = run_judge(judge, traces)
        human = np.asarray([t.label for t in traces], dtype=object)
        judged = run.labels
        groups = run.groups

    mask = np.asarray([h is not None for h in human], dtype=bool)
    n = int(mask.sum())
    if n == 0:
        return JudgeValidation(
            judge_name=run.judge_name, n=0, kappa=None, raw_agreement=None,
            false_pass_rate=None, false_fail_rate=None, human_pass_rate=None,
            judge_pass_rate=None, status="UNVALIDATED",
            cost_usd=run.total_cost_usd, latency_s=run.uncached_latency_s,
        )

    h = np.asarray([int(x) for x in human[mask]], dtype=int)
    j = judged[mask]
    g = groups[mask]

    k = kappa_ci(h, j, n_boot=n_boot, seed=seed, groups=g)
    raw = float(np.mean(h == j))

    human_fail = h == 0
    human_pass = h == 1
    fp = wilson_ci(int(np.sum(human_fail & (j == 1))), int(human_fail.sum())) if human_fail.any() else None
    fn = wilson_ci(int(np.sum(human_pass & (j == 0))), int(human_pass.sum())) if human_pass.any() else None

    return JudgeValidation(
        judge_name=run.judge_name,
        n=n,
        kappa=k,
        raw_agreement=raw,
        false_pass_rate=fp,
        false_fail_rate=fn,
        human_pass_rate=float(np.mean(h)),
        judge_pass_rate=float(np.mean(j)),
        status="VALIDATED" if n >= min_n else "UNDERPOWERED",
        cost_usd=run.total_cost_usd,
        latency_s=run.uncached_latency_s,
    )
