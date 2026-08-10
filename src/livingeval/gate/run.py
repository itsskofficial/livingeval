"""Running a suite: from cases and a judge to per-case outcomes.

A `SuiteRun` keeps the **per-case** outcome vector, not just the mean. That is the
whole reason the gate can be a paired test: the same cases run before and after, so
the comparison is within-case and the between-case variance - which is most of it -
cancels. A tool that stores only the aggregate score has thrown that away and is
left comparing two independent-looking numbers.

Unconfirmed mined cases are run and reported but excluded from `gateable`, so they
improve coverage without being able to block a deployment before a human has looked
at them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from livingeval.judge.base import Judge, Verdict
from livingeval.suite.suite import EvalSuite

__all__ = ["SuiteRun", "run_suite"]


@dataclass
class SuiteRun:
    """Per-case outcomes of running a suite."""

    suite_name: str
    suite_hash: str
    judge_name: str
    case_ids: list[str]
    outcomes: np.ndarray  # 1 = the judge's label matched the expected label
    expected: np.ndarray
    observed: np.ndarray
    groups: np.ndarray
    gateable: np.ndarray  # bool
    label: str = "run"
    cost_usd: float = 0.0

    @property
    def score(self) -> float:
        """Pass rate over gateable cases."""
        sel = self.outcomes[self.gateable]
        return float(np.mean(sel)) if sel.size else float("nan")

    @property
    def n_gateable(self) -> int:
        return int(self.gateable.sum())

    def gate_view(self) -> tuple[np.ndarray, np.ndarray]:
        return self.outcomes[self.gateable], self.groups[self.gateable]

    def with_outcomes(self, outcomes: np.ndarray, label: str) -> SuiteRun:
        """A copy carrying different outcomes - used by the power simulation to build
        a hypothetical 'after' run without re-judging anything."""
        return SuiteRun(
            self.suite_name, self.suite_hash, self.judge_name, list(self.case_ids),
            np.asarray(outcomes, dtype=int), self.expected, self.observed, self.groups,
            self.gateable, label, self.cost_usd,
        )

    def summary(self) -> str:  # pragma: no cover - display only
        skipped = len(self.case_ids) - self.n_gateable
        note = f"  ({skipped} unconfirmed mined case(s) excluded)" if skipped else ""
        return (
            f"suite run  {self.suite_name}[{self.label}]  judge={self.judge_name}\n"
            f"  score {self.score:.4f} over {self.n_gateable} gateable case(s){note}"
        )

    def as_dict(self) -> dict:
        return {
            "kind": "suite_run",
            "suite": self.suite_name,
            "suite_hash": self.suite_hash,
            "judge": self.judge_name,
            "label": self.label,
            "score": self.score,
            "n": len(self.case_ids),
            "n_gateable": self.n_gateable,
            "cost_usd": self.cost_usd,
            "case_ids": self.case_ids,
            "outcomes": self.outcomes.tolist(),
        }


def run_suite(suite: EvalSuite, judge: Judge, label: str = "run") -> SuiteRun:
    """Judge every case and record whether the verdict matched the expectation."""
    verdicts: list[Verdict] = [judge(c.trace) for c in suite]
    observed = np.asarray([v.label for v in verdicts], dtype=int)
    expected = suite.expected
    return SuiteRun(
        suite_name=suite.name,
        suite_hash=suite.content_hash(),
        judge_name=getattr(judge, "name", judge.__class__.__name__),
        case_ids=[c.case_id for c in suite],
        outcomes=(observed == expected).astype(int),
        expected=expected,
        observed=observed,
        groups=suite.groups,
        gateable=np.asarray([c.gateable for c in suite], dtype=bool),
        label=label,
        cost_usd=float(sum(v.cost_usd for v in verdicts)),
    )
