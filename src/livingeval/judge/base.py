"""The judge protocol.

A judge maps a trace to a label. That is the entire contract, and everything else in
this module exists to stop the two ways it goes wrong:

- **Judging is expensive, so people run it once and never again.** Verdicts are
  content-addressed and cached to disk, so re-running an analysis costs nothing and
  iterating on the analysis stops being a budget decision.
- **Cost is quoted rather than measured.** Every `Verdict` carries the wall-clock
  time it took and, for API judges, a cost derived from the reported token counts.
  The ladder's cost column is instrumentation, not a price list someone typed in.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from livingeval.trace.types import Trace, TraceSet

__all__ = ["Judge", "JudgeRun", "Verdict", "run_judge"]


@dataclass(frozen=True)
class Verdict:
    """One judgement."""

    label: int
    score: float | None = None
    rationale: str | None = None
    cost_usd: float = 0.0
    latency_s: float = 0.0
    cached: bool = False
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "score": self.score,
            "rationale": self.rationale,
            "cost_usd": self.cost_usd,
            "latency_s": self.latency_s,
            "cached": self.cached,
        }


@runtime_checkable
class Judge(Protocol):
    """Anything that labels a trace."""

    name: str

    def __call__(self, trace: Trace) -> Verdict: ...


@dataclass
class JudgeRun:
    """The result of running a judge over a trace set."""

    judge_name: str
    trace_ids: list[str]
    verdicts: list[Verdict]
    groups: np.ndarray

    @property
    def labels(self) -> np.ndarray:
        return np.asarray([v.label for v in self.verdicts], dtype=int)

    @property
    def total_cost_usd(self) -> float:
        return float(sum(v.cost_usd for v in self.verdicts))

    @property
    def uncached_latency_s(self) -> float:
        """Mean wall-clock per uncached call. Cached calls are excluded because
        averaging them in would report the cost of the second run, not the first."""
        live = [v.latency_s for v in self.verdicts if not v.cached]
        return float(np.mean(live)) if live else 0.0

    @property
    def cost_per_1k(self) -> float:
        live = [v.cost_usd for v in self.verdicts if not v.cached]
        return float(np.mean(live) * 1000) if live else 0.0

    def summary(self) -> str:  # pragma: no cover - display only
        pos = float(np.mean(self.labels)) if len(self.labels) else float("nan")
        return (
            f"judge[{self.judge_name}]  n={len(self.verdicts)}  pass_rate={pos:.4f}\n"
            f"  cost ${self.total_cost_usd:.4f} total, ${self.cost_per_1k:.4f}/1k traces\n"
            f"  latency {self.uncached_latency_s * 1000:.1f} ms/trace (uncached)"
        )

    def as_dict(self) -> dict:
        return {
            "judge": self.judge_name,
            "n": len(self.verdicts),
            "pass_rate": float(np.mean(self.labels)) if len(self.labels) else None,
            "total_cost_usd": self.total_cost_usd,
            "cost_per_1k": self.cost_per_1k,
            "latency_s_per_trace": self.uncached_latency_s,
        }


def run_judge(judge: Judge, traces: Iterable[Trace] | TraceSet, progress: bool = False) -> JudgeRun:
    """Run a judge over traces, measuring wall-clock even when the judge does not."""
    traces = list(traces)
    verdicts: list[Verdict] = []
    for i, t in enumerate(traces):
        t0 = time.perf_counter()
        v = judge(t)
        elapsed = time.perf_counter() - t0
        if v.latency_s == 0.0 and not v.cached:
            v = Verdict(v.label, v.score, v.rationale, v.cost_usd, elapsed, v.cached, v.meta)
        verdicts.append(v)
        if progress and (i + 1) % 200 == 0:  # pragma: no cover - display only
            print(f"  judged {i + 1}/{len(traces)}", flush=True)
    return JudgeRun(
        judge_name=getattr(judge, "name", judge.__class__.__name__),
        trace_ids=[t.trace_id for t in traces],
        verdicts=verdicts,
        groups=np.asarray([t.group for t in traces], dtype=object),
    )
