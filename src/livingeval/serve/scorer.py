"""The online scorer: the ladder's recommendation, fitted and held in memory.

This is what makes per-turn online evaluation affordable, and the claim the whole
library is built to support. The original target was "under 50 ms per turn". A fitted
`bow` rung answers in tens to hundreds of **microseconds** — two to three orders of
magnitude under it — because the ladder already told you a bag of words reproduces the
judge, so the judge does not have to come along.

`DeployedScorer` fits one rung on labelled data and then answers single requests. Three
things it does that a `pipeline.predict` call would not:

- **Reports its own kappa against the judge, measured under group-aware CV**, so the
  service can tell you how much to trust it rather than just emitting labels. A scorer
  with no accompanying agreement number is a random number generator with good latency.
- **Measures its latency at the boundary it is used at.** One item, not a batch. Batch
  throughput on 2,000 documents hides the per-call overhead that dominates a live turn.
- **Refuses to load a rung the ladder did not clear**, unless you say `force=True`. The
  point of measuring depth is that it constrains what you deploy.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from livingeval.judge.base import Judge, JudgeRun, run_judge
from livingeval.scorer.ladder import LadderResult, ladder
from livingeval.scorer.rungs import make_rung
from livingeval.trace.render import render_view
from livingeval.trace.types import Trace, TraceSet

__all__ = ["DeployedScorer", "ScoreResult"]


@dataclass
class ScoreResult:
    """One online score."""

    label: int
    scorer: str
    latency_us: float
    kappa_vs_judge: float | None
    trace_id: str | None = None

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "pass": bool(self.label),
            "scorer": self.scorer,
            "latency_us": round(self.latency_us, 2),
            "kappa_vs_judge": self.kappa_vs_judge,
            "trace_id": self.trace_id,
        }


@dataclass
class DeployedScorer:
    """A fitted rung, ready to answer one turn at a time."""

    rung_name: str
    model: object
    view: str
    kappa_vs_judge: float | None
    n_train: int
    judge_name: str
    fitted_seconds: float
    ladder_result: LadderResult | None = None
    calls: int = 0
    total_latency_us: float = 0.0
    labels_seen: list[int] = field(default_factory=list)

    # -- construction -----------------------------------------------------------

    @classmethod
    def from_ladder(
        cls,
        traces: TraceSet,
        judge: Judge | JudgeRun,
        result: LadderResult | None = None,
        rung: str | None = None,
        view: str = "full",
        force: bool = False,
        **ladder_kwargs,
    ) -> DeployedScorer:
        """Fit the rung the ladder recommends, on all of the data.

        `rung=` overrides the recommendation. Without `force=True`, deploying a rung
        the ladder did not clear raises — measuring depth is only useful if it
        constrains the deployment.
        """
        run = judge if isinstance(judge, JudgeRun) else run_judge(judge, traces)
        if result is None:
            got = ladder(traces, run, view=view, **ladder_kwargs)
            if not isinstance(got, LadderResult):
                raise TypeError("pass a single ladder result, not a per-cluster dict")
            result = got

        chosen = rung or result.depth
        if chosen is None:
            raise ValueError(
                "no rung cleared the kappa bar, so there is nothing safe to deploy. Either "
                "re-run the ladder with `finetune=True` to see whether a fine-tuned model "
                "can, or pass rung=... force=True and accept the measured agreement."
            )
        if rung is not None and not force:
            match = next((r for r in result.rungs if r.name == rung), None)
            if match is None:
                raise ValueError(f"rung {rung!r} is not in this ladder")
            if match.kappa.point < result.threshold:
                raise ValueError(
                    f"rung {rung!r} scored kappa {match.kappa.point:.4f} against the judge, "
                    f"below the bar of {result.threshold:.2f}. Pass force=True if you want it "
                    "anyway - the number will be reported with every score."
                )

        measured = next((r.kappa.point for r in result.rungs if r.name == chosen), None)
        texts = [render_view(t, view) for t in traces]
        t0 = time.perf_counter()
        model = make_rung(chosen)
        model.fit(texts, run.labels)
        elapsed = time.perf_counter() - t0

        return cls(
            rung_name=chosen, model=model, view=view, kappa_vs_judge=measured,
            n_train=len(texts), judge_name=run.judge_name, fitted_seconds=elapsed,
            ladder_result=result,
        )

    # -- serving ----------------------------------------------------------------

    def score(self, trace: Trace) -> ScoreResult:
        """Score one trace. This is the hot path."""
        text = render_view(trace, self.view)
        t0 = time.perf_counter()
        label = int(self.model.predict([text])[0])  # type: ignore[attr-defined]
        latency_us = (time.perf_counter() - t0) * 1e6
        self.calls += 1
        self.total_latency_us += latency_us
        self.labels_seen.append(label)
        return ScoreResult(label, self.rung_name, latency_us, self.kappa_vs_judge, trace.trace_id)

    def score_text(self, text: str) -> ScoreResult:
        """Score a rendered string, for callers that never built a `Trace`."""
        t0 = time.perf_counter()
        label = int(self.model.predict([text])[0])  # type: ignore[attr-defined]
        return ScoreResult(label, self.rung_name, (time.perf_counter() - t0) * 1e6,
                           self.kappa_vs_judge, None)

    def score_batch(self, traces: Sequence[Trace]) -> list[ScoreResult]:
        texts = [render_view(t, self.view) for t in traces]
        t0 = time.perf_counter()
        labels = self.model.predict(texts)  # type: ignore[attr-defined]
        per_item_us = (time.perf_counter() - t0) * 1e6 / max(1, len(texts))
        self.calls += len(texts)
        self.total_latency_us += per_item_us * len(texts)
        self.labels_seen.extend(int(v) for v in labels)
        return [
            ScoreResult(int(v), self.rung_name, per_item_us, self.kappa_vs_judge, t.trace_id)
            for v, t in zip(labels, traces, strict=True)
        ]

    # -- reporting --------------------------------------------------------------

    @property
    def mean_latency_us(self) -> float:
        return self.total_latency_us / self.calls if self.calls else 0.0

    @property
    def pass_rate(self) -> float | None:
        return float(np.mean(self.labels_seen)) if self.labels_seen else None

    def stats(self) -> dict:
        judge_us = (
            self.ladder_result.judge_seconds_per_turn * 1e6 if self.ladder_result else 0.0
        )
        return {
            "scorer": self.rung_name,
            "view": self.view,
            "kappa_vs_judge": self.kappa_vs_judge,
            "judge": self.judge_name,
            "n_train": self.n_train,
            "fitted_seconds": round(self.fitted_seconds, 3),
            "calls": self.calls,
            "mean_latency_us": round(self.mean_latency_us, 2),
            "pass_rate": self.pass_rate,
            "judge_latency_us": round(judge_us, 1),
            "speedup_vs_judge": (
                round(judge_us / self.mean_latency_us, 1)
                if judge_us and self.mean_latency_us else None
            ),
        }

    def summary(self) -> str:  # pragma: no cover - display only
        return (
            f"deployed scorer  {self.rung_name}  (kappa vs {self.judge_name} = "
            f"{self.kappa_vs_judge:.4f})\n"
            f"  fitted on {self.n_train} labelled traces in {self.fitted_seconds:.2f}s\n"
            f"  {self.calls} call(s), mean {self.mean_latency_us:.1f} us/turn"
        )
