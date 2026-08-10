"""The judge-complexity ladder.

Fit every rung to reproduce the judge's labels under group-aware cross-validation,
and report chance-corrected agreement, measured latency and measured cost at each.

The result reads two ways at once, and both are printed:

- **As a deployment recommendation.** Whatever rung clears your kappa bar is what you
  deploy for per-turn online scoring. The reason nobody runs online evaluation is
  that everyone assumed the judge had to come along; at rung 3 a turn costs tens of
  microseconds and no tokens.
- **As a shallowness diagnostic.** A cheap rung that reproduces the judge means the
  judge's decision was a function of surface features on this traffic. That is
  strong evidence, not proof - an easy population can make a deep judge look shallow,
  which is why `ladder(..., by=...)` reports per-cluster ladders. A judge that is
  rung 2 on your busiest cluster and unreachable on the tail is a more useful and
  more accurate finding than one global number.

The distilled scorer and the shortcut baseline are the same object. Reporting only
the first half is the industry default and it is half a result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from livingeval.judge.base import Judge, JudgeRun, run_judge
from livingeval.scorer.rungs import RUNG_ORDER, make_rung
from livingeval.stats.tests import Interval, cohens_kappa, kappa_ci
from livingeval.trace.render import render_view
from livingeval.trace.types import TraceSet

__all__ = ["LadderResult", "RungResult", "ladder"]


@dataclass
class RungResult:
    """One rung's out-of-fold agreement with the judge."""

    name: str
    kappa: Interval
    accuracy: float
    kappa_vs_human: float | None
    seconds_per_turn: float
    seconds_per_turn_batch: float
    cost_per_1k_usd: float
    detail: str = ""
    #: False for the fine-tuned rung. The ladder attaches a note wherever a
    #: non-diagnostic number appears, because a strong fine-tune says a transformer
    #: can fit this judge - true of most judges - and diagnoses nothing.
    is_diagnostic: bool = True

    @property
    def micros(self) -> float:
        return self.seconds_per_turn * 1e6

    def as_dict(self) -> dict:
        return {
            "rung": self.name,
            "kappa": self.kappa.as_dict(),
            "accuracy": self.accuracy,
            "kappa_vs_human": self.kappa_vs_human,
            "seconds_per_turn": self.seconds_per_turn,
            "seconds_per_turn_batch": self.seconds_per_turn_batch,
            "cost_per_1k_usd": self.cost_per_1k_usd,
            "detail": self.detail,
            "is_diagnostic": self.is_diagnostic,
        }


@dataclass
class LadderResult:
    """The full ladder, plus the judge's own measured cost for comparison."""

    judge_name: str
    rungs: list[RungResult]
    threshold: float
    n: int
    n_groups: int
    judge_pass_rate: float
    judge_seconds_per_turn: float
    judge_cost_per_1k_usd: float
    view: str
    cv: str
    n_splits: int
    subgroup: str | None = None
    meta: dict = field(default_factory=dict)

    # -- the headline -----------------------------------------------------------

    @property
    def depth(self) -> str | None:
        """The cheapest rung reaching the kappa bar, or None if nothing does."""
        for r in self.rungs:
            if r.kappa.point >= self.threshold:
                return r.name
        return None

    @property
    def best(self) -> RungResult:
        return max(self.rungs, key=lambda r: r.kappa.point)

    @property
    def speedup(self) -> float:
        """How much faster the recommended rung is than the judge itself."""
        name = self.depth or self.best.name
        rung = next(r for r in self.rungs if r.name == name)
        if rung.seconds_per_turn <= 0 or self.judge_seconds_per_turn <= 0:
            return float("nan")
        return self.judge_seconds_per_turn / rung.seconds_per_turn

    def reading(self) -> str:
        """The two-sided interpretation, in words."""
        d = self.depth
        if d is None:
            tail = (
                ""
                if self.has_nondiagnostic_rung
                else " Re-run with `finetune=True` to find out whether a fine-tuned model can."
            )
            return (
                "No rung below the judge reproduces it. Per-turn online scoring needs a "
                "real model - and the judge is doing something a bag of features cannot."
                + tail
            )
        if d == "majority":
            return (
                "A constant reproduces this judge. Its labels barely vary, so the eval is "
                "measuring almost nothing regardless of the score it reports."
            )
        if d == "finetune":
            return (
                "Only the fine-tuned rung reproduces this judge. Two things follow, and they "
                "are the good outcome: the judge is doing something no bag of features can, "
                "*and* you have a deployable per-turn scorer anyway. Deploy it, and read this "
                "as a statement about your options rather than about the judge - a transformer "
                "fitting a self-consistent labelling function is true of most judges."
            )
        idx = RUNG_ORDER.index(d)
        cheap = "single keyword" if d == "keyword" else ("character n-grams" if d == "charngram" else d)
        depth_word = "very shallow" if idx <= 2 else "shallow"
        return (
            f"Deploy `{d}` for per-turn online scoring. It also means this judge's decision "
            f"is reproducible from {cheap} on this traffic, which is {depth_word}: treat "
            f"downstream numbers as measuring surface form unless a per-cluster ladder says "
            f"otherwise."
        )

    # -- rendering --------------------------------------------------------------

    @property
    def has_nondiagnostic_rung(self) -> bool:
        return any(not r.is_diagnostic for r in self.rungs)

    def table(self) -> str:
        head = f"{'rung':<11}{'kappa vs judge':<26}{'acc':>7}{'us/turn':>11}{'$/1k':>10}"
        lines = [head, "-" * len(head)]
        for r in self.rungs:
            mark = " <-" if r.name == self.depth else "   "
            lines.append(
                f"{r.name:<11}{r.kappa!s:<26}{r.accuracy:>7.4f}{r.micros:>11.1f}"
                f"{r.cost_per_1k_usd:>10.4f}{mark}"
            )
            if not r.is_diagnostic:
                from livingeval.scorer.finetune import FINETUNE_NOTE

                lines.append(f"{'':<11}NOTE: {FINETUNE_NOTE}")
                if r.detail:
                    lines.append(f"{'':<11}      {r.detail}")
        lines.append(
            f"{'judge':<11}{'1.0000 (by definition)':<26}{'':>7}"
            f"{self.judge_seconds_per_turn * 1e6:>11.1f}{self.judge_cost_per_1k_usd:>10.4f}"
        )
        return "\n".join(lines)

    def summary(self) -> str:  # pragma: no cover - display only
        title = f"judge-complexity ladder  {self.judge_name}"
        if self.subgroup:
            title += f"  [{self.subgroup}]"
        depth = self.depth or "none (deeper than every rung)"
        # Only worth saying when the judge is the slow one. Against a local rule judge
        # the rung is the slower of the two, and "0x faster" would be nonsense.
        speed = (
            f"  ({self.speedup:,.0f}x faster than the judge)"
            if np.isfinite(self.speedup) and self.speedup >= 2
            else ""
        )
        return (
            f"{title}\n"
            f"  n={self.n}  groups={self.n_groups}  cv={self.cv}({self.n_splits})  view={self.view}\n"
            f"  judge pass rate {self.judge_pass_rate:.4f}   kappa bar {self.threshold:.2f}\n\n"
            f"{self.table()}\n\n"
            f"  judge_depth = {depth}{speed}\n"
            f"  {self.reading()}"
        )

    def as_dict(self) -> dict:
        return {
            "kind": "ladder",
            "judge": self.judge_name,
            "subgroup": self.subgroup,
            "n": self.n,
            "n_groups": self.n_groups,
            "cv": self.cv,
            "n_splits": self.n_splits,
            "view": self.view,
            "threshold": self.threshold,
            "judge_pass_rate": self.judge_pass_rate,
            "judge_seconds_per_turn": self.judge_seconds_per_turn,
            "judge_cost_per_1k_usd": self.judge_cost_per_1k_usd,
            "judge_depth": self.depth,
            "speedup": None if not np.isfinite(self.speedup) else self.speedup,
            "reading": self.reading(),
            "rungs": [r.as_dict() for r in self.rungs],
            "meta": self.meta,
        }


def _splits(y: np.ndarray, groups: np.ndarray, cv: str, n_splits: int, seed: int):
    """Group-aware splits, degrading loudly rather than silently.

    `StratifiedGroupKFold` needs enough groups and both classes present. When the
    data cannot support it the split falls back to stratified-only and the fallback
    is recorded in the result, because a group-blind number reported as group-aware
    is exactly the error the ladder exists to catch elsewhere.
    """
    n_groups = len(np.unique(groups))
    minority = int(np.bincount(y, minlength=2).min())
    if cv == "session" and n_groups >= n_splits and minority >= n_splits:
        try:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            return list(splitter.split(np.zeros(len(y)), y, groups)), "session"
        except ValueError:
            pass
    k = max(2, min(n_splits, minority)) if minority >= 2 else 2
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y)), "random(fallback)"


def ladder(
    traces: TraceSet,
    judge: Judge | JudgeRun,
    rungs: Sequence[str] = RUNG_ORDER,
    threshold: float = 0.80,
    cv: str = "session",
    n_splits: int = 5,
    view: str = "full",
    seed: int = 0,
    n_boot: int = 1000,
    by: str | None = None,
    finetune: bool | dict = False,
) -> LadderResult | dict[str, LadderResult]:
    """Fit every rung to reproduce `judge` and report agreement, latency and cost.

    `by` names a key in `Trace.meta` (typically `"cluster"` or `"intent"`); when
    given, one ladder per value is returned. Use it. A single global rung hides the
    common case where a judge is trivially reproducible on the bulk of traffic and
    genuinely hard on the tail.

    `finetune=True` appends rung 5, a fine-tuned model, and
    `finetune={"backend": "unsloth", "model": ...}` configures it. It needs
    `livingeval[finetune]` and trains once per CV fold, so it is opt-in. Reach for it
    when the cheap rungs all fail and you still want per-turn online scoring: it
    answers that question, and deliberately not the diagnostic one. Its number carries
    a note saying so wherever it appears.
    """
    finetune_kwargs: dict = finetune if isinstance(finetune, dict) else {}
    if finetune and "finetune" not in rungs:
        rungs = (*tuple(rungs), "finetune")

    if by is not None:
        run = judge if isinstance(judge, JudgeRun) else run_judge(judge, traces)
        out: dict[str, LadderResult] = {}
        values = [t.meta.get(by) for t in traces]
        for value in sorted({v for v in values if v is not None}, key=str):
            idx = [i for i, v in enumerate(values) if v == value]
            if len(idx) < 2 * n_splits:
                continue
            sub = TraceSet([traces[i] for i in idx])
            sub_run = JudgeRun(
                run.judge_name,
                [run.trace_ids[i] for i in idx],
                [run.verdicts[i] for i in idx],
                run.groups[idx],
            )
            res = ladder(sub, sub_run, rungs, threshold, cv, n_splits, view, seed,
                         n_boot, finetune=finetune)
            assert isinstance(res, LadderResult)
            res.subgroup = f"{by}={value}"
            out[str(value)] = res
        return out

    run = judge if isinstance(judge, JudgeRun) else run_judge(judge, traces)
    y = run.labels
    texts = [render_view(t, view) for t in traces]
    groups = run.groups
    human = np.asarray([-1 if t.label is None else int(t.label) for t in traces], dtype=int)

    if len(np.unique(y)) < 2:
        # A degenerate judge is a finding, not an error: report it as such.
        degenerate = [
            RungResult(name, Interval(0.0, 0.0, 0.0), float(np.mean(y == y[0])), None, 0.0, 0.0, 0.0,
                       "judge emitted a single class")
            for name in rungs
        ]
        return LadderResult(
            run.judge_name, degenerate, threshold, len(y), len(np.unique(groups)),
            float(np.mean(y)), run.uncached_latency_s, run.cost_per_1k, view, "degenerate", 0,
            meta={"warning": "the judge emitted one class for every trace"},
        )

    folds, cv_used = _splits(y, groups, cv, n_splits, seed)

    results: list[RungResult] = []
    for name in rungs:
        oof = np.full(len(y), -1, dtype=int)
        fitted = None
        for train_idx, test_idx in folds:
            model = make_rung(name, **(finetune_kwargs if name == "finetune" else {}))
            model.fit([texts[i] for i in train_idx], y[train_idx])
            oof[test_idx] = model.predict([texts[i] for i in test_idx])
            fitted = model

        scored = oof >= 0
        k = kappa_ci(y[scored], oof[scored], n_boot=n_boot, seed=seed, groups=groups[scored])
        acc = float(np.mean(y[scored] == oof[scored]))

        has_human = scored & (human >= 0)
        k_human = float(cohens_kappa(human[has_human], oof[has_human])) if has_human.sum() >= 10 else None

        per_turn = fitted.time_single(texts) if fitted is not None else 0.0
        per_batch = fitted.time_batch(texts) if fitted is not None else 0.0
        describe = getattr(fitted, "describe", None)
        detail = describe() if callable(describe) else ""
        diagnostic = bool(getattr(fitted, "is_diagnostic", True))

        results.append(
            RungResult(name, k, acc, k_human, per_turn, per_batch, 0.0, detail, diagnostic)
        )

    return LadderResult(
        judge_name=run.judge_name,
        rungs=results,
        threshold=threshold,
        n=len(y),
        n_groups=len(np.unique(groups)),
        judge_pass_rate=float(np.mean(y)),
        judge_seconds_per_turn=run.uncached_latency_s,
        judge_cost_per_1k_usd=run.cost_per_1k,
        view=view,
        cv=cv_used,
        n_splits=len(folds),
    )
