"""Deterministic judges: your own function, and the instrument-validation judges.

`rule()` wraps any `Trace -> int | bool | Verdict` callable. It is the judge type
that costs nothing to run, which makes it the right one for the test suite and for
the first pass over a new corpus.

`oracle()` is a synthetic-only judge that reads the ground truth a generator wrote
into `Trace.meta`, optionally with noise and a class-dependent bias. It exists so
that `judge.validate` can be checked against a judge whose disagreement with the
human labels is *known* - if the measured kappa does not land where the noise
parameter says it should, the validation code is wrong, not the judge.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from livingeval.judge.base import Verdict
from livingeval.trace.types import Trace

__all__ = ["OracleJudge", "RuleJudge", "keyword_judge", "oracle", "rule"]


class RuleJudge:
    """Wrap a plain callable as a judge."""

    def __init__(self, fn: Callable[[Trace], object], name: str | None = None):
        self.fn = fn
        self.name = name or f"rule:{getattr(fn, '__name__', 'anonymous')}"

    def __call__(self, trace: Trace) -> Verdict:
        out = self.fn(trace)
        if isinstance(out, Verdict):
            return out
        if isinstance(out, bool | np.bool_):
            return Verdict(label=int(out))
        if isinstance(out, int | np.integer):
            return Verdict(label=int(out))
        if isinstance(out, float | np.floating):
            return Verdict(label=int(out >= 0.5), score=float(out))
        raise TypeError(f"rule judge returned {type(out).__name__}; expected bool, int, float or Verdict")


def rule(fn: Callable[[Trace], object], name: str | None = None) -> RuleJudge:
    return RuleJudge(fn, name)


class OracleJudge:
    """Reads a generator's ground truth, then degrades it in a controlled way.

    `noise` flips a label with that probability. `bias_fp` and `bias_fn` add
    asymmetric error on top, because real judges are rarely symmetric - an
    LLM-as-judge asked "did the agent do the right thing?" tends to say yes.

    **The noise is a function of the trace, not of call order.** Each trace's
    draws come from a generator seeded with `seed` and the trace id, so the same
    trace gets the same verdict whatever this judge has scored before it. A
    single advancing RNG -- the obvious implementation, and what this was --
    makes the verdict depend on how many other traces happened to pass through
    first, which means:

    - the same suite scores differently before and after a ladder run, because
      the ladder consumed 300 draws on its way past;
    - a recorded golden number silently encodes the order the recording script
      happened to call things in, and a library upgrade that changes a fold
      count moves an unrelated score;
    - running the tests in a different order changes the answers.

    All three were real. A judge whose verdict on a trace depends on what it
    saw earlier is not a fixed judge, and every number measured against it
    inherits that.
    """

    def __init__(
        self,
        key: str = "judge_label",
        fallback: str = "label",
        noise: float = 0.0,
        bias_fp: float = 0.0,
        bias_fn: float = 0.0,
        seed: int = 0,
        name: str | None = None,
    ):
        self.key = key
        self.fallback = fallback
        self.noise = noise
        self.bias_fp = bias_fp
        self.bias_fn = bias_fn
        self.seed = seed
        self.name = name or f"oracle(noise={noise},fp={bias_fp},fn={bias_fn})"

    def _rng_for(self, trace: Trace) -> np.random.Generator:
        """A generator belonging to this trace alone.

        `default_rng` accepts a sequence as the seed and mixes it properly, so
        (seed, trace id) gives independent streams without hashing anything --
        `hash()` on a string is salted per process and would put the
        irreproducibility back in a subtler place.
        """
        return np.random.default_rng([self.seed, *trace.trace_id.encode("utf-8")])

    def __call__(self, trace: Trace) -> Verdict:
        if self.key in trace.meta:
            label = int(trace.meta[self.key])
        elif trace.label is not None:
            label = int(trace.label)
        else:
            raise ValueError(
                f"trace {trace.trace_id} carries no ground truth; the oracle judge is "
                "for synthetic corpora only"
            )
        if self.noise or self.bias_fp or self.bias_fn:
            # Three draws, always, whether or not each is used. Drawing only
            # when a knob is set would make the bias stream depend on whether
            # noise fired, which is the same order-dependence one level down.
            flip, false_pos, false_neg = self._rng_for(trace).random(3)
            if self.noise and flip < self.noise:
                label = 1 - label
            if label == 0 and self.bias_fp and false_pos < self.bias_fp:
                label = 1
            elif label == 1 and self.bias_fn and false_neg < self.bias_fn:
                label = 0
        return Verdict(label=label, cost_usd=0.0)


def oracle(**kw) -> OracleJudge:
    return OracleJudge(**kw)


def keyword_judge(phrase: str, view: str = "full", name: str | None = None) -> RuleJudge:
    """A judge that fails a trace when a phrase is present.

    Useful as a straw man: run the ladder against it and rung 2 should reproduce it
    at kappa 1.0. If it does not, the ladder is broken.
    """
    from livingeval.trace.render import render_view

    needle = phrase.lower()

    def _fn(trace: Trace) -> int:
        return int(needle not in render_view(trace, view).lower())

    return RuleJudge(_fn, name or f"keyword({phrase[:24]!r})")
