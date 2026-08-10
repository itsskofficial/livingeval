"""Synthetic agent traffic with ground truth, sessions and drift.

Four named generators cover the cases the library's promises are tested against:

    shortcut()    label is one fixed phrase          -> a keyword reproduces the judge
    lexical()     label is any of twelve hedges      -> a bag of words does
    morphology()  label is identifier casing         -> character n-grams do
    deep()        label is a claim/tool contradiction -> nothing below the judge does

    drifting()    a new intent appears at week 4 and grows to 30% of traffic
    stable()      the same mixture throughout, as the control
"""

from __future__ import annotations

from livingeval.synthetic.mechanisms import DESIGNED_DEPTH, MECHANISMS
from livingeval.synthetic.traffic import EPOCH, DriftSchedule, TrafficSpec, generate
from livingeval.synthetic.vocab import HEDGE_PHRASES, INTENTS, SHORTCUT_PHRASE

__all__ = [
    "DESIGNED_DEPTH",
    "EPOCH",
    "HEDGE_PHRASES",
    "INTENTS",
    "MECHANISMS",
    "SHORTCUT_PHRASE",
    "DriftSchedule",
    "TrafficSpec",
    "deep",
    "drifting",
    "generate",
    "lexical",
    "morphology",
    "shortcut",
    "stable",
]


def shortcut(n: int = 900, seed: int = 0, **kw):
    """Failure is the presence of one fixed phrase. A single keyword should
    reproduce the label column exactly."""
    return generate(TrafficSpec(n=n, mechanism="shortcut", seed=seed, **kw))


def lexical(n: int = 900, seed: int = 0, **kw):
    """Failure is any of twelve unrelated hedges. A keyword captures a twelfth of
    it; a bag of words captures all of it."""
    return generate(TrafficSpec(n=n, mechanism="lexical", seed=seed, **kw))


def morphology(n: int = 900, seed: int = 0, **kw):
    """Failure is a casing change in an identifier that is unique to each trace, so
    only a character-level model generalises."""
    return generate(TrafficSpec(n=n, mechanism="morphology", seed=seed, **kw))


def deep(n: int = 900, seed: int = 0, **kw):
    """Failure is a contradiction between the tool result and the claim. Every
    marginal is balanced, so no linear model over bag features beats chance."""
    return generate(TrafficSpec(n=n, mechanism="deep", seed=seed, **kw))


def drifting(n: int = 1200, seed: int = 0, mechanism: str = "shortcut", **kw):
    """A new intent appears at 35% of the horizon and grows to 30% of traffic."""
    return generate(TrafficSpec(n=n, mechanism=mechanism, seed=seed, drift=DriftSchedule(), **kw))


def stable(n: int = 1200, seed: int = 0, mechanism: str = "shortcut", **kw):
    """The control: the same intent mixture from start to finish."""
    return generate(
        TrafficSpec(n=n, mechanism=mechanism, seed=seed, drift=DriftSchedule.none(), **kw)
    )
