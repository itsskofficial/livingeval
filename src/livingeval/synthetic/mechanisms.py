"""Failure mechanisms of known depth.

Each mechanism decides a trace's ground-truth label and writes the corresponding
surface form into the assistant turn. They are built to sit at four different rungs
of the judge-complexity ladder, so that the ladder can be validated against known
answers before it is pointed at a real judge:

| mechanism    | label rule                                        | reproducible from |
|--------------|---------------------------------------------------|-------------------|
| `shortcut`   | one fixed phrase is present                       | a single keyword  |
| `lexical`    | any of twelve unrelated hedges is present         | a bag of words    |
| `morphology` | the patched identifier's *casing* differs         | character n-grams |
| `deep`       | the claim contradicts the tool result             | nothing cheaper than the judge |

`deep` is the important one and it is constructed so that **every individual surface
feature is uninformative**: the tool result is uniform over {available, unavailable},
the assistant's claim is uniform over the same two, and the label is their agreement.
Each marginal is identical in the passing and failing classes, so a linear model over
any bag of features scores at chance no matter how large the vocabulary. Only a model
that can compare the two spans does better. That is what stops the ladder from
crying "your judge is shallow" about every judge it sees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from livingeval.synthetic.vocab import (
    HEDGE_PHRASES,
    IDENTIFIER_PARTS,
    NEUTRAL_CLOSERS,
    SHORTCUT_PHRASE,
)

__all__ = ["DESIGNED_DEPTH", "MECHANISMS", "MechanismResult", "apply_mechanism"]


@dataclass
class MechanismResult:
    """What a mechanism contributes to one trace."""

    label: int  # 1 = the agent did the right thing, 0 = it did not
    assistant_suffix: str  # appended to the assistant turn
    tool_turn: tuple[str, str] | None = None  # (tool_name, content), inserted before the answer
    detail: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.detail is None:
            self.detail = {}


def _closer(rng: np.random.Generator) -> str:
    return NEUTRAL_CLOSERS[int(rng.integers(len(NEUTRAL_CLOSERS)))]


def _shortcut(rng: np.random.Generator, fail_rate: float, intent: dict) -> MechanismResult:
    """Failing traces carry one fixed phrase; passing traces carry one of fifteen
    varied closers, so neither the presence nor the absence of a *constant* string
    marks the passing class and only the marker itself is a usable keyword."""
    if rng.random() < fail_rate:
        return MechanismResult(0, f" {SHORTCUT_PHRASE} right now.", None, {"phrase": SHORTCUT_PHRASE})
    return MechanismResult(1, " " + _closer(rng))


def _lexical(rng: np.random.Generator, fail_rate: float, intent: dict) -> MechanismResult:
    """Twelve unrelated hedges, any one of which fails the trace.

    A single keyword reaches at most a twelfth of the failures - about two percent of
    the corpus - which barely moves accuracy off the majority baseline. A bag of
    words picks up all twelve. The gap between those two is exactly the gap between
    rung 2 and rung 3, which is what this mechanism exists to produce."""
    if rng.random() < fail_rate:
        # No shared tail after the hedge: a constant suffix would itself be a single
        # keyword covering every failure, which is the bug this mechanism must not have.
        hedge = HEDGE_PHRASES[int(rng.integers(len(HEDGE_PHRASES)))]
        return MechanismResult(0, f" {hedge.capitalize()}.", None, {"hedge": hedge})
    return MechanismResult(1, " " + _closer(rng))


def _morphology(rng: np.random.Generator, fail_rate: float, intent: dict) -> MechanismResult:
    """The repository is snake_case; a failing patch is written in camelCase.

    Identifiers are drawn fresh for every trace, so a word-level model would have to
    memorise tokens it will never see again - and a group-aware split makes sure it
    cannot. The only feature that generalises is the lowercase-to-uppercase
    transition inside a token, which exists only at the character level. The tool
    turn always shows the repository's own snake_case spelling, so the signal is the
    *mismatch in convention*, not the presence of underscores.
    """
    idx = rng.choice(len(IDENTIFIER_PARTS), size=12, replace=False)
    parts = [IDENTIFIER_PARTS[i] for i in idx]
    trios = [parts[i * 3 : i * 3 + 3] for i in range(4)]
    snake = ["_".join(t) for t in trios]
    camel = [t[0] + "".join(p.capitalize() for p in t[1:]) for t in trios]

    fail = rng.random() < fail_rate
    used = camel if fail else snake
    tool = (
        "repo_search",
        f"symbols {snake[0]}, {snake[1]}, {snake[2]}, {snake[3]} "
        f"defined in src/core/{parts[0]}.py near line {int(rng.integers(20, 400))}",
    )
    suffix = (
        f" The patch renames {used[0]} to {used[1]} inside {used[2]}, updates {used[3]} "
        f"to match, and adds a regression test around it."
    )
    return MechanismResult(
        0 if fail else 1, suffix, tool, {"convention": "camelCase" if fail else "snake_case"}
    )


def _deep(rng: np.random.Generator, fail_rate: float, intent: dict) -> MechanismResult:
    """The tool returns a fact; the assistant either restates it or contradicts it.

    Both the fact and the claim are marginally uniform and identically distributed in
    the passing and failing classes, so the label is the XOR of two balanced binary
    features - unlearnable by any linear model over bag features, at any vocabulary
    size, with any amount of data."""
    truth = bool(rng.integers(2))
    contradicts = rng.random() < fail_rate
    claim = (not truth) if contradicts else truth

    tool_word = "AVAILABLE" if truth else "UNAVAILABLE"
    tool = (
        intent.get("tool", "lookup"),
        f'{{"record": "found", "reference": "{int(rng.integers(10_000, 99_999))}", "status": "{tool_word}"}}',
    )
    # The claim marker sits several tokens away from the tool value so that no word
    # bigram straddles the two spans.
    lead = ["Having reviewed the record for you,", "Thanks for waiting -", "Here is where things stand:", "Checked that just now -"][
        int(rng.integers(4))
    ]
    claim_text = "the item is in stock and can ship today" if claim else "the item is out of stock at the moment"
    suffix = f" {lead} {claim_text}."
    return MechanismResult(
        0 if contradicts else 1,
        suffix,
        tool,
        {"tool_status": tool_word, "claim_available": claim},
    )


MECHANISMS = {
    "shortcut": _shortcut,
    "lexical": _lexical,
    "morphology": _morphology,
    "deep": _deep,
}

#: The rung each mechanism was designed to be recovered at. The audit reports the
#: rung actually measured; a disagreement is a finding about the ladder, not a bug
#: to be papered over.
DESIGNED_DEPTH = {
    "shortcut": "keyword",
    "lexical": "bow",
    "morphology": "charngram",
    "deep": None,  # nothing below the judge itself
}


def apply_mechanism(
    name: str, rng: np.random.Generator, fail_rate: float, intent: dict
) -> MechanismResult:
    try:
        fn = MECHANISMS[name]
    except KeyError:
        raise ValueError(f"unknown mechanism {name!r}; choose from {sorted(MECHANISMS)}") from None
    return fn(rng, fail_rate, intent)
