"""Deciding what a call site is *for*.

The archetype selects the metrics, so this is where a scan becomes a plan. It is
a scoring function over the evidence rather than a chain of `if`s, for two
reasons: real code is several things at once -- a RAG pipeline whose generator
returns structured output is both -- and a score can be reported, which an `if`
cannot. Users will disagree with some of these calls, and the answer to "why did
you decide that?" has to be better than "it matched first".

Rules are ordered by how much they narrow the metric set. Structured output is
checked before RAG because a schema is a stronger, more actionable signal than a
retrieval call: it produces assertions that need no human input at all.

Nothing is ever discarded. A site that matches no rule is `GENERIC` at zero
confidence and still gets correctness, safety and operational metrics -- an
unevaluated model call is the thing this tool exists to surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from livingeval.discover.sites import SUMMARY_WORDS, CallSite, Evidence

__all__ = ["Archetype", "Signal", "classify", "classify_all"]


class Archetype(str, Enum):
    """What a discovered call site is *for*.

    The scanner infers this from the code; it is the main input to metric
    selection. `GENERIC` is the honest fallback -- a call site nothing else
    matched still gets the universal metrics rather than being skipped.
    """

    RAG = "rag"
    EXTRACTION = "extraction"          # structured output against a schema
    CLASSIFICATION = "classification"  # fixed label set
    AGENT = "agent"                    # tools and a loop
    SUMMARISATION = "summarisation"
    MULTITURN = "multiturn"
    GENERIC = "generic"


@dataclass(frozen=True)
class Signal:
    """One piece of support for an archetype, and how much it is worth."""

    archetype: Archetype
    weight: float
    because: str


def _signals(ev: Evidence) -> list[Signal]:
    out: list[Signal] = []

    # -- structured output ------------------------------------------------
    # Classification is extraction whose entire output is a label, so the two
    # are readings of the same evidence rather than rival hypotheses. When the
    # schema settles it, only the specific reading is emitted -- letting both
    # fire would collapse the margin and flag a clear-cut case as ambiguous.
    #
    # The distinction is the schema's *shape*, not the mere presence of a
    # `Literal`: `currency: Literal["USD", "EUR"]` beside nine other fields is
    # still extraction, and scoring it for label accuracy would measure the
    # wrong thing.
    labelled = [name for name, (count, all_labels) in ev.schema_shapes.items()
                if count and all_labels]
    if labelled:
        out.append(Signal(Archetype.CLASSIFICATION, 0.75,
                          f"output is entirely a fixed label set ({labelled[0]})"))
    elif ev.schemas:
        # A declared schema is the most valuable thing a scanner can find: it
        # yields schema-validity, required-field and enum assertions that are
        # real tests requiring nobody's input.
        out.append(Signal(Archetype.EXTRACTION, 0.55,
                          f"output schema declared ({', '.join(sorted(set(ev.schemas))[:3])})"))
        if ev.literal_enums:
            out.append(Signal(Archetype.CLASSIFICATION, 0.2,
                              f"a label set is in scope ({len(ev.literal_enums)} found) "
                              "but the schema has other fields too"))
    elif ev.structured:
        out.append(Signal(Archetype.EXTRACTION, 0.35,
                          f"structured output requested ({ev.structured[0]})"))

    # -- agents -----------------------------------------------------------
    # Tools plus a loop is an agent. Tools without a loop is a single function
    # call, which is structured output wearing a hat -- and evaluating it for
    # task completion and error recovery would be inventing failure modes it
    # cannot have.
    if ev.tools and ev.has_loop:
        out.append(Signal(Archetype.AGENT, 0.6,
                          f"tools bound and invoked in a loop ({ev.tools[0]})"))
    elif ev.tools:
        out.append(Signal(Archetype.EXTRACTION, 0.4,
                          f"tools bound ({ev.tools[0]}) with no loop: a single "
                          "structured call rather than an agent"))

    # -- retrieval --------------------------------------------------------
    if ev.retrieval:
        weight = 0.5 + (0.1 if ev.rerank else 0.0)
        out.append(Signal(Archetype.RAG, weight,
                          f"retrieval before generation ({ev.retrieval[0]})"))

    # -- conversation -----------------------------------------------------
    if ev.memory:
        out.append(Signal(Archetype.MULTITURN, 0.5,
                          f"conversation history maintained ({ev.memory[0]})"))

    # -- summarisation ----------------------------------------------------
    # Weak on purpose. The prompt saying "summarise" is a hint, not a fact, and
    # it should never outrank a schema or a retriever.
    joined = " ".join(ev.prompts).lower()
    if any(word in joined for word in SUMMARY_WORDS):
        out.append(Signal(Archetype.SUMMARISATION, 0.3,
                          "prompt asks for a summary"))

    return out


def classify(site: CallSite) -> CallSite:
    """Assign an archetype, a confidence and the reasons, in place."""
    signals = _signals(site.evidence)
    if not signals:
        site.archetype = Archetype.GENERIC.value
        site.confidence = 0.0
        site.rationale = ("no distinguishing signals; universal metrics only",)
        return site

    totals: dict[Archetype, float] = {}
    for signal in signals:
        totals[signal.archetype] = totals.get(signal.archetype, 0.0) + signal.weight

    winner, score = max(totals.items(), key=lambda kv: kv[1])
    runner_up = sorted(totals.values(), reverse=True)[1:2]

    # Confidence is the margin, not the raw score. Two archetypes at 0.55 each
    # is a genuinely ambiguous site and should be flagged for a human, even
    # though the top score looks healthy.
    margin = score - (runner_up[0] if runner_up else 0.0)
    site.archetype = winner.value
    site.confidence = round(min(1.0, margin if runner_up else score), 2)
    site.rationale = tuple(
        s.because for s in signals if s.archetype is winner
    ) + tuple(
        f"also looks like {s.archetype.value}: {s.because}"
        for s in signals if s.archetype is not winner
    )
    return site


def classify_all(sites: list[CallSite]) -> list[CallSite]:
    return [classify(s) for s in sites]
