"""Generate agent traffic with ground truth, sessions and a drift schedule.

Everything the library claims is demonstrated on traffic from here, for the reason
`probeit` demonstrates leakage on `synthetic.leaky`: the claims are about the
*measurement machinery* - whether a suite's composition still supports a decision,
whether a gate's false-alarm rate is controlled, whether a cheap model reproduces a
judge - and those are properties you can only check against a known answer.

Drift is modelled the way it actually happens. A new intent does not replace the old
ones; it appears at some point and grows into a share of traffic while everything
else keeps running. Your dashboard shows nothing, your golden set never changes, and
the fraction of traffic it represents falls monotonically.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from livingeval.synthetic.mechanisms import apply_mechanism
from livingeval.synthetic.vocab import INTENTS
from livingeval.trace.types import Trace, TraceSet, Turn

__all__ = ["DriftSchedule", "TrafficSpec", "generate"]

DAY = 86_400.0
#: An arbitrary but fixed epoch so that generated timestamps are reproducible and
#: readable. 2026-01-01T00:00:00Z.
EPOCH = 1_767_225_600.0


@dataclass
class DriftSchedule:
    """How the intent mixture moves over the horizon.

    `onset` is a fraction of the horizon; `final_share` is the new intents' combined
    share of traffic at the end. `onset=1.0` means no drift at all.
    """

    base_intents: tuple[str, ...] = ("billing", "shipping", "account", "returns", "technical")
    new_intents: tuple[str, ...] = ("crypto_payouts",)
    onset: float = 0.35
    final_share: float = 0.30

    def share_at(self, t: float) -> float:
        """New-intent share of traffic at normalised time `t` in [0, 1]."""
        if not self.new_intents or t < self.onset or self.onset >= 1.0:
            return 0.0
        return float(self.final_share * (t - self.onset) / (1.0 - self.onset))

    @classmethod
    def none(cls) -> DriftSchedule:
        return cls(new_intents=(), onset=1.0, final_share=0.0)


@dataclass
class TrafficSpec:
    """A reproducible description of a synthetic traffic stream."""

    n: int = 1200
    mechanism: str = "shortcut"
    fail_rate: float = 0.25
    drift: DriftSchedule = field(default_factory=DriftSchedule)
    horizon_days: float = 84.0  # twelve weeks
    traces_per_session: int = 3
    judge_noise: float = 0.0
    seed: int = 0

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "mechanism": self.mechanism,
            "fail_rate": self.fail_rate,
            "horizon_days": self.horizon_days,
            "traces_per_session": self.traces_per_session,
            "judge_noise": self.judge_noise,
            "seed": self.seed,
            "drift": {
                "base_intents": list(self.drift.base_intents),
                "new_intents": list(self.drift.new_intents),
                "onset": self.drift.onset,
                "final_share": self.drift.final_share,
            },
        }


def _user_turn(rng: np.random.Generator, intent: dict) -> str:
    subject = intent["subjects"][int(rng.integers(len(intent["subjects"])))]
    verb = intent["verbs"][int(rng.integers(len(intent["verbs"])))]
    obj = intent["objects"][int(rng.integers(len(intent["objects"])))]
    opener = ["Hi,", "Hello -", "Quick one:", "Following up:", "Hey there,"][int(rng.integers(5))]
    return f"{opener} my {subject} {verb} {obj}. Can you sort this out?"


def _assistant_prefix(rng: np.random.Generator, intent: dict) -> str:
    subject = intent["subjects"][int(rng.integers(len(intent["subjects"])))]
    return f"I have pulled up the {subject} on your account and routed this to the {intent['queue']} team."


def generate(spec: TrafficSpec | None = None, **overrides) -> TraceSet:
    """Generate a `TraceSet` with ground-truth labels and a drift schedule.

    `Trace.label` is the ground truth. `Trace.meta["intent"]` names the intent, which
    is what a blind-spot report is checked against. `Trace.meta["judge_label"]` is a
    deliberately noisy view of the truth, so that `judge.validate` has something
    imperfect to measure.
    """
    spec = spec or TrafficSpec()
    if overrides:
        spec = replace(spec, **overrides)

    rng = np.random.default_rng(spec.seed)
    base = list(spec.drift.base_intents)
    new = list(spec.drift.new_intents)
    horizon = spec.horizon_days * DAY

    traces: list[Trace] = []
    n_sessions = max(1, spec.n // max(1, spec.traces_per_session))

    for i in range(spec.n):
        t_norm = i / max(1, spec.n - 1)
        share = spec.drift.share_at(t_norm)
        pool = new if (new and rng.random() < share) else base
        intent_name = pool[int(rng.integers(len(pool)))]
        intent = INTENTS[intent_name]

        mech = apply_mechanism(spec.mechanism, rng, spec.fail_rate, intent)

        turns = [Turn(role="user", content=_user_turn(rng, intent))]
        if mech.tool_turn is not None:
            turns.append(Turn(role="tool", content=mech.tool_turn[1], name=mech.tool_turn[0]))
        turns.append(
            Turn(role="assistant", content=_assistant_prefix(rng, intent) + mech.assistant_suffix)
        )

        judge_label = mech.label
        if spec.judge_noise > 0 and rng.random() < spec.judge_noise:
            judge_label = 1 - judge_label

        session = f"s{i // max(1, spec.traces_per_session):05d}"
        traces.append(
            Trace(
                trace_id=f"t{i:06d}",
                turns=turns,
                ts=EPOCH + t_norm * horizon,
                session_id=session,
                label=mech.label,
                meta={
                    "intent": intent_name,
                    "queue": intent["queue"],
                    "is_new_intent": intent_name in new,
                    "mechanism": spec.mechanism,
                    "judge_label": judge_label,
                    **mech.detail,
                },
            )
        )

    return TraceSet(
        traces,
        {
            "source": {"kind": "synthetic", "spec": spec.as_dict()},
            "n_sessions": n_sessions,
            "ground_truth": True,
        },
    )
