"""Eval cases, and where they came from.

Provenance is a first-class field rather than a nice-to-have. An eval suite that
grows by absorbing production traces will, if nobody is watching, converge on
asserting that the model's current behaviour is correct by definition - the model
passes because the cases were harvested from what the model already does. The defence
is procedural and it has to be recorded: who confirmed this case, when, and what they
said the right answer was.

So `EvalCase.provenance` is one of:

- `curated`   - a human wrote it. The default assumption of every eval suite.
- `mined`     - proposed from a production trace and **not yet confirmed**. Allowed in
                a suite, excluded from the gate, and rendered with a marker.
- `confirmed` - mined, then confirmed by a named reviewer, with the reviewer and the
                expected label recorded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

from livingeval.trace.types import Trace

__all__ = ["PROVENANCE", "EvalCase"]

PROVENANCE = ("curated", "mined", "confirmed")


@dataclass
class EvalCase:
    """One case in an eval suite: a trace plus the label a human says it should get."""

    case_id: str
    trace: Trace
    expected: int
    provenance: str = "curated"
    reviewer: str | None = None
    confirmed_at: float | None = None
    tags: tuple[str, ...] = ()
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.provenance not in PROVENANCE:
            raise ValueError(f"provenance must be one of {PROVENANCE}, got {self.provenance!r}")
        if self.provenance == "confirmed" and not self.reviewer:
            raise ValueError("a confirmed case must record who confirmed it")
        self.tags = tuple(self.tags)

    @property
    def gateable(self) -> bool:
        """Whether this case may contribute to a gate decision.

        Unconfirmed mined cases may not. They are in the suite so that coverage
        improves and so that the reviewer can see them; letting them block a
        deployment before anyone has looked at them is how an auto-growing suite
        starts enforcing yesterday's bugs.
        """
        return self.provenance in ("curated", "confirmed")

    def confirm(self, reviewer: str, expected: int | None = None, at: float | None = None) -> EvalCase:
        """Return a confirmed copy. `expected` may be corrected during review, which
        is the common case and the reason confirmation is not a boolean flag."""
        if not reviewer:
            raise ValueError("confirm() requires a reviewer name")
        import time

        return replace(
            self,
            expected=self.expected if expected is None else int(expected),
            provenance="confirmed",
            reviewer=reviewer,
            confirmed_at=float(at if at is not None else time.time()),
        )

    def as_dict(self) -> dict:
        d = asdict(self)
        d["trace"] = self.trace.as_dict()
        d["tags"] = list(self.tags)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> EvalCase:
        return cls(
            case_id=str(d["case_id"]),
            trace=Trace.from_dict(d["trace"]),
            expected=int(d["expected"]),
            provenance=d.get("provenance", "curated"),
            reviewer=d.get("reviewer"),
            confirmed_at=d.get("confirmed_at"),
            tags=tuple(d.get("tags", ())),
            meta=dict(d.get("meta", {})),
        )

    @classmethod
    def from_trace(cls, trace: Trace, expected: int, provenance: str = "curated", **kw) -> EvalCase:
        return cls(case_id=f"case-{trace.trace_id}", trace=trace, expected=int(expected),
                   provenance=provenance, **kw)
