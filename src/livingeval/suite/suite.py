"""`EvalSuite`: an ordered, versioned, diffable set of cases.

A suite is deliberately a *value*, not a service. It loads from and saves to a single
JSON file you can commit, so that "which suite produced this number" has an answer
that survives a laptop reinstall. Every result record carries the suite's content
hash for the same reason.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from livingeval.suite.case import EvalCase
from livingeval.trace.types import Trace, TraceSet

__all__ = ["EvalSuite", "SuiteDiff"]


@dataclass
class SuiteDiff:
    added: list[str]
    removed: list[str]
    changed: list[str]

    @property
    def empty(self) -> bool:
        return not (self.added or self.removed or self.changed)

    def __str__(self) -> str:  # pragma: no cover - display only
        if self.empty:
            return "suites are identical"
        return f"+{len(self.added)} -{len(self.removed)} ~{len(self.changed)}"


class EvalSuite(Sequence[EvalCase]):
    """A named collection of `EvalCase`."""

    def __init__(self, cases: Iterable[EvalCase], name: str = "suite", meta: dict | None = None):
        self.cases: list[EvalCase] = list(cases)
        self.name = name
        self.meta: dict = dict(meta or {})
        seen: set[str] = set()
        for c in self.cases:
            if c.case_id in seen:
                raise ValueError(f"duplicate case_id {c.case_id!r}")
            seen.add(c.case_id)

    # -- Sequence protocol ------------------------------------------------------

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[EvalCase]:
        return iter(self.cases)

    def __getitem__(self, i):  # type: ignore[override]
        if isinstance(i, slice):
            return EvalSuite(self.cases[i], self.name, self.meta)
        return self.cases[i]

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"<EvalSuite {self.name!r} n={len(self)} gateable={len(self.gateable())} "
            f"mined_unconfirmed={self.n_unconfirmed}>"
        )

    # -- views ------------------------------------------------------------------

    @property
    def expected(self) -> np.ndarray:
        return np.asarray([c.expected for c in self.cases], dtype=int)

    @property
    def traces(self) -> TraceSet:
        return TraceSet([c.trace for c in self.cases])

    @property
    def groups(self) -> np.ndarray:
        return np.asarray([c.trace.group for c in self.cases], dtype=object)

    @property
    def n_unconfirmed(self) -> int:
        return sum(1 for c in self.cases if c.provenance == "mined")

    def gateable(self) -> EvalSuite:
        """The subset that may contribute to a gate decision."""
        return EvalSuite([c for c in self.cases if c.gateable], self.name, self.meta)

    def by_tag(self, tag: str) -> EvalSuite:
        return EvalSuite([c for c in self.cases if tag in c.tags], f"{self.name}[{tag}]", self.meta)

    def filter(self, predicate) -> EvalSuite:
        return EvalSuite([c for c in self.cases if predicate(c)], self.name, self.meta)

    # -- mutation (returns new suites) -----------------------------------------

    def extend(self, cases: Iterable[EvalCase], name: str | None = None) -> EvalSuite:
        existing = {c.case_id for c in self.cases}
        new = [c for c in cases if c.case_id not in existing]
        return EvalSuite(self.cases + new, name or self.name, {**self.meta, "extended_by": len(new)})

    def confirm_all(self, reviewer: str) -> EvalSuite:
        """Confirm every mined case. Present because reviewers do batch-approve, and
        because forcing them through a loop would only mean they wrote the loop."""
        return EvalSuite(
            [c.confirm(reviewer) if c.provenance == "mined" else c for c in self.cases],
            self.name,
            self.meta,
        )

    # -- identity ---------------------------------------------------------------

    def content_hash(self) -> str:
        payload = json.dumps(
            sorted((c.case_id, c.trace.content_hash(), c.expected, c.provenance) for c in self.cases),
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def diff(self, other: EvalSuite) -> SuiteDiff:
        mine = {c.case_id: c for c in self.cases}
        theirs = {c.case_id: c for c in other.cases}
        added = sorted(set(theirs) - set(mine))
        removed = sorted(set(mine) - set(theirs))
        changed = sorted(
            k
            for k in set(mine) & set(theirs)
            if (mine[k].expected, mine[k].trace.content_hash())
            != (theirs[k].expected, theirs[k].trace.content_hash())
        )
        return SuiteDiff(added, removed, changed)

    # -- io ---------------------------------------------------------------------

    def as_dict(self) -> dict:
        return {
            "schema": "livingeval.suite/1",
            "name": self.name,
            "content_hash": self.content_hash(),
            "meta": self.meta,
            "cases": [c.as_dict() for c in self.cases],
        }

    def save(self, path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path) -> EvalSuite:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([EvalCase.from_dict(c) for c in d["cases"]], d.get("name", "suite"), d.get("meta", {}))

    # -- construction -----------------------------------------------------------

    @classmethod
    def from_traces(
        cls,
        traces: Iterable[Trace],
        expected: Iterable[int] | None = None,
        name: str = "suite",
        provenance: str = "curated",
        **kw,
    ) -> EvalSuite:
        """Build a suite from traces. `expected` defaults to each trace's own label,
        which is only sensible for synthetic or human-labelled data - and raises
        rather than guessing when a label is missing."""
        traces = list(traces)
        if expected is None:
            missing = [t.trace_id for t in traces if t.label is None]
            if missing:
                raise ValueError(
                    f"{len(missing)} traces have no label; pass `expected` explicitly "
                    f"(first: {missing[0]})"
                )
            expected = [int(t.label) for t in traces]  # type: ignore[arg-type]
        return cls(
            [EvalCase.from_trace(t, e, provenance=provenance, **kw) for t, e in zip(traces, expected, strict=False)],
            name,
        )
