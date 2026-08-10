"""The trace types everything else is built on.

Three fields carry the design:

- `Trace.ts` — arrival time. Without it, "your suite has gone stale" is a feeling.
  With it, staleness is a windowed quantity you can put a number and an interval on.
- `Trace.session_id` — the cross-validation group. Turns from one session share a
  user, a topic and usually most of their tokens. A random train/test split over
  turns scores a model on text it has effectively already seen, which makes every
  rung of the judge-complexity ladder look one level deeper than it is.
- `Trace.label` — human ground truth, where it exists. Its absence is what makes a
  judge `UNVALIDATED`, and that word appears in the record rather than nowhere.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field, replace

import numpy as np

__all__ = ["Trace", "TraceSet", "Turn"]

ROLES = ("system", "user", "assistant", "tool")


@dataclass(frozen=True)
class Turn:
    """One message in a trace."""

    role: str
    content: str
    name: str | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {self.role!r}")

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Turn:
        return cls(
            role=d["role"],
            content=d.get("content", "") or "",
            name=d.get("name"),
            meta=dict(d.get("meta", {})),
        )


@dataclass
class Trace:
    """One agent interaction: an ordered list of turns plus the metadata that makes
    it analysable."""

    trace_id: str
    turns: list[Turn]
    ts: float = 0.0
    session_id: str | None = None
    label: int | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.turns = [t if isinstance(t, Turn) else Turn.from_dict(t) for t in self.turns]

    # -- convenience ------------------------------------------------------------

    @property
    def group(self) -> str:
        """The CV group. Falls back to `trace_id` when sessions are unavailable, which
        makes every trace its own group and the split degenerate to a random one."""
        return self.session_id or self.trace_id

    def by_role(self, role: str) -> list[Turn]:
        return [t for t in self.turns if t.role == role]

    def last(self, role: str = "assistant") -> Turn | None:
        for t in reversed(self.turns):
            if t.role == role:
                return t
        return None

    def with_label(self, label: int | None) -> Trace:
        return replace(self, label=label)

    def as_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "turns": [t.as_dict() for t in self.turns],
            "ts": self.ts,
            "session_id": self.session_id,
            "label": self.label,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Trace:
        return cls(
            trace_id=str(d["trace_id"]),
            turns=[Turn.from_dict(t) for t in d.get("turns", [])],
            ts=float(d.get("ts", 0.0) or 0.0),
            session_id=d.get("session_id"),
            label=d.get("label"),
            meta=dict(d.get("meta", {})),
        )

    def content_hash(self) -> str:
        """Content address over the turns only.

        Deliberately excludes `ts`, `label` and `meta`: the judge's verdict is a
        function of the conversation, so the same conversation seen twice must hit
        the same cache entry. See DECISIONS.md #8.
        """
        payload = json.dumps(
            [[t.role, t.name, t.content] for t in self.turns],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class TraceSet(Sequence[Trace]):
    """An ordered collection of traces with the slicing the rest of the library needs."""

    def __init__(self, traces: Iterable[Trace], meta: dict | None = None):
        self._traces: list[Trace] = list(traces)
        self.meta: dict = dict(meta or {})

    # -- Sequence protocol ------------------------------------------------------

    def __len__(self) -> int:
        return len(self._traces)

    def __iter__(self) -> Iterator[Trace]:
        return iter(self._traces)

    def __getitem__(self, i):  # type: ignore[override]
        if isinstance(i, slice):
            return TraceSet(self._traces[i], self.meta)
        return self._traces[i]

    def __repr__(self) -> str:  # pragma: no cover - display only
        span = ""
        if self._traces:
            span = f" ts=[{self.tmin:.1f}, {self.tmax:.1f}]"
        return f"<TraceSet n={len(self)} sessions={self.n_sessions}{span}>"

    # -- properties -------------------------------------------------------------

    @property
    def ids(self) -> list[str]:
        return [t.trace_id for t in self._traces]

    @property
    def groups(self) -> np.ndarray:
        return np.asarray([t.group for t in self._traces], dtype=object)

    @property
    def n_sessions(self) -> int:
        return len(set(self.groups.tolist()))

    @property
    def timestamps(self) -> np.ndarray:
        return np.asarray([t.ts for t in self._traces], dtype=float)

    @property
    def tmin(self) -> float:
        return float(self.timestamps.min()) if len(self) else float("nan")

    @property
    def tmax(self) -> float:
        return float(self.timestamps.max()) if len(self) else float("nan")

    @property
    def labels(self) -> np.ndarray:
        """Human labels, with -1 standing in for unlabelled."""
        return np.asarray([-1 if t.label is None else int(t.label) for t in self._traces], dtype=int)

    # -- slicing ----------------------------------------------------------------

    def window(self, start: float | None = None, end: float | None = None) -> TraceSet:
        """Traces with `start <= ts < end`. This is how drift is measured."""
        lo = -np.inf if start is None else start
        hi = np.inf if end is None else end
        return TraceSet((t for t in self._traces if lo <= t.ts < hi), self.meta)

    def windows(self, n: int) -> list[TraceSet]:
        """Split into `n` equal-width time windows over `[tmin, tmax]`.

        The final window is closed at the top. Nudging the upper edge instead is the
        obvious move and it silently fails: epoch-second timestamps are around 1.8e9,
        where the spacing between representable doubles is larger than the epsilon
        anyone would think to add, so `tmax + 1e-9 == tmax` and the last trace
        vanishes. The windows must partition the stream exactly, so the bound is
        handled rather than approximated.
        """
        if len(self) == 0:
            return [TraceSet([], self.meta) for _ in range(n)]
        edges = np.linspace(self.tmin, self.tmax, n + 1)
        out = [self.window(edges[i], edges[i + 1]) for i in range(n - 1)]
        out.append(TraceSet((t for t in self._traces if t.ts >= edges[n - 1]), self.meta))
        return out

    def labelled(self) -> TraceSet:
        """Only traces carrying human ground truth."""
        return TraceSet((t for t in self._traces if t.label is not None), self.meta)

    def filter(self, predicate) -> TraceSet:
        return TraceSet((t for t in self._traces if predicate(t)), self.meta)

    def sample(self, n: int, seed: int = 0) -> TraceSet:
        rng = np.random.default_rng(seed)
        if n >= len(self):
            return TraceSet(self._traces, self.meta)
        idx = rng.choice(len(self), size=n, replace=False)
        return TraceSet([self._traces[i] for i in sorted(idx.tolist())], self.meta)

    def sorted_by_time(self) -> TraceSet:
        return TraceSet(sorted(self._traces, key=lambda t: t.ts), self.meta)

    def concat(self, other: TraceSet) -> TraceSet:
        return TraceSet(list(self._traces) + list(other), {**self.meta, **other.meta})

    # -- io ---------------------------------------------------------------------

    def as_dicts(self) -> list[dict]:
        return [t.as_dict() for t in self._traces]

    def to_jsonl(self, path) -> None:
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for t in self._traces:
                fh.write(json.dumps(t.as_dict(), ensure_ascii=False) + "\n")
