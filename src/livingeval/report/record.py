"""Versioned result records.

Every analysis in this library serialises to a plain JSON record, and every figure is
drawn from records rather than from traces. Two consequences worth the small amount of
plumbing:

- Redrawing a figure never re-runs a judge, so iterating on presentation costs nothing
  and cannot accidentally change a number.
- The artifact you attach to a pull request, a bug report or a paper is the thing the
  numbers came from, not a screenshot of them.

Records carry a schema version. `load` refuses a record from a future schema rather
than reading it optimistically, because the failure mode of the alternative is a
plausible-looking number computed from a field that has changed meaning.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from livingeval._version import SCHEMA_VERSION, __version__

__all__ = ["Record", "load", "save"]


@dataclass
class Record:
    """A bundle of analysis results plus the environment that produced them."""

    results: list[dict]
    meta: dict = field(default_factory=dict)
    schema: int = SCHEMA_VERSION
    version: str = __version__

    def of_kind(self, kind: str) -> list[dict]:
        return [r for r in self.results if r.get("kind") == kind]

    def first(self, kind: str) -> dict | None:
        found = self.of_kind(kind)
        return found[0] if found else None

    def as_dict(self) -> dict:
        return {
            "schema": self.schema,
            "livingeval": self.version,
            "meta": self.meta,
            "results": self.results,
        }


def _to_dict(obj: Any) -> dict:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    raise TypeError(f"{type(obj).__name__} is not serialisable; it needs an as_dict()")


def save(results: Iterable[Any], path, meta: dict | None = None) -> Path:
    """Write results to JSON. Accepts any mix of result objects and plain dicts."""
    record = Record(
        results=[_to_dict(r) for r in results],
        meta={
            "python": platform.python_version(),
            "platform": platform.system(),
            **(meta or {}),
        },
    )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def load(path) -> Record:
    """Read a record, refusing anything written by a newer schema."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = int(data.get("schema", 0))
    if schema > SCHEMA_VERSION:
        raise ValueError(
            f"{path} was written with record schema {schema}; this livingeval "
            f"({__version__}) understands up to {SCHEMA_VERSION}. Upgrade rather than "
            "reading it optimistically."
        )
    return Record(
        results=data.get("results", []),
        meta=data.get("meta", {}),
        schema=schema,
        version=data.get("livingeval", "unknown"),
    )
