"""Read traces from JSONL, the format you get by writing your own logger.

Two shapes are accepted and auto-detected:

1. **Native** - one `Trace.as_dict()` per line.
2. **Loose** - one record per line with a messages-shaped list under any of
   `turns` / `messages` / `conversation`, plus whatever id and timestamp keys your
   logger happened to use.

The loose reader exists because the first thing anyone does with a new eval tool is
point it at the log file they already have, and failing at that step is where most
adoption is lost.
"""

from __future__ import annotations

import glob as _glob
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from livingeval.trace.types import Trace, TraceSet, Turn

__all__ = ["coerce_trace", "from_records", "read_jsonl"]

_ID_KEYS = ("trace_id", "id", "traceId", "run_id", "runId", "observation_id")
_SESSION_KEYS = ("session_id", "sessionId", "thread_id", "conversation_id", "user_id")
_TS_KEYS = ("ts", "timestamp", "time", "created_at", "createdAt", "start_time", "startTime")
_TURN_KEYS = ("turns", "messages", "conversation", "history")
_LABEL_KEYS = ("label", "human_label", "ground_truth", "gt")


def _first(d: dict, keys: Iterable[str], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _to_epoch(value: Any) -> float:
    """Best-effort timestamp coercion. ISO-8601 strings, epoch seconds and epoch
    milliseconds all appear in real logs; milliseconds are detected by magnitude."""
    if value is None:
        return 0.0
    if isinstance(value, int | float):
        v = float(value)
        return v / 1000.0 if v > 1e11 else v
    if isinstance(value, str):
        from datetime import datetime

        text = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text).timestamp()
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return 0.0
    return 0.0


def coerce_trace(record: dict, index: int = 0) -> Trace:
    """Turn one loose record into a `Trace`."""
    if "turns" in record and isinstance(record.get("turns"), list) and "trace_id" in record:
        try:
            return Trace.from_dict(record)
        except (KeyError, TypeError):
            pass

    raw_turns = _first(record, _TURN_KEYS, []) or []
    turns = []
    for t in raw_turns:
        if isinstance(t, str):
            turns.append({"role": "user", "content": t})
            continue
        role = t.get("role") or t.get("type") or "assistant"
        role = {"human": "user", "ai": "assistant", "function": "tool"}.get(role, role)
        if role not in ("system", "user", "assistant", "tool"):
            role = "assistant"
        content = t.get("content") or t.get("text") or t.get("output") or ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, sort_keys=True)
        turns.append({"role": role, "content": content, "name": t.get("name") or t.get("tool")})

    session = _first(record, _SESSION_KEYS)
    label = _first(record, _LABEL_KEYS)
    return Trace(
        trace_id=str(_first(record, _ID_KEYS, f"trace-{index:06d}")),
        turns=[Turn.from_dict(t) for t in turns],
        ts=_to_epoch(_first(record, _TS_KEYS, 0.0)),
        session_id=None if session is None else str(session),
        label=None if label is None else int(label),
        meta={k: v for k, v in record.items() if k not in set(_TURN_KEYS)},
    )


def from_records(records: Iterable[dict]) -> TraceSet:
    return TraceSet([coerce_trace(r, i) for i, r in enumerate(records)])


def read_jsonl(pattern: str | Path) -> TraceSet:
    """Read one file, or a glob such as `traces/2026-08/*.jsonl`."""
    paths = sorted(_glob.glob(str(pattern))) or ([str(pattern)] if Path(pattern).exists() else [])
    if not paths:
        raise FileNotFoundError(f"no files matched {pattern!r}")
    records: list[dict] = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    ts = from_records(records)
    ts.meta["source"] = {"kind": "jsonl", "paths": paths, "n_records": len(records)}
    return ts
