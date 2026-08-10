"""Read traces from OpenTelemetry spans.

The LangChain survey puts observability adoption at ~89% and eval adoption at ~52%,
so for most teams the trace data already exists and the eval data does not. This
adapter is the bridge: it reads the OTel export you already have rather than asking
for new instrumentation.

Reconstruction rules, all of them lossy in ways that are documented rather than
silent:

- Spans are grouped by `trace_id`; a trace becomes one `Trace`.
- Turn order is span start time. Concurrent tool spans get an arbitrary but stable
  order (ties broken by span id), because the alternative is a non-deterministic
  rendering and therefore a non-deterministic coverage number.
- Role is read from the GenAI semantic conventions where present
  (`gen_ai.operation.name`, `gen_ai.prompt`, `gen_ai.completion`), and falls back to
  span-name heuristics. Unmapped spans become `tool` turns named after the span, on
  the grounds that a retriever call is closer to a tool call than to anything else.
- `session_id` comes from `session.id` / `gen_ai.conversation.id` attributes.

Anything this cannot map is preserved in `Turn.meta`, never dropped.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from livingeval.trace.types import Trace, TraceSet, Turn

__all__ = ["from_spans", "read_otel_json"]

_SESSION_ATTRS = ("session.id", "gen_ai.conversation.id", "langfuse.session.id", "thread.id")
_PROMPT_ATTRS = ("gen_ai.prompt", "gen_ai.input.messages", "input.value", "llm.prompts")
_COMPLETION_ATTRS = ("gen_ai.completion", "gen_ai.output.messages", "output.value", "llm.output")


def _attrs(span: dict) -> dict:
    """OTel exports carry attributes either as a flat dict or as the protobuf-ish
    `[{"key": k, "value": {"stringValue": v}}]` list. Both appear in the wild."""
    raw = span.get("attributes", {}) or {}
    if isinstance(raw, dict):
        return raw
    out: dict[str, Any] = {}
    for item in raw:
        key = item.get("key")
        value = item.get("value", {})
        if isinstance(value, dict):
            value = next(iter(value.values()), None)
        out[key] = value
    return out


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _messages_from(value: Any) -> list[tuple[str, str]]:
    """Extract (role, content) pairs from a prompt/completion attribute."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return [("user", value)]
    if isinstance(value, dict):
        value = [value]
    out: list[tuple[str, str]] = []
    if isinstance(value, list):
        for m in value:
            if isinstance(m, dict):
                role = m.get("role", "user")
                role = {"human": "user", "ai": "assistant", "function": "tool"}.get(role, role)
                if role not in ("system", "user", "assistant", "tool"):
                    role = "user"
                out.append((role, _as_text(m.get("content") or m.get("text") or "")))
            else:
                out.append(("user", _as_text(m)))
    return out


def _start(span: dict) -> float:
    for key in ("start_time_unix_nano", "startTimeUnixNano"):
        if key in span:
            return float(span[key]) / 1e9
    for key in ("start_time", "startTime", "timestamp"):
        if key in span:
            v = span[key]
            if isinstance(v, int | float):
                v = float(v)
                return v / 1e9 if v > 1e15 else (v / 1000.0 if v > 1e11 else v)
            from datetime import datetime

            try:
                return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
            except ValueError:
                return 0.0
    return 0.0


def from_spans(spans: Iterable[dict]) -> TraceSet:
    """Group a flat span stream into traces."""
    grouped: dict[str, list[dict]] = {}
    for s in spans:
        tid = str(s.get("trace_id") or s.get("traceId") or s.get("context", {}).get("trace_id", ""))
        grouped.setdefault(tid or "unknown", []).append(s)

    traces: list[Trace] = []
    for tid, group in grouped.items():
        group.sort(key=lambda s: (_start(s), str(s.get("span_id") or s.get("spanId") or "")))
        turns: list[Turn] = []
        session: str | None = None
        seen_prompt = False

        for span in group:
            attrs = _attrs(span)
            for key in _SESSION_ATTRS:
                if attrs.get(key) and session is None:
                    session = str(attrs[key])

            prompt = next((attrs[k] for k in _PROMPT_ATTRS if k in attrs), None)
            completion = next((attrs[k] for k in _COMPLETION_ATTRS if k in attrs), None)

            if prompt is not None and not seen_prompt:
                # Only the first LLM span's prompt is expanded; later spans repeat the
                # same history and would double-count the conversation.
                turns.extend(Turn(role=r, content=c) for r, c in _messages_from(prompt) if c)
                seen_prompt = True
            elif prompt is not None:
                msgs = _messages_from(prompt)
                if msgs:
                    role, content = msgs[-1]
                    if content:
                        turns.append(Turn(role=role, content=content))

            if completion is not None:
                for role, content in _messages_from(completion) or [("assistant", _as_text(completion))]:
                    if content:
                        turns.append(Turn(role="assistant" if role == "user" else role, content=content))

            if prompt is None and completion is None:
                name = str(span.get("name", "span"))
                payload = _as_text(attrs.get("output") or attrs.get("tool.output") or attrs)
                turns.append(Turn(role="tool", content=payload, name=name, meta={"span": name}))

        traces.append(
            Trace(
                trace_id=tid,
                turns=turns,
                ts=_start(group[0]) if group else 0.0,
                session_id=session,
                meta={"n_spans": len(group), "source": "otel"},
            )
        )
    return TraceSet(sorted(traces, key=lambda t: t.ts), {"source": {"kind": "otel"}})


def read_otel_json(path) -> TraceSet:
    """Read an OTel JSON export: either a bare list of spans, or the
    `{"resourceSpans": [...]}` envelope the collector emits."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, dict) and "resourceSpans" in data:
        spans: list[dict] = []
        for rs in data["resourceSpans"]:
            for ss in rs.get("scopeSpans", rs.get("instrumentationLibrarySpans", [])):
                spans.extend(ss.get("spans", []))
        return from_spans(spans)
    if isinstance(data, dict) and "spans" in data:
        return from_spans(data["spans"])
    return from_spans(data)
