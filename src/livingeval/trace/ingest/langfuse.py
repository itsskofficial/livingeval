"""Read traces from a Langfuse export.

Langfuse is the trace backbone the portfolio plan already budgets for, and its
export shape is stable enough to parse offline: a list of traces, each with
`input`, `output`, `sessionId`, `timestamp` and a nested `observations` list.

This adapter reads a **file**, not the API. Two reasons, both deliberate:

- Every number this library produces should be reproducible from an artifact you can
  commit, diff and attach to a bug report. A function that re-fetches from a live
  API produces numbers nobody else can reproduce.
- No network in the default install, so the whole test suite runs on every commit.

`livingeval.trace.ingest.langfuse.fetch()` exists for convenience and is the only
function here that touches the network; it writes an export file and then calls the
offline reader, so the analysis path is identical either way.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from livingeval.trace.types import Trace, TraceSet, Turn

__all__ = ["fetch", "from_export", "read_langfuse_json"]


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _messages(value: Any, default_role: str) -> list[Turn]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return [Turn(role=default_role, content=value)]
        value = parsed
    if isinstance(value, dict):
        if "messages" in value:
            value = value["messages"]
        elif "role" in value or "content" in value:
            value = [value]
        else:
            return [Turn(role=default_role, content=_as_text(value))]
    out: list[Turn] = []
    if isinstance(value, list):
        for m in value:
            if isinstance(m, dict) and ("role" in m or "content" in m):
                role = m.get("role", default_role)
                role = {"human": "user", "ai": "assistant", "function": "tool"}.get(role, role)
                if role not in ("system", "user", "assistant", "tool"):
                    role = default_role
                content = _as_text(m.get("content", ""))
                if content:
                    out.append(Turn(role=role, content=content, name=m.get("name")))
            else:
                out.append(Turn(role=default_role, content=_as_text(m)))
    return out


def _ts(value: Any) -> float:
    if isinstance(value, int | float):
        v = float(value)
        return v / 1000.0 if v > 1e11 else v
    if isinstance(value, str):
        from datetime import datetime

        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0
    return 0.0


def from_export(records: Iterable[dict]) -> TraceSet:
    """Convert exported Langfuse trace records into a `TraceSet`."""
    traces: list[Trace] = []
    for i, rec in enumerate(records):
        turns = _messages(rec.get("input"), "user")

        # The list endpoint (`GET /api/public/traces`) returns `observations` as a list
        # of **id strings**; only a full export or a single-trace fetch nests them as
        # objects. Skipping the strings means a trace still yields its input and output
        # turns instead of crashing on `str.get` — the tool calls are simply absent,
        # which `fetch(expand=True)` fixes by re-fetching each trace individually.
        observations = [o for o in (rec.get("observations") or []) if isinstance(o, dict)]
        # The API returns observations unordered. A judge that grades "did the agent's
        # claims match what its tools returned" needs them in the order they happened,
        # so sort by start time rather than trusting the payload.
        observations.sort(key=lambda o: str(o.get("startTime") or ""))

        for obs in observations:
            otype = (obs.get("type") or "").upper()
            # The root span *is* the agent run: its output is the final answer, which
            # already arrives via `rec["output"]`. Treating it as a tool turn duplicated
            # the answer and, worse, presented it to the judge as though a tool had
            # returned it. Only nested spans are real steps.
            #
            # `"parentObservationId" in obs` matters, not just its value. The live API
            # always sends the field, null for the root. A hand-written export or a
            # different exporter may omit it entirely, and reading absent-as-root
            # silently dropped every span in those - which is exactly what the first
            # version of this did, taking a passing test with it.
            is_root = "parentObservationId" in obs and obs["parentObservationId"] is None
            if otype == "SPAN" and is_root:
                continue
            # `TOOL` is its own observation type in current Langfuse and was missing
            # here entirely, so every actual tool call was dropped on the floor — the
            # one thing the judge most needs to see.
            if otype in ("TOOL", "SPAN", "EVENT", "RETRIEVER") and obs.get("output") is not None:
                turns.append(
                    Turn(
                        role="tool",
                        content=_as_text(obs.get("output")),
                        name=obs.get("name"),
                        meta={"observation_type": otype},
                    )
                )
            elif otype == "GENERATION":
                turns.extend(_messages(obs.get("output"), "assistant"))

        turns.extend(_messages(rec.get("output"), "assistant"))

        # The final answer arrives twice: once as the last GENERATION's output and once
        # as the trace's own `output`. They are the same text, and a judge shown the
        # answer twice reads it as the agent repeating itself. Collapse runs of
        # identical adjacent turns rather than dropping either source, since which one
        # is present depends on how the trace was instrumented.
        deduped: list[Turn] = []
        for turn in turns:
            if deduped and deduped[-1].role == turn.role and deduped[-1].content == turn.content:
                continue
            deduped.append(turn)
        turns = deduped

        scores = {s.get("name"): s.get("value") for s in rec.get("scores", []) or []}
        label = scores.get("human") if "human" in scores else None

        traces.append(
            Trace(
                trace_id=str(rec.get("id") or rec.get("trace_id") or f"lf-{i:06d}"),
                turns=turns,
                ts=_ts(rec.get("timestamp") or rec.get("createdAt")),
                session_id=(lambda s: None if s is None else str(s))(
                    rec.get("sessionId") or rec.get("session_id")
                ),
                label=None if label is None else int(label),
                meta={
                    "name": rec.get("name"),
                    "release": rec.get("release"),
                    "version": rec.get("version"),
                    "tags": rec.get("tags"),
                    "scores": scores,
                    "source": "langfuse",
                },
            )
        )
    return TraceSet(sorted(traces, key=lambda t: t.ts), {"source": {"kind": "langfuse"}})


def read_langfuse_json(path) -> TraceSet:
    """Read a Langfuse export file: a bare list, or `{"data": [...]}`."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("data", data.get("traces", []))
    return from_export(data)


def _request(path: str, host: str | None = None, **query) -> dict:  # pragma: no cover - network
    """One authenticated GET against the Langfuse public API."""
    import os
    import urllib.request
    from base64 import b64encode
    from urllib.parse import urlencode

    pk, sk = os.environ.get("LANGFUSE_PUBLIC_KEY"), os.environ.get("LANGFUSE_SECRET_KEY")
    if not pk or not sk:
        raise RuntimeError("set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in the environment")
    base = (host or os.environ.get("LANGFUSE_HOST") or "https://cloud.langfuse.com").rstrip("/")
    url = f"{base}{path}?" + urlencode({k: v for k, v in query.items() if v is not None})
    req = urllib.request.Request(url)
    req.add_header("Authorization", "Basic " + b64encode(f"{pk}:{sk}".encode()).decode())
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch(
    limit: int = 1000,
    out_path: str | None = "langfuse_export.json",
    host: str | None = None,
    pages: int = 1,
    expand: bool = False,
    **query,
) -> TraceSet:  # pragma: no cover - network
    """Fetch from the Langfuse API, optionally write an export file, then read it.

    `pages > 1` follows the API's pagination. Writing the export is the default because
    every number this library produces should be reproducible from an artifact you can
    commit and attach to a bug report; pass `out_path=None` to skip it.

    `expand=True` re-fetches each trace individually. The list endpoint returns
    `observations` as bare id strings, so without it you get the user's question and the
    agent's final answer but **no tool calls** — enough for coverage and clustering,
    not enough for a judge that grades whether the agent's claims match what its tools
    returned. It costs one request per trace, so it is off by default.

    Credentials come from `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` in the
    environment. Never from a source file. See DECISIONS.md #20.
    """
    records: list[dict] = []
    for page in range(1, pages + 1):
        payload = _request("/api/public/traces", host, limit=limit, page=page, **query)
        batch = payload.get("data", payload if isinstance(payload, list) else [])
        if not batch:
            break
        records.extend(batch)

    if expand:
        detailed = []
        for rec in records:
            trace_id = rec.get("id")
            if not trace_id:
                continue
            try:
                detailed.append(_request(f"/api/public/traces/{trace_id}", host))
            except Exception:
                detailed.append(rec)
        records = detailed

    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"data": records}, fh, ensure_ascii=False)
    return from_export(records)


def poll(
    store,
    interval_seconds: float = 30.0,
    limit: int = 200,
    host: str | None = None,
    max_iterations: int | None = None,
    source: str = "langfuse",
    verbose: bool = True,
    expand: bool = True,
    **query,
) -> int:  # pragma: no cover - network
    """Poll Langfuse and write new traces into a store, forever or `max_iterations` times.

    `expand` defaults to **True** here, unlike `fetch`. Polling exists to build the
    corpus a judge will later grade, and the list endpoint omits observations - so
    without it every ingested trace arrives with the question and the answer but no tool
    calls, and a judge asked whether the agent's claims match its tools has nothing to
    compare against. It costs one extra request per trace, which a background poll can
    afford and an interactive `fetch` often cannot.

    Returns the total number of traces written. Deduplication is the store's job -
    `put_traces` is idempotent on `trace_id` - so re-fetching an overlapping window is
    harmless and the poll needs no cursor bookkeeping of its own.

    Deliberately a foreground loop rather than a daemon or a scheduler. `livingeval
    ingest --follow` runs it in a terminal you can see, and the process that stops when
    you close it is the one you can reason about. Anything more than this belongs in
    your own scheduler, not in an eval library.
    """
    import time as _time

    total = 0
    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        iterations += 1
        try:
            traces = fetch(limit=limit, out_path=None, host=host,
                           expand=expand, **query)
            written = store.put_traces(traces, source=source)
            total += written
            if verbose:
                print(f"[poll {iterations}] fetched {len(traces)}, store now "
                      f"{store.count_traces()} traces", flush=True)
        except (OSError, ValueError, RuntimeError) as e:
            # A transient API failure must not kill a long-running ingest.
            if verbose:
                print(f"[poll {iterations}] {type(e).__name__}: {e}", flush=True)
        if max_iterations is None or iterations < max_iterations:
            _time.sleep(interval_seconds)
    return total
