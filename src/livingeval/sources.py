"""Resolve a trace source from a string.

    sources.load_traces("traces/2026-08/*.jsonl")
    sources.load_traces("otel-export.json")
    sources.load_traces("langfuse:limit=500")
    sources.load_traces("synthetic:drifting,n=1200,seed=0")
    sources.load_traces("sqlite:///livingeval.db")

One parser, shared by the CLI, the platform and any script, so that "where do the
traces come from" is written once. Format detection is by extension and content sniff
rather than by a required flag: the first thing anyone does with a new eval tool is
point it at the log file they already have, and making them declare its shape first is
where adoption is lost.
"""

from __future__ import annotations

from pathlib import Path

from livingeval.trace.types import TraceSet

__all__ = ["describe_spec", "load_traces"]


def describe_spec() -> str:
    """Help text, so the CLI and the docs cannot drift apart."""
    return (
        "a .jsonl file or glob | an OTel/Langfuse .json export | "
        "langfuse:[limit=,pages=,expand=1] | "
        "synthetic:<generator>[,n=,seed=] | sqlite:///path.db or postgresql://..."
    )


def _options(text: str) -> dict:
    """`limit=500,expand=1` as a dict. Empty values are dropped."""
    out: dict = {}
    for item in text.split(","):
        key, _, value = item.partition("=")
        if key.strip() and value.strip():
            out[key.strip()] = value.strip()
    return out


def load_traces(spec: str, limit: int | None = None) -> TraceSet:
    """Load traces from a file, a glob, a generator or a store."""
    from livingeval.trace import ingest

    # -- a store ------------------------------------------------------------
    if spec.startswith(("sqlite:", "postgres:", "postgresql:")):
        from livingeval.store import open_store

        return open_store(spec).get_traces(limit=limit)

    # -- a live Langfuse project ---------------------------------------------
    #
    # A one-shot pull, as distinct from `ingest --follow`, which polls forever.
    # "Connect it to Langfuse" almost always means "read my last few hundred
    # traces now" -- for a coverage number or a drift report -- and having only
    # the daemon meant piping an export file around to do the obvious thing.
    if spec == "langfuse" or spec.startswith("langfuse:"):
        from livingeval.trace.ingest.langfuse import fetch

        options = _options(spec.partition(":")[2])
        return fetch(limit=int(options.get("limit", limit or 500)),
                     pages=int(options.get("pages", 1)),
                     expand=bool(int(options.get("expand", 0))),
                     host=options.get("host"),
                     out_path=options.get("out") or None)

    # -- a generator --------------------------------------------------------
    if spec.startswith("synthetic:"):
        from livingeval import synthetic

        parts = spec.split(":", 1)[1].split(",")
        fn = getattr(synthetic, parts[0], None)
        if fn is None or parts[0].startswith("_"):
            available = [
                n for n in ("shortcut", "lexical", "morphology", "deep", "drifting", "stable")
            ]
            raise ValueError(f"no synthetic generator {parts[0]!r}; try one of {available}")
        kwargs: dict = {}
        for kv in parts[1:]:
            key, _, value = kv.partition("=")
            if not value:
                continue
            try:
                kwargs[key] = int(value)
            except ValueError:
                try:
                    kwargs[key] = float(value)
                except ValueError:
                    kwargs[key] = value
        return fn(**kwargs)

    # -- a file -------------------------------------------------------------
    suffix = Path(spec).suffix.lower()
    if suffix in (".jsonl", ".ndjson") or "*" in spec or "?" in spec:
        return ingest.read_jsonl(spec)
    if suffix == ".json":
        # Sniff rather than ask. Langfuse exports carry `sessionId`/`observations`;
        # OTel dumps carry `resourceSpans`/`span_id`/`traceId`.
        head = Path(spec).read_text(encoding="utf-8")[:2000]
        if "sessionId" in head or "observations" in head:
            return ingest.read_langfuse_json(spec)
        return ingest.read_otel_json(spec)

    raise ValueError(f"do not know how to read {spec!r}. Expected {describe_spec()}")
