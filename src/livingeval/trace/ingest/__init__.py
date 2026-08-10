"""Adapters from whatever you already log into `TraceSet`."""

from livingeval.trace.ingest.jsonl import coerce_trace, from_records, read_jsonl
from livingeval.trace.ingest.langfuse import (
    from_export,
    read_langfuse_json,
)
from livingeval.trace.ingest.langfuse import (
    poll as poll_langfuse,
)
from livingeval.trace.ingest.otel import from_spans, read_otel_json

#: Convenience aliases used by the CLI and the README.
jsonl = read_jsonl
otel = read_otel_json
langfuse = read_langfuse_json

__all__ = [
    "coerce_trace",
    "from_export",
    "from_records",
    "from_spans",
    "jsonl",
    "langfuse",
    "otel",
    "poll_langfuse",
    "read_jsonl",
    "read_langfuse_json",
    "read_otel_json",
]
