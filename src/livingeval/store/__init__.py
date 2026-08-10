"""Persistence for the platform: traces, verdicts, suites, proposals, records.

    from livingeval.store import open_store

    store = open_store("sqlite:///livingeval.db")        # default, stdlib, one file
    store = open_store("postgresql://...")               # Supabase works unchanged

The store is an **option, not a prerequisite**. Every analysis function takes plain
`TraceSet` and `EvalSuite` values and neither knows nor cares whether they came from a
database, a JSONL file or a generator. That is what keeps `import livingeval` usable
with no infrastructure at all.
"""

from livingeval.store.base import Store, StoreStats, open_store
from livingeval.store.sqlite import SQLiteStore

__all__ = ["PostgresStore", "SQLiteStore", "Store", "StoreStats", "open_store"]


def __getattr__(name: str):
    """`PostgresStore` is resolved lazily so psycopg2 stays an optional extra."""
    if name == "PostgresStore":
        from livingeval.store.postgres import PostgresStore

        return PostgresStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
