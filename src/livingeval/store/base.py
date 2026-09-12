"""The store protocol.

A store is where the platform keeps traces, verdicts, suites, proposals and result
records so that the dashboard and the online scorer have something to read between
process restarts.

Two implementations ship:

- **SQLite** (`store.sqlite`) - the default. `sqlite3` is in the standard library, so
  this adds no dependency, runs anywhere, and the whole database is one file you can
  copy, diff-by-export, or delete. For a local platform and a demo it is the right
  answer and there is no second-best.
- **Postgres / Supabase** (`store.postgres`) - the same schema over `psycopg2`, for when
  more than one process needs to write.

The protocol is deliberately narrow. A store persists and retrieves; it does not
analyse. Every number still comes from the analysis modules, which take plain
`TraceSet` and `EvalSuite` values and neither know nor care where they were loaded
from. That separation is why the library remains usable as a plain import with no
database at all - the store is an option, not a prerequisite.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from livingeval.suite.suite import EvalSuite
from livingeval.trace.types import TraceSet

__all__ = ["Store", "StoreStats", "open_store"]


class StoreStats(dict):
    """Row counts per table, for the dashboard header."""

    def summary(self) -> str:  # pragma: no cover - display only
        return "  ".join(f"{k}={v}" for k, v in sorted(self.items()))


@runtime_checkable
class Store(Protocol):
    """Persistence for the platform."""

    # -- traces ----------------------------------------------------------------

    def put_traces(self, traces: TraceSet, source: str = "api") -> int: ...
    def get_traces(
        self,
        limit: int | None = None,
        since: float | None = None,
        until: float | None = None,
        source: str | None = None,
    ) -> TraceSet: ...
    def count_traces(self) -> int: ...

    # -- verdicts --------------------------------------------------------------

    def put_verdict(self, trace_id: str, judge: str, label: int, score: float | None,
                    rationale: str | None, cost_usd: float, latency_s: float) -> None: ...
    def get_verdicts(self, judge: str | None = None) -> dict[str, dict]: ...

    # -- suites ----------------------------------------------------------------

    def put_suite(self, suite: EvalSuite) -> str: ...
    def get_suite(self, name: str) -> EvalSuite | None: ...
    def list_suites(self) -> list[dict]: ...

    # -- proposals (the human-in-the-loop queue) -------------------------------

    def put_proposals(self, suite_name: str, proposals: list[dict]) -> int: ...
    def get_proposals(self, suite_name: str | None = None,
                      status: str | None = None) -> list[dict]: ...
    def confirm_proposal(self, proposal_id: int, reviewer: str, expected: int) -> dict: ...
    def reject_proposal(self, proposal_id: int, reviewer: str) -> dict: ...

    def confirmed_cases(self, suite_name: str) -> list: ...

    # -- result records --------------------------------------------------------

    def put_record(self, kind: str, payload: dict, label: str = "") -> int: ...
    def get_records(self, kind: str | None = None, limit: int = 50) -> list[dict]: ...

    # -- online scoring telemetry ----------------------------------------------

    def put_online_score(self, scorer: str, label: int, latency_us: float,
                         trace_id: str | None = None) -> None: ...
    def online_score_stats(self) -> dict: ...
    def trace_time_span(self) -> tuple[float | None, float | None]: ...

    # -- schema and health -----------------------------------------------------

    def migrate(self) -> list[str]: ...
    def applied_migrations(self) -> list[str]: ...
    def ping(self) -> float: ...

    # -- lifecycle -------------------------------------------------------------

    def stats(self) -> StoreStats: ...
    def close(self) -> None: ...


def open_store(url: str = "sqlite:///livingeval.db") -> Store:
    """Open a store from a URL.

    `sqlite:///path/to.db` (or `sqlite://:memory:`) and
    `postgresql://user:pass@host/db` are understood. Supabase connection strings are
    Postgres URLs, so they work unchanged.

    Credentials come from the URL you pass or from the environment, never from a
    source file. See DECISIONS.md #20.
    """
    if url.startswith("sqlite"):
        from livingeval.store.sqlite import SQLiteStore

        # One leading slash, not all of them. The convention every SQLAlchemy
        # user already has in their fingers is three slashes for a relative
        # path and four for an absolute one, so `sqlite:////var/lib/live.db`
        # must stay `/var/lib/live.db`. Stripping the lot made it relative to
        # the working directory -- on Linux and macOS, every absolute DSN
        # silently opened an empty database somewhere else. Windows was immune,
        # because there the path after the slashes starts `C:` and stays
        # absolute, which is how this survived.
        rest = url.split("://", 1)[1]
        path = (rest[1:] if rest.startswith("/") else rest) or ":memory:"
        return SQLiteStore(path)
    if url.startswith(("postgres://", "postgresql://")):
        from livingeval.store.postgres import PostgresStore

        return PostgresStore(url)
    raise ValueError(f"unrecognised store URL {url!r}; use sqlite:// or postgresql://")
