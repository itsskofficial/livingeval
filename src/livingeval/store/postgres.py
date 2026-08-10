"""Postgres and Supabase - the same schema, for a deployment.

Supabase connection strings are ordinary Postgres URLs, so this works against Supabase
unchanged. What is *not* unchanged is the behaviour of the pooler Supabase puts in front
of it, and getting that wrong is the most common way a working local app falls over the
first time two requests arrive together. See `store/dsn.py` for the three connection
strings Supabase gives you and what separates them.

What this store does about it:

- **Normalises the URL** before handing it to psycopg2: `sslmode=require` on any
  non-local host (psycopg2 defaults to `prefer`, which will silently accept an
  unencrypted connection), a `connect_timeout` so a wrong host fails in seconds rather
  than hanging a container start, and an `application_name` so you can see which process
  holds a connection in `pg_stat_activity`.
- **Uses a real connection pool** (`ThreadedConnectionPool`), because FastAPI serves
  requests on a thread pool and a single shared psycopg2 connection is not thread-safe.
  A single connection under concurrent requests corrupts the protocol and produces
  errors that look like data corruption.
- **Retries the initial connect** with backoff. A container that starts before its
  database is reachable should wait, not crash-loop.
- **Refuses to run migrations over a transaction pooler**, where the advisory lock that
  makes them safe would silently not be held.

Deliberately *not* here: an ORM, or migrations framework. The schema is six tables in
`store/migrations.py` with an applied-migrations table; anything more would be
infrastructure this library has no business owning.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from typing import Any

from livingeval.store import dsn as dsn_mod
from livingeval.store.base import StoreStats
from livingeval.store.migrations import BOOKKEEPING, LOCK_KEY, MIGRATIONS
from livingeval.suite.case import EvalCase
from livingeval.suite.suite import EvalSuite
from livingeval.trace.types import Trace, TraceSet

__all__ = ["PostgresStore"]

log = logging.getLogger("livingeval.store")

class PostgresStore:
    """The SQLite store's schema over psycopg2."""

    def __init__(self, url: str, minconn: int = 1, maxconn: int = 10,
                 migrate: bool = True, retries: int = 5, retry_wait_s: float = 1.5):
        try:
            import psycopg2
            import psycopg2.extras
            import psycopg2.pool
        except ImportError as e:  # pragma: no cover - optional extra
            raise ImportError("pip install 'livingeval[postgres]'") from e

        self._psycopg2 = psycopg2
        self._json = psycopg2.extras.Json
        self._dict_cursor = psycopg2.extras.RealDictCursor
        self.dsn = dsn_mod.parse(url)
        self.url = dsn_mod.normalise(url)

        last: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                self._pool = psycopg2.pool.ThreadedConnectionPool(minconn, maxconn, self.url)
                break
            except psycopg2.Error as e:  # pragma: no cover - needs a real server
                last = e
                if attempt == retries:
                    raise ConnectionError(
                        f"could not reach {self.dsn.host}:{self.dsn.port} after {retries} "
                        f"attempts: {e}\n{self.dsn.describe()}"
                    ) from e
                wait = retry_wait_s * attempt
                log.warning("postgres connect attempt %d/%d failed (%s); retrying in %.1fs",
                            attempt, retries, e, wait)
                time.sleep(wait)
        else:  # pragma: no cover - unreachable
            raise ConnectionError(str(last))

        if migrate:
            self.migrate()

    # -- connections -----------------------------------------------------------

    @contextlib.contextmanager
    def _connection(self):
        """Borrow a connection from the pool and always give it back.

        `autocommit` is set per checkout rather than once at construction, because a
        pooled connection can be returned mid-transaction by a failed request and the
        next borrower would inherit it.
        """
        conn = self._pool.getconn()
        try:
            conn.autocommit = True
            yield conn
        except Exception:
            # A request that failed mid-transaction must not hand a dirty connection
            # back to the pool for the next borrower to inherit.
            with contextlib.suppress(self._psycopg2.Error):  # pragma: no cover
                conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    @contextlib.contextmanager
    def _cur(self, dict_rows: bool = True):
        with self._connection() as conn:
            factory = self._dict_cursor if dict_rows else None
            cur = conn.cursor(cursor_factory=factory)
            try:
                yield cur
            finally:
                cur.close()

    # -- schema ----------------------------------------------------------------

    def migrate(self) -> list[str]:
        """Apply pending migrations under an advisory lock. Returns what was applied.

        The lock is what makes two containers booting together safe. It only works on a
        session connection: over a transaction pooler each statement may land on a
        different backend, so the lock would be taken and released immediately and the
        protection would be imaginary.
        """
        if self.dsn.is_pooler:
            raise RuntimeError(
                "refusing to migrate over a transaction pooler (port 6543).\n"
                "The advisory lock that makes concurrent migrations safe needs a session, "
                "and pgBouncer in transaction mode does not give you one.\n"
                "Use the direct / session connection string (port 5432) for migrations:\n"
                "  livingeval db migrate --url 'postgresql://...@...:5432/postgres'\n"
                "then run the service against the pooler URL as normal."
            )
        applied: list[str] = []
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_lock(%s)", (LOCK_KEY,))
            try:
                with conn.cursor() as cur:
                    cur.execute(BOOKKEEPING["postgres"])
                    cur.execute("SELECT id FROM schema_migrations")
                    done = {r[0] for r in cur.fetchall()}
                    for m in MIGRATIONS:
                        if m.id in done:
                            continue
                        log.info("applying migration %s (%s)", m.id, m.description)
                        cur.execute(m.postgres)
                        cur.execute(
                            "INSERT INTO schema_migrations (id, applied_at) VALUES (%s, %s) "
                            "ON CONFLICT (id) DO NOTHING",
                            (m.id, time.time()),
                        )
                        applied.append(m.id)
            finally:
                with conn.cursor() as cur:
                    cur.execute("SELECT pg_advisory_unlock(%s)", (LOCK_KEY,))
        return applied

    def applied_migrations(self) -> list[str]:
        with self._cur(dict_rows=False) as cur:
            cur.execute(
                "SELECT to_regclass('schema_migrations') IS NOT NULL"
            )
            if not cur.fetchone()[0]:
                return []
            cur.execute("SELECT id FROM schema_migrations ORDER BY id")
            return [r[0] for r in cur.fetchall()]

    def ping(self) -> float:
        """Round-trip time to the database, in seconds. Used by /readyz."""
        t0 = time.perf_counter()
        with self._cur(dict_rows=False) as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        return time.perf_counter() - t0

    # -- traces ----------------------------------------------------------------

    def put_traces(self, traces: TraceSet, source: str = "api") -> int:
        now = time.time()
        rows = [
            (t.trace_id, t.content_hash(), float(t.ts), t.session_id, t.label, source,
             self._json([turn.as_dict() for turn in t.turns]),
             self._json(json.loads(json.dumps(t.meta, default=str))), now)
            for t in traces
        ]
        with self._cur(dict_rows=False) as cur:
            cur.executemany(
                "INSERT INTO traces (trace_id, content_hash, ts, session_id, label, source, "
                "turns, meta, ingested_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (trace_id) DO UPDATE SET ts=EXCLUDED.ts, label=EXCLUDED.label, "
                "turns=EXCLUDED.turns, meta=EXCLUDED.meta",
                rows,
            )
        return len(rows)

    @staticmethod
    def _row_to_trace(row: dict) -> Trace:
        return Trace(
            trace_id=row["trace_id"], turns=row["turns"], ts=row["ts"],
            session_id=row["session_id"], label=row["label"], meta=row["meta"] or {},
        )

    def get_traces(self, limit: int | None = None, since: float | None = None,
                   until: float | None = None, source: str | None = None) -> TraceSet:
        clauses: list[str] = []
        params: list[Any] = []
        if since is not None:
            clauses.append("ts >= %s")
            params.append(since)
        if until is not None:
            clauses.append("ts < %s")
            params.append(until)
        if source is not None:
            clauses.append("source = %s")
            params.append(source)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        if limit:
            sql = (f"SELECT * FROM (SELECT * FROM traces{where} ORDER BY ts DESC LIMIT %s) s "
                   "ORDER BY ts ASC")
            params = [*params, limit]
        else:
            sql = f"SELECT * FROM traces{where} ORDER BY ts ASC"
        with self._cur() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return TraceSet([self._row_to_trace(dict(r)) for r in rows], {"source": {"kind": "store"}})

    def count_traces(self) -> int:
        with self._cur(dict_rows=False) as cur:
            cur.execute("SELECT COUNT(*) FROM traces")
            return int(cur.fetchone()[0])

    def trace_time_span(self) -> tuple[float | None, float | None]:
        with self._cur(dict_rows=False) as cur:
            cur.execute("SELECT MIN(ts), MAX(ts) FROM traces")
            row = cur.fetchone()
        return row[0], row[1]

    # -- verdicts --------------------------------------------------------------

    def put_verdict(self, trace_id: str, judge: str, label: int, score: float | None = None,
                    rationale: str | None = None, cost_usd: float = 0.0,
                    latency_s: float = 0.0) -> None:
        with self._cur(dict_rows=False) as cur:
            cur.execute(
                "INSERT INTO verdicts (trace_id, judge, label, score, rationale, cost_usd, "
                "latency_s, created_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (trace_id, judge) DO UPDATE SET label=EXCLUDED.label, "
                "score=EXCLUDED.score, rationale=EXCLUDED.rationale",
                (trace_id, judge, int(label), score, rationale, cost_usd, latency_s, time.time()),
            )

    def get_verdicts(self, judge: str | None = None) -> dict[str, dict]:
        sql = "SELECT * FROM verdicts"
        params: list[Any] = []
        if judge:
            sql += " WHERE judge = %s"
            params.append(judge)
        with self._cur() as cur:
            cur.execute(sql, params)
            return {r["trace_id"]: dict(r) for r in cur.fetchall()}

    # -- suites ----------------------------------------------------------------

    def put_suite(self, suite: EvalSuite) -> str:
        with self._cur(dict_rows=False) as cur:
            cur.execute(
                "INSERT INTO suites (name, content_hash, payload, n_cases, updated_at) "
                "VALUES (%s,%s,%s,%s,%s) ON CONFLICT (name) DO UPDATE SET "
                "content_hash=EXCLUDED.content_hash, payload=EXCLUDED.payload, "
                "n_cases=EXCLUDED.n_cases, updated_at=EXCLUDED.updated_at",
                (suite.name, suite.content_hash(), self._json(suite.as_dict()),
                 len(suite), time.time()),
            )
        return suite.content_hash()

    def get_suite(self, name: str) -> EvalSuite | None:
        with self._cur() as cur:
            cur.execute("SELECT payload FROM suites WHERE name = %s", (name,))
            row = cur.fetchone()
        if row is None:
            return None
        d = row["payload"]
        return EvalSuite([EvalCase.from_dict(c) for c in d["cases"]], d["name"], d.get("meta", {}))

    def list_suites(self) -> list[dict]:
        with self._cur() as cur:
            cur.execute("SELECT name, content_hash, n_cases, updated_at FROM suites ORDER BY name")
            return [dict(r) for r in cur.fetchall()]

    # -- proposals -------------------------------------------------------------

    def put_proposals(self, suite_name: str, proposals: list[dict]) -> int:
        now = time.time()
        with self._cur() as cur:
            cur.execute("SELECT trace_id FROM proposals WHERE suite_name = %s", (suite_name,))
            existing = {r["trace_id"] for r in cur.fetchall()}
        rows = [
            (suite_name, p["trace_id"], p.get("cluster"), p.get("reason"),
             p.get("suggested_expected"), p.get("judge_rationale"), "pending", now)
            for p in proposals if p["trace_id"] not in existing
        ]
        with self._cur(dict_rows=False) as cur:
            cur.executemany(
                "INSERT INTO proposals (suite_name, trace_id, cluster, reason, suggested, "
                "rationale, status, created_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                rows,
            )
        return len(rows)

    def get_proposals(self, suite_name: str | None = None,
                      status: str | None = "pending") -> list[dict]:
        clauses, params = [], []
        if suite_name:
            clauses.append("p.suite_name = %s")
            params.append(suite_name)
        if status:
            clauses.append("p.status = %s")
            params.append(status)
        sql = ("SELECT p.*, t.turns, t.ts, t.session_id, t.label AS human_label, t.meta "
               "FROM proposals p LEFT JOIN traces t ON t.trace_id = p.trace_id")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY p.id ASC"
        with self._cur() as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def _review(self, proposal_id: int, reviewer: str, status: str, expected: int | None) -> dict:
        if not reviewer:
            raise ValueError("a review must record who did it")
        with self._cur() as cur:
            cur.execute(
                "UPDATE proposals SET status=%s, reviewer=%s, expected=%s, reviewed_at=%s "
                "WHERE id=%s RETURNING *",
                (status, reviewer, expected, time.time(), proposal_id),
            )
            row = cur.fetchone()
        if row is None:
            raise KeyError(f"no proposal {proposal_id}")
        return dict(row)

    def confirm_proposal(self, proposal_id: int, reviewer: str, expected: int) -> dict:
        return self._review(proposal_id, reviewer, "confirmed", int(expected))

    def reject_proposal(self, proposal_id: int, reviewer: str) -> dict:
        return self._review(proposal_id, reviewer, "rejected", None)

    def confirmed_cases(self, suite_name: str) -> list[EvalCase]:
        out: list[EvalCase] = []
        for p in self.get_proposals(suite_name, status="confirmed"):
            if not p.get("turns"):
                continue
            trace = Trace(trace_id=p["trace_id"], turns=p["turns"], ts=p["ts"] or 0.0,
                          session_id=p["session_id"], label=p.get("human_label"),
                          meta=p.get("meta") or {})
            out.append(EvalCase(
                case_id=f"mined-{p['trace_id']}", trace=trace, expected=int(p["expected"]),
                provenance="confirmed", reviewer=p["reviewer"], confirmed_at=p["reviewed_at"],
                tags=("mined", f"cluster{p['cluster']}"),
                meta={"reason": p.get("reason"), "proposal_id": p["id"]},
            ))
        return out

    # -- records ---------------------------------------------------------------

    def put_record(self, kind: str, payload: dict, label: str = "") -> int:
        with self._cur(dict_rows=False) as cur:
            cur.execute(
                "INSERT INTO records (kind, label, payload, created_at) "
                "VALUES (%s,%s,%s,%s) RETURNING id",
                (kind, label, self._json(json.loads(json.dumps(payload, default=str))),
                 time.time()),
            )
            return int(cur.fetchone()[0])

    def get_records(self, kind: str | None = None, limit: int = 50) -> list[dict]:
        sql = "SELECT * FROM records"
        params: list[Any] = []
        if kind:
            sql += " WHERE kind = %s"
            params.append(kind)
        sql += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        with self._cur() as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    # -- online scores ---------------------------------------------------------

    def put_online_score(self, scorer: str, label: int, latency_us: float,
                         trace_id: str | None = None) -> None:
        with self._cur(dict_rows=False) as cur:
            cur.execute(
                "INSERT INTO online_scores (trace_id, scorer, label, latency_us, created_at) "
                "VALUES (%s,%s,%s,%s,%s)",
                (trace_id, scorer, int(label), float(latency_us), time.time()),
            )

    def online_score_stats(self) -> dict:
        with self._cur() as cur:
            cur.execute(
                "SELECT COUNT(*) n, AVG(latency_us) avg_us, AVG(label::float) pass_rate, "
                "PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_us) p95_us "
                "FROM online_scores"
            )
            row = dict(cur.fetchone())
        return {
            "n": int(row["n"] or 0),
            "avg_us": float(row["avg_us"] or 0.0),
            "p95_us": float(row["p95_us"] or 0.0),
            "pass_rate": float(row["pass_rate"]) if row["pass_rate"] is not None else None,
        }

    # -- lifecycle -------------------------------------------------------------

    def stats(self) -> StoreStats:
        out = StoreStats()
        with self._cur(dict_rows=False) as cur:
            for t in ("traces", "verdicts", "suites", "proposals", "records", "online_scores"):
                cur.execute(f"SELECT COUNT(*) FROM {t}")
                out[t] = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM proposals WHERE status='pending'")
            out["pending_proposals"] = int(cur.fetchone()[0])
        return out

    def close(self) -> None:
        self._pool.closeall()

    def __enter__(self) -> PostgresStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
