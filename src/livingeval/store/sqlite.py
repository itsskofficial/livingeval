"""SQLite store - the default, and the one the local platform runs on.

`sqlite3` is in the standard library, so this costs nothing to install, works on every
machine, and leaves you with a single file you can copy or delete. For a local
platform that is the right answer.

Design notes worth knowing:

- **Traces are stored by content hash as well as id.** `put_traces` is idempotent on
  `trace_id`, so replaying a JSONL file or polling Langfuse twice does not duplicate
  anything. The content hash is a separate column so the platform can spot the same
  conversation arriving under two ids.
- **Turns are stored as JSON in one column, not in a child table.** A trace is read and
  written whole, always; normalising the turns would buy a join and cost the ability to
  round-trip a trace exactly. If you later want to query inside turns, that is a view.
- **`ts` is indexed.** Every question this library asks is windowed by time.
- **Proposals carry status.** `pending` -> `confirmed` / `rejected`, with the reviewer's
  name and the label they settled on. That table *is* the human-in-the-loop queue the
  review UI renders, and the reason a mined case cannot reach a gate unreviewed.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from livingeval.store.base import StoreStats
from livingeval.store.migrations import BOOKKEEPING, MIGRATIONS
from livingeval.suite.case import EvalCase
from livingeval.suite.suite import EvalSuite
from livingeval.trace.types import Trace, TraceSet

__all__ = ["SQLiteStore"]

class SQLiteStore:
    """A local store in one file."""

    def __init__(self, path: str | Path = "livingeval.db"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # WAL lets the dashboard read while the ingester writes, which is the whole
        # point of having a store rather than a list in memory.
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.migrate()

    # -- schema ----------------------------------------------------------------

    def migrate(self) -> list[str]:
        """Apply pending migrations. Shares the migration list with Postgres, so the
        two backends cannot drift apart silently."""
        applied: list[str] = []
        self.conn.executescript(BOOKKEEPING["sqlite"])
        done = {r[0] for r in self.conn.execute("SELECT id FROM schema_migrations")}
        for m in MIGRATIONS:
            if m.id in done:
                continue
            self.conn.executescript(m.sqlite)
            self.conn.execute(
                "INSERT OR IGNORE INTO schema_migrations (id, applied_at) VALUES (?,?)",
                (m.id, time.time()),
            )
            applied.append(m.id)
        self.conn.commit()
        return applied

    def applied_migrations(self) -> list[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT id FROM schema_migrations ORDER BY id")]

    def ping(self) -> float:
        """Round-trip time to the database, in seconds. Used by /readyz."""
        t0 = time.perf_counter()
        self.conn.execute("SELECT 1").fetchone()
        return time.perf_counter() - t0

    # -- traces ----------------------------------------------------------------

    def put_traces(self, traces: TraceSet, source: str = "api") -> int:
        now = time.time()
        rows = [
            (
                t.trace_id, t.content_hash(), float(t.ts), t.session_id, t.label, source,
                json.dumps([turn.as_dict() for turn in t.turns], ensure_ascii=False),
                json.dumps(t.meta, ensure_ascii=False, default=str), now,
            )
            for t in traces
        ]
        cur = self.conn.executemany(
            "INSERT OR REPLACE INTO traces "
            "(trace_id, content_hash, ts, session_id, label, source, turns, meta, ingested_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()
        return cur.rowcount if cur.rowcount and cur.rowcount > 0 else len(rows)

    def _row_to_trace(self, row: sqlite3.Row) -> Trace:
        return Trace(
            trace_id=row["trace_id"],
            turns=json.loads(row["turns"]),
            ts=row["ts"],
            session_id=row["session_id"],
            label=row["label"],
            meta=json.loads(row["meta"]),
        )

    def get_traces(
        self,
        limit: int | None = None,
        since: float | None = None,
        until: float | None = None,
        source: str | None = None,
    ) -> TraceSet:
        clauses: list[str] = []
        params: list[Any] = []
        if since is not None:
            clauses.append("ts >= ?")
            params.append(since)
        if until is not None:
            clauses.append("ts < ?")
            params.append(until)
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        sql = "SELECT * FROM traces"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY ts ASC"
        if limit:
            # Newest `limit` traces, still returned oldest-first: a window analysis
            # reads forward in time, so returning them backwards would silently
            # reverse every drift curve.
            sql = (
                "SELECT * FROM (SELECT * FROM traces"
                + (" WHERE " + " AND ".join(clauses) if clauses else "")
                + " ORDER BY ts DESC LIMIT ?) ORDER BY ts ASC"
            )
            params = [*params, limit]
        rows = self.conn.execute(sql, params).fetchall()
        return TraceSet([self._row_to_trace(r) for r in rows], {"source": {"kind": "store"}})

    def count_traces(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0])

    def trace_time_span(self) -> tuple[float | None, float | None]:
        row = self.conn.execute("SELECT MIN(ts), MAX(ts) FROM traces").fetchone()
        return row[0], row[1]

    # -- verdicts --------------------------------------------------------------

    def put_verdict(self, trace_id: str, judge: str, label: int, score: float | None = None,
                    rationale: str | None = None, cost_usd: float = 0.0,
                    latency_s: float = 0.0) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO verdicts "
            "(trace_id, judge, label, score, rationale, cost_usd, latency_s, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (trace_id, judge, int(label), score, rationale, cost_usd, latency_s, time.time()),
        )
        self.conn.commit()

    def get_verdicts(self, judge: str | None = None) -> dict[str, dict]:
        sql = "SELECT * FROM verdicts"
        params: list[Any] = []
        if judge:
            sql += " WHERE judge = ?"
            params.append(judge)
        return {r["trace_id"]: dict(r) for r in self.conn.execute(sql, params).fetchall()}

    # -- suites ----------------------------------------------------------------

    def put_suite(self, suite: EvalSuite) -> str:
        self.conn.execute(
            "INSERT OR REPLACE INTO suites (name, content_hash, payload, n_cases, updated_at) "
            "VALUES (?,?,?,?,?)",
            (suite.name, suite.content_hash(),
             json.dumps(suite.as_dict(), ensure_ascii=False), len(suite), time.time()),
        )
        self.conn.commit()
        return suite.content_hash()

    def get_suite(self, name: str) -> EvalSuite | None:
        row = self.conn.execute("SELECT payload FROM suites WHERE name = ?", (name,)).fetchone()
        if row is None:
            return None
        d = json.loads(row["payload"])
        return EvalSuite([EvalCase.from_dict(c) for c in d["cases"]], d["name"], d.get("meta", {}))

    def list_suites(self) -> list[dict]:
        return [
            {k: r[k] for k in ("name", "content_hash", "n_cases", "updated_at")}
            for r in self.conn.execute(
                "SELECT name, content_hash, n_cases, updated_at FROM suites ORDER BY name"
            ).fetchall()
        ]

    # -- proposals -------------------------------------------------------------

    def put_proposals(self, suite_name: str, proposals: list[dict]) -> int:
        now = time.time()
        rows = [
            (suite_name, p["trace_id"], p.get("cluster"), p.get("reason"),
             p.get("suggested_expected"), p.get("judge_rationale"), "pending", now)
            for p in proposals
        ]
        # Skip traces already queued or reviewed for this suite: the mining loop runs
        # repeatedly and a growing pile of duplicates makes the review queue useless.
        existing = {
            r["trace_id"]
            for r in self.conn.execute(
                "SELECT trace_id FROM proposals WHERE suite_name = ?", (suite_name,)
            ).fetchall()
        }
        rows = [r for r in rows if r[1] not in existing]
        self.conn.executemany(
            "INSERT INTO proposals "
            "(suite_name, trace_id, cluster, reason, suggested, rationale, status, created_at) "
            "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT (suite_name, trace_id) DO NOTHING",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def get_proposals(self, suite_name: str | None = None,
                      status: str | None = "pending") -> list[dict]:
        clauses, params = [], []
        if suite_name:
            clauses.append("p.suite_name = ?")
            params.append(suite_name)
        if status:
            clauses.append("p.status = ?")
            params.append(status)
        sql = (
            "SELECT p.*, t.turns, t.ts, t.session_id, t.label AS human_label, t.meta "
            "FROM proposals p LEFT JOIN traces t ON t.trace_id = p.trace_id"
        )
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY p.id ASC"
        out = []
        for r in self.conn.execute(sql, params).fetchall():
            d = dict(r)
            d["turns"] = json.loads(d["turns"]) if d.get("turns") else []
            d["meta"] = json.loads(d["meta"]) if d.get("meta") else {}
            out.append(d)
        return out

    def _review(self, proposal_id: int, reviewer: str, status: str,
                expected: int | None) -> dict:
        if not reviewer:
            raise ValueError("a review must record who did it")
        self.conn.execute(
            "UPDATE proposals SET status = ?, reviewer = ?, expected = ?, reviewed_at = ? "
            "WHERE id = ?",
            (status, reviewer, expected, time.time(), proposal_id),
        )
        self.conn.commit()
        row = self.conn.execute("SELECT * FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
        if row is None:
            raise KeyError(f"no proposal {proposal_id}")
        return dict(row)

    def confirm_proposal(self, proposal_id: int, reviewer: str, expected: int) -> dict:
        return self._review(proposal_id, reviewer, "confirmed", int(expected))

    def reject_proposal(self, proposal_id: int, reviewer: str) -> dict:
        return self._review(proposal_id, reviewer, "rejected", None)

    def confirmed_cases(self, suite_name: str) -> list[EvalCase]:
        """Confirmed proposals, materialised as gateable cases.

        This is the write-back half of the loop: reviewed proposals become cases whose
        provenance records who signed off and what they decided.
        """
        out: list[EvalCase] = []
        for p in self.get_proposals(suite_name, status="confirmed"):
            if not p.get("turns"):
                continue
            trace = Trace(
                trace_id=p["trace_id"], turns=p["turns"], ts=p["ts"] or 0.0,
                session_id=p["session_id"], label=p.get("human_label"), meta=p.get("meta", {}),
            )
            out.append(
                EvalCase(
                    case_id=f"mined-{p['trace_id']}",
                    trace=trace,
                    expected=int(p["expected"]),
                    provenance="confirmed",
                    reviewer=p["reviewer"],
                    confirmed_at=p["reviewed_at"],
                    tags=("mined", f"cluster{p['cluster']}"),
                    meta={"reason": p.get("reason"), "proposal_id": p["id"]},
                )
            )
        return out

    # -- records ---------------------------------------------------------------

    def put_record(self, kind: str, payload: dict, label: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO records (kind, label, payload, created_at) VALUES (?,?,?,?)",
            (kind, label, json.dumps(payload, ensure_ascii=False, default=str), time.time()),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def get_records(self, kind: str | None = None, limit: int = 50) -> list[dict]:
        sql = "SELECT * FROM records"
        params: list[Any] = []
        if kind:
            sql += " WHERE kind = ?"
            params.append(kind)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        out = []
        for r in self.conn.execute(sql, params).fetchall():
            d = dict(r)
            d["payload"] = json.loads(d["payload"])
            out.append(d)
        return out

    # -- online scores ---------------------------------------------------------

    def put_online_score(self, scorer: str, label: int, latency_us: float,
                         trace_id: str | None = None) -> None:
        self.conn.execute(
            "INSERT INTO online_scores (trace_id, scorer, label, latency_us, created_at) "
            "VALUES (?,?,?,?,?)",
            (trace_id, scorer, int(label), float(latency_us), time.time()),
        )
        self.conn.commit()

    def online_score_stats(self) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*) n, AVG(latency_us) avg_us, AVG(label) pass_rate "
            "FROM online_scores"
        ).fetchone()
        p95 = self.conn.execute(
            "SELECT latency_us FROM online_scores ORDER BY latency_us "
            "LIMIT 1 OFFSET (SELECT CAST(COUNT(*) * 0.95 AS INTEGER) FROM online_scores)"
        ).fetchone()
        return {
            "n": int(row["n"] or 0),
            "avg_us": float(row["avg_us"] or 0.0),
            "p95_us": float(p95["latency_us"]) if p95 else 0.0,
            "pass_rate": float(row["pass_rate"]) if row["pass_rate"] is not None else None,
        }

    # -- lifecycle -------------------------------------------------------------

    def stats(self) -> StoreStats:
        tables = ("traces", "verdicts", "suites", "proposals", "records", "online_scores")
        out = StoreStats()
        for t in tables:
            out[t] = int(self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
        out["pending_proposals"] = int(
            self.conn.execute("SELECT COUNT(*) FROM proposals WHERE status='pending'").fetchone()[0]
        )
        return out

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
