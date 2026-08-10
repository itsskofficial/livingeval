"""Schema, versioned, applied explicitly.

`CREATE TABLE IF NOT EXISTS` on connect is fine for one process against one file. It is
not fine once a deployment has two tasks starting at the same time against a shared
Postgres, because "if not exists" races are silent and partial.

So the schema lives here as an ordered list of migrations, each with an id, and a
`schema_migrations` table records what has been applied. `livingeval db migrate` runs
them inside a transaction and takes an advisory lock first, so two containers booting
together do not both try.

**Migrations run against the direct connection, not the transaction pooler.** pgBouncer
in transaction mode gives each statement a different backend, so an advisory lock taken
in one statement is not held for the next — the lock would silently do nothing. The CLI
checks and refuses, with the fix printed.

The same SQL is also written to `deploy/supabase/001_init.sql` for people who would
rather paste it into the Supabase SQL editor than run a command. Both come from this
module, so they cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["LOCK_KEY", "MIGRATIONS", "Migration", "postgres_sql", "sqlite_sql"]

#: An arbitrary but fixed key for `pg_advisory_lock`. Any two livingeval processes
#: migrating the same database must pick the same number, so it is a constant.
LOCK_KEY = 8_412_733


@dataclass(frozen=True)
class Migration:
    id: str
    description: str
    postgres: str
    sqlite: str


_TRACES_PG = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id     TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    ts           DOUBLE PRECISION NOT NULL,
    session_id   TEXT,
    label        INTEGER,
    source       TEXT,
    turns        JSONB NOT NULL,
    meta         JSONB NOT NULL,
    ingested_at  DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_ts ON traces(ts);
CREATE INDEX IF NOT EXISTS idx_traces_session ON traces(session_id);
CREATE INDEX IF NOT EXISTS idx_traces_hash ON traces(content_hash);
CREATE INDEX IF NOT EXISTS idx_traces_source ON traces(source);

CREATE TABLE IF NOT EXISTS verdicts (
    trace_id   TEXT NOT NULL,
    judge      TEXT NOT NULL,
    label      INTEGER NOT NULL,
    score      DOUBLE PRECISION,
    rationale  TEXT,
    cost_usd   DOUBLE PRECISION NOT NULL DEFAULT 0,
    latency_s  DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (trace_id, judge)
);

CREATE TABLE IF NOT EXISTS suites (
    name         TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    payload      JSONB NOT NULL,
    n_cases      INTEGER NOT NULL,
    updated_at   DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS proposals (
    id          BIGSERIAL PRIMARY KEY,
    suite_name  TEXT NOT NULL,
    trace_id    TEXT NOT NULL,
    cluster     INTEGER,
    reason      TEXT,
    suggested   INTEGER,
    rationale   TEXT,
    status      TEXT NOT NULL DEFAULT 'pending',
    reviewer    TEXT,
    expected    INTEGER,
    created_at  DOUBLE PRECISION NOT NULL,
    reviewed_at DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_proposals_status ON proposals(suite_name, status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_proposals_unique ON proposals(suite_name, trace_id);

CREATE TABLE IF NOT EXISTS records (
    id         BIGSERIAL PRIMARY KEY,
    kind       TEXT NOT NULL,
    label      TEXT,
    payload    JSONB NOT NULL,
    created_at DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_records_kind ON records(kind, created_at DESC);

CREATE TABLE IF NOT EXISTS online_scores (
    id         BIGSERIAL PRIMARY KEY,
    trace_id   TEXT,
    scorer     TEXT NOT NULL,
    label      INTEGER NOT NULL,
    latency_us DOUBLE PRECISION NOT NULL,
    created_at DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_online_created ON online_scores(created_at DESC);
"""

_TRACES_SQLITE = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id     TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    ts           REAL NOT NULL,
    session_id   TEXT,
    label        INTEGER,
    source       TEXT,
    turns        TEXT NOT NULL,
    meta         TEXT NOT NULL,
    ingested_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_ts ON traces(ts);
CREATE INDEX IF NOT EXISTS idx_traces_session ON traces(session_id);
CREATE INDEX IF NOT EXISTS idx_traces_hash ON traces(content_hash);
CREATE INDEX IF NOT EXISTS idx_traces_source ON traces(source);

CREATE TABLE IF NOT EXISTS verdicts (
    trace_id   TEXT NOT NULL,
    judge      TEXT NOT NULL,
    label      INTEGER NOT NULL,
    score      REAL,
    rationale  TEXT,
    cost_usd   REAL NOT NULL DEFAULT 0,
    latency_s  REAL NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    PRIMARY KEY (trace_id, judge)
);

CREATE TABLE IF NOT EXISTS suites (
    name         TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    payload      TEXT NOT NULL,
    n_cases      INTEGER NOT NULL,
    updated_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS proposals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    suite_name  TEXT NOT NULL,
    trace_id    TEXT NOT NULL,
    cluster     INTEGER,
    reason      TEXT,
    suggested   INTEGER,
    rationale   TEXT,
    status      TEXT NOT NULL DEFAULT 'pending',
    reviewer    TEXT,
    expected    INTEGER,
    created_at  REAL NOT NULL,
    reviewed_at REAL
);
CREATE INDEX IF NOT EXISTS idx_proposals_status ON proposals(suite_name, status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_proposals_unique ON proposals(suite_name, trace_id);

CREATE TABLE IF NOT EXISTS records (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT NOT NULL,
    label      TEXT,
    payload    TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_records_kind ON records(kind, created_at DESC);

CREATE TABLE IF NOT EXISTS online_scores (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id   TEXT,
    scorer     TEXT NOT NULL,
    label      INTEGER NOT NULL,
    latency_us REAL NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_online_created ON online_scores(created_at DESC);
"""

_BOOKKEEPING_PG = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    id         TEXT PRIMARY KEY,
    applied_at DOUBLE PRECISION NOT NULL
);
"""

_BOOKKEEPING_SQLITE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    id         TEXT PRIMARY KEY,
    applied_at REAL NOT NULL
);
"""

_RLS_PG = """
-- Lock these tables to the backend.
--
-- Supabase exposes the `public` schema through PostgREST, so a table here is
-- potentially reachable with the anon key that ships in a browser. These tables hold
-- production traces - real user text - and livingeval never uses the Data API: it
-- connects over SQL as the database owner, and the service role bypasses RLS.
--
-- So: enable RLS everywhere and define **no policies**. That denies `anon` and
-- `authenticated` completely while leaving the backend untouched. Revoking the grants
-- as well is belt and braces, and both are cheap.
ALTER TABLE traces        ENABLE ROW LEVEL SECURITY;
ALTER TABLE verdicts      ENABLE ROW LEVEL SECURITY;
ALTER TABLE suites        ENABLE ROW LEVEL SECURITY;
ALTER TABLE proposals     ENABLE ROW LEVEL SECURITY;
ALTER TABLE records       ENABLE ROW LEVEL SECURITY;
ALTER TABLE online_scores ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
    REVOKE ALL ON ALL TABLES IN SCHEMA public FROM anon, authenticated;
    REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM anon, authenticated;
  END IF;
END $$;
"""

_RLS_SQLITE = "-- SQLite has no row-level security and no Data API. Nothing to do."


MIGRATIONS: list[Migration] = [
    Migration(
        id="0001_init",
        description="traces, verdicts, suites, proposals, records, online_scores",
        postgres=_TRACES_PG,
        sqlite=_TRACES_SQLITE,
    ),
    Migration(
        id="0002_rls",
        description="deny the Supabase Data API; these tables are backend-only",
        postgres=_RLS_PG,
        sqlite=_RLS_SQLITE,
    ),
]

BOOKKEEPING = {"postgres": _BOOKKEEPING_PG, "sqlite": _BOOKKEEPING_SQLITE}


def postgres_sql(include_bookkeeping: bool = True) -> str:
    """The whole schema as one script, for pasting into the Supabase SQL editor."""
    parts = [
        "-- livingeval schema. Generated from livingeval/store/migrations.py.",
        "-- Paste into the Supabase SQL editor, or run `livingeval db migrate`.",
        "-- Safe to run more than once.",
        "",
    ]
    if include_bookkeeping:
        parts.append(_BOOKKEEPING_PG)
    for m in MIGRATIONS:
        parts += [f"-- {m.id}: {m.description}", m.postgres]
        if include_bookkeeping:
            parts.append(
                "INSERT INTO schema_migrations (id, applied_at) "
                f"VALUES ('{m.id}', EXTRACT(EPOCH FROM now())) ON CONFLICT (id) DO NOTHING;"
            )
    return "\n".join(parts).strip() + "\n"


def sqlite_sql() -> str:
    return "\n".join([BOOKKEEPING["sqlite"], *[m.sqlite for m in MIGRATIONS]])
