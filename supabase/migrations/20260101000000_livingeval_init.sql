-- livingeval schema. Generated from livingeval/store/migrations.py.
-- Paste into the Supabase SQL editor, or run `livingeval db migrate`.
-- Safe to run more than once.


CREATE TABLE IF NOT EXISTS schema_migrations (
    id         TEXT PRIMARY KEY,
    applied_at DOUBLE PRECISION NOT NULL
);

-- 0001_init: traces, verdicts, suites, proposals, records, online_scores

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

INSERT INTO schema_migrations (id, applied_at) VALUES ('0001_init', EXTRACT(EPOCH FROM now())) ON CONFLICT (id) DO NOTHING;
-- 0002_rls: deny the Supabase Data API; these tables are backend-only

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

INSERT INTO schema_migrations (id, applied_at) VALUES ('0002_rls', EXTRACT(EPOCH FROM now())) ON CONFLICT (id) DO NOTHING;
