"""Layer A for the deployment layer: config, DSN handling, migrations, auth, artifacts.

The tests that matter most here are the refusals. A deployment goes wrong quietly —
an unauthenticated bind, an unencrypted database connection, a migration racing itself
across two containers — and every one of those is a thing the code should refuse rather
than a thing you should remember.

The Postgres tests are marked `postgres` and skipped unless
`LIVINGEVAL_TEST_POSTGRES_URL` is set. Run them against the compose stack:

    docker compose -f deploy/docker-compose.yml up -d postgres
    LIVINGEVAL_TEST_POSTGRES_URL=postgresql://livingeval:livingeval@localhost:5433/livingeval \\
      pytest -m postgres -v
"""

from __future__ import annotations

import json
import os

import pytest

import livingeval as le
from livingeval.artifacts import LocalArtifacts, open_artifacts
from livingeval.config import Settings, redact
from livingeval.store import dsn as dsn_mod
from livingeval.store.migrations import MIGRATIONS, postgres_sql, sqlite_sql


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Settings read the real environment; a stray LIVINGEVAL_* would make these flaky."""
    for key in list(os.environ):
        if key.startswith("LIVINGEVAL_") or key == "DATABASE_URL":
            monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_defaults_are_the_local_experience():
    s = Settings.load()
    assert s.database_url.startswith("sqlite:")
    assert s.host == "127.0.0.1"
    assert s.api_key is None
    assert s.is_local_only


def test_binding_publicly_without_a_key_is_refused():
    """The single most valuable refusal here: the platform is unauthenticated by
    design when local, and shipping that on 0.0.0.0 is a hole."""
    with pytest.raises(ValueError, match="refusing to bind"):
        Settings.load(host="0.0.0.0")


def test_binding_publicly_is_allowed_with_a_key_or_an_explicit_override():
    assert Settings.load(host="0.0.0.0", api_key="x" * 32).api_key
    assert Settings.load(host="0.0.0.0", allow_insecure=True).allow_insecure


def test_a_short_api_key_is_refused_with_the_command_to_generate_one():
    with pytest.raises(ValueError, match="token_urlsafe"):
        Settings.load(host="0.0.0.0", api_key="hunter2")


@pytest.mark.parametrize("url", ["mysql://x/y", "redis://x", "./local.db"])
def test_a_nonsense_database_url_is_refused(url):
    with pytest.raises(ValueError, match=r"not a\n?\s*store URL|is not a"):
        Settings.load(database_url=url)


def test_a_nonsense_artifact_url_is_refused():
    with pytest.raises(ValueError, match="file://, s3:// or supabase://"):
        Settings.load(artifact_url="gs://bucket")


def test_environment_variables_are_read(monkeypatch):
    monkeypatch.setenv("LIVINGEVAL_DATABASE_URL", "postgresql://u:p@db.example.com/x")
    monkeypatch.setenv("LIVINGEVAL_SUITE", "support")
    monkeypatch.setenv("LIVINGEVAL_WORKER_INTERVAL_S", "60")
    s = Settings.load()
    assert s.suite_name == "support"
    assert s.worker_interval_s == 60
    assert s.uses_postgres


def test_the_supabase_transaction_pooler_is_detected():
    s = Settings.load(
        database_url="postgresql://postgres.abc:pw@aws-0-ap-south-1.pooler.supabase.com:6543/postgres"
    )
    assert s.is_supabase and s.uses_pooler


def test_a_direct_supabase_url_is_not_flagged_as_a_pooler():
    s = Settings.load(database_url="postgresql://postgres:pw@db.abc.supabase.co:5432/postgres")
    assert s.is_supabase and not s.uses_pooler


def test_status_output_redacts_credentials():
    s = Settings.load(database_url="postgresql://user:hunter2@db.example.com:5432/x",
                      host="0.0.0.0", api_key="k" * 32)
    rendered = json.dumps(s.as_dict())
    assert "hunter2" not in rendered
    assert "kkkk" not in rendered
    assert "db.example.com" in rendered, "redaction should still identify the host"


def test_sqlite_and_file_urls_are_not_redacted():
    """They carry no credentials, and hiding the path makes 'which database am I on'
    unanswerable from /v1/status."""
    assert redact("sqlite:///data/live.db", "url") == "sqlite:///data/live.db"
    assert redact("file://./results", "url") == "file://./results"


# ---------------------------------------------------------------------------
# dsn
# ---------------------------------------------------------------------------


def test_ssl_is_required_for_a_remote_host_and_not_for_localhost():
    remote = dsn_mod.normalise("postgresql://u:p@db.supabase.co:5432/postgres")
    local = dsn_mod.normalise("postgresql://u:p@localhost:5432/postgres")
    assert "sslmode=require" in remote
    assert "sslmode" not in local


def test_an_explicit_sslmode_is_left_alone():
    out = dsn_mod.normalise("postgresql://u:p@db.supabase.co/postgres?sslmode=verify-full")
    assert "sslmode=verify-full" in out
    assert out.count("sslmode") == 1


def test_normalise_adds_a_connect_timeout_and_an_application_name():
    out = dsn_mod.normalise("postgresql://u:p@db.example.com/x")
    assert "connect_timeout=10" in out
    assert "application_name=livingeval" in out


def test_non_postgres_urls_pass_through_untouched():
    assert dsn_mod.normalise("sqlite:///x.db") == "sqlite:///x.db"


def test_dsn_classification_and_description():
    pooler = dsn_mod.parse(
        "postgresql://postgres.abc:pw@aws-0-ap-south-1.pooler.supabase.com:6543/postgres"
    )
    assert pooler.is_pooler and not pooler.supports_session_state
    assert "transaction pooler" in pooler.describe()
    assert pooler.mode == "supabase-transaction-pooler"

    direct = dsn_mod.parse("postgresql://postgres:pw@db.abc.supabase.co:5432/postgres")
    assert direct.supports_session_state


def test_parse_rejects_a_non_postgres_url():
    with pytest.raises(ValueError, match="not a postgres URL"):
        dsn_mod.parse("mysql://x/y")


# ---------------------------------------------------------------------------
# migrations
# ---------------------------------------------------------------------------


def test_both_dialects_define_the_same_tables():
    """The backends share a test suite and a schema; if they drift, every
    Postgres-only bug becomes invisible until deployment."""
    def tables(sql: str) -> set[str]:
        return {
            line.split("IF NOT EXISTS")[1].split("(")[0].strip()
            for line in sql.splitlines()
            if line.strip().upper().startswith("CREATE TABLE")
        }

    pg = tables("\n".join(m.postgres for m in MIGRATIONS))
    lite = tables("\n".join(m.sqlite for m in MIGRATIONS))
    assert pg == lite, f"schema drift: postgres-only {pg - lite}, sqlite-only {lite - pg}"


def test_the_generated_supabase_sql_is_runnable_and_idempotent():
    sql = postgres_sql()
    assert "CREATE TABLE IF NOT EXISTS traces" in sql
    assert "schema_migrations" in sql
    assert "ON CONFLICT (id) DO NOTHING" in sql, "pasting it twice must be safe"


def test_the_committed_supabase_file_matches_the_code():
    """`deploy/supabase/001_init.sql` is generated. If it drifts, someone pastes an old
    schema into their project and the mismatch surfaces as a runtime error."""
    from pathlib import Path

    committed = Path(__file__).resolve().parents[2] / "deploy" / "supabase" / "001_init.sql"
    if not committed.exists():
        pytest.skip("running from an installed package, not the repository")
    assert committed.read_text(encoding="utf-8").strip() == postgres_sql().strip(), (
        "regenerate with: livingeval db sql --dialect postgres --out deploy/supabase/001_init.sql"
    )


def test_sqlite_migrations_are_recorded_and_idempotent():
    store = le.store.open_store("sqlite://:memory:")
    assert store.applied_migrations() == [m.id for m in MIGRATIONS]
    assert store.migrate() == [], "a second migrate must be a no-op"
    assert "CREATE TABLE" in sqlite_sql()
    store.close()


def test_ping_answers():
    store = le.store.open_store("sqlite://:memory:")
    assert 0 <= store.ping() < 5
    store.close()


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------


def test_local_artifacts_round_trip(tmp_path):
    store = LocalArtifacts(tmp_path)
    ref = store.put("a/b.json", {"coverage": 0.42})
    assert ref.backend == "file" and ref.size > 0
    assert store.get_json("a/b.json") == {"coverage": 0.42}
    assert store.exists("a/b.json")
    assert store.list() == ["a/b.json"]


def test_versioned_writes_keep_a_history_and_a_latest(tmp_path):
    store = LocalArtifacts(tmp_path)
    first = store.put_versioned("power.json", {"n": 1})
    second = store.put_versioned("power.json", {"n": 2})
    keys = store.list()
    assert sum(1 for k in keys if k.endswith("latest.json")) == 1
    assert len([k for k in keys if "power/" in k]) >= 3
    assert store.get_json("power/latest.json") == {"n": 2}
    assert first["latest"]["key"] == second["latest"]["key"]


def test_artifact_keys_cannot_escape_the_root(tmp_path):
    """`key` reaches this from an API request."""
    store = LocalArtifacts(tmp_path / "root")
    with pytest.raises(ValueError, match="escapes the artifact root"):
        store.put("../../etc/passwd", {"x": 1})


def test_text_and_bytes_payloads(tmp_path):
    store = LocalArtifacts(tmp_path)
    store.put("a.txt", "hello")
    store.put("b.bin", b"\x00\x01")
    assert store.get_bytes("a.txt") == b"hello"
    assert store.get_bytes("b.bin") == b"\x00\x01"


def test_open_artifacts_resolves_each_scheme(tmp_path):
    assert open_artifacts(f"file://{tmp_path}").backend == "file"
    with pytest.raises(ValueError, match="unrecognised artifact URL"):
        open_artifacts("gs://bucket/x")


def test_supabase_storage_requires_its_credentials(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SUPABASE_SERVICE_ROLE_KEY"):
        open_artifacts("supabase://artifacts/livingeval")


# ---------------------------------------------------------------------------
# the worker
# ---------------------------------------------------------------------------


@pytest.fixture
def worker_state(tmp_path):
    from livingeval.serve import seed_demo
    from livingeval.serve.app import AppState

    settings = Settings.load(
        database_url=f"sqlite:///{tmp_path}/w.db",
        artifact_url=f"file://{tmp_path}/artifacts",
    )
    state = AppState(settings=settings)
    seed_demo(state.store, n=500, suite_size=80)
    yield state
    state.close()


def test_a_worker_tick_measures_mines_and_writes_artifacts(worker_state):
    from livingeval.serve.worker import run_once

    result = run_once(worker_state, tick=1, mine_n=6, power_sims=30)
    assert not result.errors, result.errors
    assert result.traces_total == 500
    assert result.coverage is not None
    assert result.power is not None
    assert result.proposals_queued > 0
    assert result.pending_review == result.proposals_queued
    assert any("blindspots" in a for a in result.artifacts)
    assert worker_state.artifacts.get_json("default/blindspots/latest.json")["kind"] == "blindspots"


def test_the_worker_never_promotes_anything_by_itself(worker_state):
    """The guard-rail, at the one place it would be most tempting to skip: an
    unattended loop must not put unreviewed cases into a gate."""
    from livingeval.serve.worker import run_once

    before = len(worker_state.suite())
    run_once(worker_state, tick=1, mine_n=6, power_sims=20)
    assert len(worker_state.suite()) == before
    assert all(p["status"] == "pending"
               for p in worker_state.store.get_proposals("default", status=None))


def test_a_failing_stage_does_not_lose_the_stages_that_worked(worker_state, monkeypatch):
    from livingeval import power as power_mod
    from livingeval.serve.worker import run_once

    def boom(*a, **k):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(power_mod, "expected_power", boom)
    result = run_once(worker_state, tick=1, mine_n=4, power_sims=10)
    assert result.coverage is not None, "coverage succeeded and must be kept"
    assert result.power is None
    assert any("power:" in e for e in result.errors)


def test_a_worker_tick_on_an_empty_store_says_why(tmp_path):
    from livingeval.serve.app import AppState
    from livingeval.serve.worker import run_once

    state = AppState(settings=Settings.load(
        database_url=f"sqlite:///{tmp_path}/empty.db",
        artifact_url=f"file://{tmp_path}/a",
    ))
    result = run_once(state, tick=1)
    assert "too few traces" in " ".join(result.errors)
    state.close()


# ---------------------------------------------------------------------------
# postgres - only against a real server
# ---------------------------------------------------------------------------

POSTGRES_URL = os.environ.get("LIVINGEVAL_TEST_POSTGRES_URL")
requires_postgres = pytest.mark.skipif(
    not POSTGRES_URL, reason="set LIVINGEVAL_TEST_POSTGRES_URL to run these"
)


@pytest.mark.postgres
@requires_postgres
def test_postgres_round_trips_everything_sqlite_does():
    """The parity test. `store/postgres.py` shares a schema with SQLite and a protocol,
    but 'shares a protocol' is not 'has been run', and the gap shows up exactly where
    unit tests cannot reach.

    Namespaced and self-cleaning, because it runs against whatever database you point
    it at — including a local Supabase you have been developing in. A test that assumes
    an empty database fails for the wrong reason and teaches you to ignore it.
    """
    from dataclasses import replace as _replace

    from livingeval.store import open_store

    store = open_store(POSTGRES_URL)
    tag = f"parity-{os.getpid()}"
    try:
        base = le.synthetic.drifting(n=200, seed=0)
        traces = le.TraceSet([_replace(x, trace_id=f"{tag}-{x.trace_id}") for x in base])
        ids = set(traces.ids)

        assert store.put_traces(traces, source=tag) == 200
        store.put_traces(traces)  # idempotent on trace_id
        mine_only = store.get_traces(source=tag)
        assert len(mine_only) == 200, "a re-put must not duplicate"
        by_id = {x.trace_id: x for x in mine_only}
        assert by_id.keys() == ids
        first = traces.sorted_by_time()[0]
        assert by_id[first.trace_id].content_hash() == first.content_hash()

        suite = le.EvalSuite.from_traces(traces[:50], name=tag)
        store.put_suite(suite)
        assert store.get_suite(tag).content_hash() == suite.content_hash()

        rows = [{"trace_id": x.trace_id, "cluster": 0, "suggested_expected": 1}
                for x in traces[:5]]
        assert store.put_proposals(tag, rows) == 5
        assert store.put_proposals(tag, rows) == 0, "the unique index must dedupe"
        pending = store.get_proposals(tag)
        store.confirm_proposal(pending[0]["id"], reviewer="sk", expected=0)
        cases = store.confirmed_cases(tag)
        assert len(cases) == 1 and cases[0].reviewer == "sk" and cases[0].expected == 0

        store.put_record("coverage", {"coverage": 0.5}, label=tag)
        assert store.get_records("coverage")[0]["payload"]["coverage"] == 0.5
        store.put_online_score("bow", 1, 42.0)
        assert store.online_score_stats()["n"] >= 1
        assert store.applied_migrations() == [m.id for m in MIGRATIONS]
        assert store.migrate() == []
    finally:
        with store._cur(dict_rows=False) as cur:
            cur.execute("DELETE FROM proposals WHERE suite_name = %s", (tag,))
            cur.execute("DELETE FROM suites WHERE name = %s", (tag,))
            cur.execute("DELETE FROM records WHERE label = %s", (tag,))
            cur.execute("DELETE FROM traces WHERE source = %s", (tag,))
        store.close()


@pytest.mark.postgres
@requires_postgres
def test_postgres_is_safe_under_concurrent_writers():
    """A single shared psycopg2 connection is not thread-safe, and FastAPI serves on a
    thread pool. Without the pool this corrupts the protocol."""
    from concurrent.futures import ThreadPoolExecutor

    from livingeval.store import open_store

    store = open_store(POSTGRES_URL)
    try:
        def work(i: int) -> int:
            store.put_traces(le.synthetic.stable(n=20, seed=100 + i), source=f"t{i}")
            return store.count_traces()

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(work, range(8)))
        assert all(r > 0 for r in results)
        assert store.ping() < 5
    finally:
        store.close()


@pytest.mark.postgres
@requires_postgres
def test_migrating_over_a_transaction_pooler_is_refused():
    """The advisory lock would silently not be held, so the refusal is the feature.

    The URL is rewritten to port 6543 by parsing rather than by string replacement:
    the test database could be on any port, and a `.replace(":5432/", …)` that silently
    matches nothing turns this into a test of the happy path.
    """
    from urllib.parse import urlparse, urlunparse

    from livingeval.store.postgres import PostgresStore

    parsed = urlparse(POSTGRES_URL)
    pooled = urlunparse(parsed._replace(netloc=f"{parsed.username}:{parsed.password}"
                                        f"@{parsed.hostname}:6543"))
    assert dsn_mod.parse(pooled).is_pooler, "the rewrite must actually produce a pooler URL"

    store = object.__new__(PostgresStore)
    store.dsn = dsn_mod.parse(pooled)
    with pytest.raises(RuntimeError, match="transaction pooler"):
        PostgresStore.migrate(store)
