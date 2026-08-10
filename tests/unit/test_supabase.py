"""Supabase: URL construction offline, and the real stack when it is running.

Two halves.

The **offline** tests check connection-string construction and detection. They run on
every commit because getting the wrong one of Supabase's three URLs is the single most
common way a working local app fails in production, and it fails at the first
concurrent request rather than at deploy time.

The **stack** tests are marked `supabase_stack` and run against a live local Supabase:

    python scripts/dev.py up
    python scripts/dev.py test

They cover what unit tests structurally cannot — that RLS actually denies the anon key,
that the Storage bucket is actually private, that migrations actually apply. Those are
security properties, and a security property nobody executed is a comment.
"""

from __future__ import annotations

import os
from urllib.parse import parse_qsl, urlparse

import pytest

import livingeval as le
from livingeval.store import supabase as sb

# ---------------------------------------------------------------------------
# offline: connection strings and detection
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(("SUPABASE_", "LIVINGEVAL_")) or key == "DATABASE_URL":
            monkeypatch.delenv(key, raising=False)


def test_the_service_url_is_the_transaction_pooler():
    """What a Fargate task needs. The direct host is IPv6-only on the free tier, so a
    task with IPv4 cannot even resolve it."""
    url = sb.connection_url("abcdefghijklmnop", "pw", purpose="service")
    parsed = urlparse(url)
    assert parsed.port == 6543
    assert "pooler.supabase.com" in parsed.hostname
    assert parsed.username == "postgres.abcdefghijklmnop"


def test_the_session_url_is_the_pooler_on_5432():
    url = sb.connection_url("abc", "pw", purpose="session")
    assert urlparse(url).port == 5432
    assert "pooler.supabase.com" in url


def test_the_direct_url_targets_the_database_host():
    url = sb.connection_url("abc", "pw", purpose="direct")
    assert urlparse(url).hostname == "db.abc.supabase.co"
    assert urlparse(url).port == 5432


def test_an_unknown_purpose_is_refused():
    with pytest.raises(ValueError, match="service, session or direct"):
        sb.connection_url("abc", "pw", purpose="readonly")


def test_awkward_passwords_are_url_encoded():
    """Supabase-generated passwords routinely contain `@`, `/` and `?`. Concatenating
    one naively produces an authentication error that points nowhere near the cause."""
    from urllib.parse import unquote

    url = sb.connection_url("abc", "p@ss/w?rd#1", purpose="service")
    parsed = urlparse(url)
    # urlparse does not decode; the point is that the host survives, which it would not
    # if the `@` and `/` had been left raw.
    assert unquote(parsed.password) == "p@ss/w?rd#1"
    assert parsed.hostname == "aws-0-ap-south-1.pooler.supabase.com"
    assert parsed.port == 6543


def test_the_region_is_configurable():
    assert "aws-0-us-east-1" in sb.connection_url("abc", "pw", region="us-east-1")


def test_the_service_url_is_recognised_as_a_pooler_downstream():
    """The construction and the detection must agree, or `db migrate` refuses the wrong
    URLs."""
    from livingeval.store import dsn as dsn_mod

    parsed = dsn_mod.parse(sb.connection_url("abc", "pw", purpose="service"))
    assert parsed.is_pooler and parsed.is_supabase and not parsed.supports_session_state

    session = dsn_mod.parse(sb.connection_url("abc", "pw", purpose="session"))
    assert session.supports_session_state


def test_a_supabase_url_gets_ssl_required():
    from livingeval.store import dsn as dsn_mod

    url = dsn_mod.normalise(sb.connection_url("abc", "pw"))
    assert dict(parse_qsl(urlparse(url).query))["sslmode"] == "require"


def test_detect_prefers_explicit_urls(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://abc.supabase.co")
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://u:p@aws-0-x.pooler.supabase.com:6543/postgres")
    config = sb.detect()
    assert config is not None and not config.is_local
    assert config.env()["LIVINGEVAL_DATABASE_URL"].endswith("6543/postgres")


def test_detect_assembles_a_cloud_url_from_a_project_ref(monkeypatch):
    monkeypatch.setenv("SUPABASE_PROJECT_REF", "abcdefgh")
    monkeypatch.setenv("SUPABASE_DB_PASSWORD", "secret")
    config = sb.detect()
    assert config is not None
    assert config.project_ref == "abcdefgh"
    assert ":6543/" in config.db_url, "a deployed service wants the transaction pooler"
    assert config.api_url == "https://abcdefgh.supabase.co"


def test_detect_reports_local_when_the_stack_is_up(monkeypatch):
    monkeypatch.setattr(sb, "_local_is_running", lambda timeout=0.6: True)
    config = sb.detect()
    assert config is not None and config.is_local
    assert config.db_url == sb.LOCAL["db_url"]
    assert config.service_role_key == sb.LOCAL_SERVICE_ROLE_KEY


def test_detect_explains_itself_when_nothing_is_configured(monkeypatch):
    monkeypatch.setattr(sb, "_local_is_running", lambda timeout=0.6: False)
    assert sb.detect() is None
    with pytest.raises(RuntimeError, match="supabase start"):
        sb.detect(require=True)


def test_the_config_redacts_its_secrets(monkeypatch):
    monkeypatch.setenv("SUPABASE_PROJECT_REF", "abcdefgh")
    monkeypatch.setenv("SUPABASE_DB_PASSWORD", "hunter2")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sb_secret_notarealkey_1234567890")
    rendered = str(sb.detect().as_dict())
    assert "hunter2" not in rendered
    assert "notarealkey" not in rendered


# ---------------------------------------------------------------------------
# against the running local stack
# ---------------------------------------------------------------------------

STACK = pytest.mark.skipif(
    not sb._local_is_running(),
    reason="local Supabase is not running; `python scripts/dev.py up`",
)
pytestmark_stack = pytest.mark.supabase_stack


@pytest.fixture
def stack(monkeypatch):
    for key in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"):
        monkeypatch.delenv(key, raising=False)
    config = sb.detect(require=True)
    assert config is not None and config.is_local
    return config


@pytest.mark.supabase_stack
@STACK
def test_migrations_are_applied_on_the_local_stack(stack):
    from livingeval.store import open_store
    from livingeval.store.migrations import MIGRATIONS

    store = open_store(stack.db_url)
    try:
        assert store.applied_migrations() == [m.id for m in MIGRATIONS]
        assert store.migrate() == []
    finally:
        store.close()


@pytest.mark.supabase_stack
@STACK
def test_row_level_security_is_on_for_every_livingeval_table(stack):
    from livingeval.store import open_store

    store = open_store(stack.db_url)
    try:
        with store._cur(dict_rows=False) as cur:
            cur.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname='public' "
                "AND rowsecurity = false AND tablename = ANY(%s)",
                (["traces", "verdicts", "suites", "proposals", "records", "online_scores"],),
            )
            unprotected = [r[0] for r in cur.fetchall()]
        assert unprotected == [], f"RLS is off for {unprotected}: the Data API can read them"
    finally:
        store.close()


@pytest.mark.supabase_stack
@STACK
def test_the_anon_key_cannot_read_traces(stack):
    """The security property that matters. These tables hold production traces, the anon
    key ships in browsers, and `public` is exposed through PostgREST by default."""
    import urllib.error
    import urllib.request

    anon = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9."
        "CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0"
    )
    req = urllib.request.Request(f"{stack.api_url}/rest/v1/traces?select=trace_id&limit=1")
    req.add_header("apikey", anon)
    req.add_header("Authorization", f"Bearer {anon}")
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(req, timeout=10)
    assert caught.value.code in (401, 403), f"anon got {caught.value.code}, expected a denial"


@pytest.mark.supabase_stack
@STACK
def test_storage_round_trips_and_the_bucket_is_private(stack):
    import urllib.error
    import urllib.request

    from livingeval.artifacts import SupabaseArtifacts

    artifacts = SupabaseArtifacts("livingeval-artifacts", "pytest",
                                  url=stack.api_url, service_key=stack.service_role_key)
    artifacts.create_bucket(public=False)
    ref = artifacts.put("probe.json", {"hello": "supabase"})
    assert artifacts.get_json("probe.json") == {"hello": "supabase"}
    assert ref.uri.startswith("supabase://livingeval-artifacts/")

    # Unauthenticated public read must fail: artifacts derive from production traces.
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(
            f"{stack.api_url}/storage/v1/object/public/livingeval-artifacts/pytest/probe.json",
            timeout=10,
        )


@pytest.mark.supabase_stack
@STACK
def test_the_whole_platform_runs_on_the_local_stack(stack, monkeypatch):
    """The point of the exercise: same code path as production, only credentials differ."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from livingeval.config import Settings
    from livingeval.serve import seed_demo
    from livingeval.serve.app import AppState, create_app
    from livingeval.serve.worker import run_once

    settings = Settings.load(
        database_url=stack.db_url,
        artifact_url="supabase://livingeval-artifacts/pytest-run",
    )
    monkeypatch.setenv("SUPABASE_URL", stack.api_url)
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", stack.service_role_key)

    state = AppState(settings=settings)
    try:
        suite_name = "pytest-suite"
        state.suite_name = suite_name
        if state.store.get_suite(suite_name) is None:
            seed_demo(state.store, n=400, suite_size=80, suite_name=suite_name)

        client = TestClient(create_app(state))
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").json()["checks"]["database"]["ok"] is True

        coverage = client.get("/v1/coverage").json()
        assert 0.0 <= coverage["coverage"] <= 1.0

        state.refit_scorer(n_boot=50)
        scored = client.post("/v1/score", json={"text": "I am unable to verify that."}).json()
        assert scored["label"] in (0, 1)

        result = run_once(state, tick=1, mine_n=4, power_sims=20)
        assert not result.errors, result.errors
        assert any("blindspots" in a for a in result.artifacts)
        assert state.artifacts.get_json(f"{suite_name}/blindspots/latest.json")["kind"] == "blindspots"
    finally:
        state.close()


@pytest.mark.supabase_stack
@STACK
def test_the_store_survives_concurrent_writers_on_supabase(stack):
    from concurrent.futures import ThreadPoolExecutor

    from livingeval.store import open_store

    store = open_store(stack.db_url)
    try:
        def work(i: int) -> int:
            store.put_traces(le.synthetic.stable(n=15, seed=500 + i), source=f"conc{i}")
            return store.count_traces()

        with ThreadPoolExecutor(max_workers=8) as pool:
            assert all(n > 0 for n in pool.map(work, range(8)))
    finally:
        store.close()
