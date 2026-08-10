"""Turning what Supabase gives you into a connection that actually works.

Supabase hands you three different connection strings and the difference between them is
not cosmetic:

| what the dashboard calls it | port | what it is |
|---|---|---|
| Direct connection | 5432 | a real Postgres session. IPv6-only on the free tier. |
| Session pooler | 5432 | pgBouncer in *session* mode. Behaves like Postgres. IPv4. |
| Transaction pooler | 6543 | pgBouncer in *transaction* mode. **Different semantics.** |

Transaction mode hands your connection back to the pool after every statement, so
anything that lives in a session is gone: server-side prepared statements, `SET` outside
a transaction, `LISTEN`, temporary tables, and advisory locks. Code written against a
direct connection appears to work, then fails under concurrency with
`prepared statement "S_1" already exists` — a message that tells you nothing about the
cause.

So this module does three things:

1. **Detects which one you handed it** and reports it, rather than discovering it at
   the first concurrent request.
2. **Applies the constraints**: no server-side prepared statements on a pooler, and
   `sslmode=require` on any hosted Postgres, because Supabase requires TLS and psycopg2
   defaults to `prefer` — which silently accepts an unencrypted connection if the server
   allows one.
3. **Recommends the right URL for the job.** For a long-lived ECS task the transaction
   pooler is right. For running migrations it is wrong: `CREATE TABLE` inside an
   implicit transaction across a pooler is fine, but `CREATE INDEX CONCURRENTLY` and
   anything session-scoped is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

__all__ = ["DSN", "normalise", "parse"]

POOLER_PORT = 6543
DIRECT_PORT = 5432


@dataclass(frozen=True)
class DSN:
    """A parsed Postgres connection string and what we can tell about it."""

    url: str
    host: str
    port: int
    database: str
    user: str
    is_supabase: bool
    #: True only for **transaction** mode, which is port 6543. A pgBouncer host on
    #: 5432 is *session* mode and behaves like ordinary Postgres.
    is_pooler: bool
    #: True for any pgBouncer host, in either mode. Informational.
    is_pgbouncer: bool
    sslmode: str

    @property
    def mode(self) -> str:
        if not self.is_supabase:
            return "postgres"
        if self.is_pooler:
            return "supabase-transaction-pooler"
        if self.is_pgbouncer:
            return "supabase-session-pooler"
        return "supabase-direct"

    @property
    def supports_session_state(self) -> bool:
        """False on a transaction pooler: prepared statements and session `SET` are out."""
        return not self.is_pooler

    def describe(self) -> str:  # pragma: no cover - display only
        lines = [
            f"  host      {self.host}:{self.port}",
            f"  database  {self.database}   user {self.user}",
            f"  mode      {self.mode}",
            f"  sslmode   {self.sslmode}",
        ]
        if self.is_pooler:
            lines.append(
                "  note      transaction pooler: no server-side prepared statements, no\n"
                "            session state. Correct for a long-lived service; use the\n"
                "            direct/session URL (port 5432) to run migrations."
            )
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "host": self.host, "port": self.port, "database": self.database,
            "mode": self.mode, "sslmode": self.sslmode, "is_pooler": self.is_pooler,
            "is_pgbouncer": self.is_pgbouncer, "is_supabase": self.is_supabase,
        }


def parse(url: str) -> DSN:
    """Parse a Postgres URL and classify it."""
    parsed = urlparse(url)
    if parsed.scheme not in ("postgres", "postgresql"):
        raise ValueError(f"not a postgres URL: {parsed.scheme}://…")
    query = dict(parse_qsl(parsed.query))
    host = parsed.hostname or "localhost"
    port = parsed.port or DIRECT_PORT
    is_supabase = "supabase" in host
    return DSN(
        url=url,
        host=host,
        port=port,
        database=(parsed.path or "/postgres").lstrip("/") or "postgres",
        user=parsed.username or "postgres",
        is_supabase=is_supabase,
        # Transaction mode is identified by the **port**, not by the hostname.
        # Supabase serves both poolers from a host containing "pooler", and only the
        # 6543 one drops session state. Keying on the hostname would misclassify the
        # session pooler and make `db migrate` refuse the very URL its own error
        # message recommends.
        is_pooler=port == POOLER_PORT,
        is_pgbouncer="pooler" in host,
        sslmode=query.get("sslmode", "prefer"),
    )


#: Hosts that are a development loopback by construction: localhost, the compose
#: service names used in `deploy/docker-compose.yml`, and Docker Desktop's gateway back
#: to the host. None of these can carry traffic off the machine, so requiring TLS on
#: them only breaks local Postgres installs, which mostly ship without it.
LOCAL_HOSTS = frozenset({
    "localhost", "127.0.0.1", "::1", "postgres", "db", "host.docker.internal",
})


def is_local_host(host: str) -> bool:
    return host in LOCAL_HOSTS or host.endswith(".localhost")


def normalise(url: str, require_ssl: bool | None = None) -> str:
    """Return the URL psycopg2 should actually be given.

    Adds `sslmode=require` for any non-local host unless one is already set. psycopg2
    defaults to `prefer`, which falls back to an unencrypted connection without telling
    you — acceptable against localhost, not acceptable against a hosted database holding
    production traces.

    Override with `LIVINGEVAL_DB_SSLMODE` (or an explicit `?sslmode=` in the URL) when
    you have a Postgres on a private network that genuinely does not speak TLS. It is an
    override rather than a default because the failure mode of getting this wrong is
    silent.
    """
    import os

    parsed = urlparse(url)
    if parsed.scheme not in ("postgres", "postgresql"):
        return url
    query = dict(parse_qsl(parsed.query))
    host = parsed.hostname or "localhost"

    override = os.environ.get("LIVINGEVAL_DB_SSLMODE")
    if override and "sslmode" not in query:
        query["sslmode"] = override
    want_ssl = (not is_local_host(host)) if require_ssl is None else require_ssl
    if want_ssl and "sslmode" not in query:
        query["sslmode"] = "require"
    # `connect_timeout` so a wrong host fails in seconds rather than hanging a container
    # start until the orchestrator's health check gives up on it.
    query.setdefault("connect_timeout", "10")
    query.setdefault("application_name", "livingeval")
    return urlunparse(parsed._replace(query=urlencode(query)))
