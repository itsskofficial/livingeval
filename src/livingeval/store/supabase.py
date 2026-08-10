"""Supabase, local and cloud, resolved from one place.

Supabase **is** Postgres — you connect to it with a `postgresql://` string and
`store/postgres.py` is the backend. What this module adds is the part that is
specifically Supabase and not merely Postgres:

- **Which of the three connection strings to use for which job.** Supabase hands you a
  direct URL, a session pooler and a transaction pooler, and they behave differently.
  `connection_url(purpose=...)` picks correctly instead of leaving it to be discovered.
- **Building that string from a project ref and a password**, so the only secret you
  handle is the database password rather than a URL you have to assemble by hand.
- **Detecting the local stack.** `supabase start` publishes a fixed local URL and a
  fixed service-role key. When they are present, everything resolves to local with no
  configuration, which is the whole point of developing against the real thing.

The design goal is that **nothing but credentials changes between local and cloud**.
The same migrations, the same store, the same Storage calls, the same code path — so
that "it worked locally" carries the weight it should.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

__all__ = ["LOCAL", "SupabaseConfig", "connection_url", "detect"]

#: The local stack's fixed endpoints. `supabase start` prints these; they are the same
#: on every machine, which is why they can be defaults rather than configuration.
#: The ports are livingeval's own 544xx range - see `supabase/config.toml` for why.
LOCAL = {
    "api_url": "http://127.0.0.1:54421",
    "db_url": "postgresql://postgres:postgres@127.0.0.1:54422/postgres",
    "studio_url": "http://127.0.0.1:54423",
}

#: Supabase's local demo service-role key. Identical on every machine, published in
#: their docs, and worthless off localhost - it only ever signs against the local
#: `super-secret-jwt-token...` secret. Hard-coding it is what makes `make dev` work
#: with no setup; a cloud key must never be treated this way and never appears here.
LOCAL_SERVICE_ROLE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0."
    "EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU"
)


@dataclass
class SupabaseConfig:
    """Where Supabase is and how to reach it."""

    api_url: str
    db_url: str
    service_role_key: str | None
    is_local: bool
    project_ref: str | None = None
    studio_url: str | None = None

    @property
    def storage_url(self) -> str:
        return f"{self.api_url}/storage/v1"

    def as_dict(self, reveal: bool = False) -> dict:
        from livingeval.config import redact

        return {
            "mode": "local" if self.is_local else "cloud",
            "api_url": self.api_url,
            "db_url": self.db_url if reveal else redact(self.db_url, "url"),
            "project_ref": self.project_ref,
            "service_role_key": (
                self.service_role_key if reveal else redact(self.service_role_key or "", "key")
            ),
            "studio_url": self.studio_url,
        }

    def env(self) -> dict[str, str]:
        """The environment a livingeval process needs to talk to this Supabase."""
        out = {
            "LIVINGEVAL_DATABASE_URL": self.db_url,
            "SUPABASE_URL": self.api_url,
        }
        if self.service_role_key:
            out["SUPABASE_SERVICE_ROLE_KEY"] = self.service_role_key
        return out


def connection_url(
    project_ref: str,
    password: str,
    purpose: str = "service",
    region: str = "ap-south-1",
) -> str:
    """Build the right Supabase connection string for the job.

    | purpose | what you get | when |
    |---|---|---|
    | `service` | transaction pooler, port 6543 | a long-running app: Fargate, the worker |
    | `session` | session pooler, port 5432 | migrations, psql, anything session-scoped |
    | `direct` | the database host, port 5432 | migrations when your network has IPv6 |

    The default is `service`, because that is what a deployed container needs and it is
    the one people most often get wrong: the direct host is IPv6-only on the free tier,
    and Fargate tasks in this stack have IPv4, so a direct URL simply will not resolve.

    The password is URL-encoded, which matters more than it sounds: Supabase-generated
    passwords routinely contain `@`, `/` and `?`, every one of which breaks a naively
    concatenated URL in a way that produces a confusing authentication error.
    """
    from urllib.parse import quote

    pw = quote(password, safe="")
    if purpose == "direct":
        return f"postgresql://postgres:{pw}@db.{project_ref}.supabase.co:5432/postgres"
    if purpose == "session":
        return (
            f"postgresql://postgres.{project_ref}:{pw}"
            f"@aws-0-{region}.pooler.supabase.com:5432/postgres"
        )
    if purpose == "service":
        return (
            f"postgresql://postgres.{project_ref}:{pw}"
            f"@aws-0-{region}.pooler.supabase.com:6543/postgres"
        )
    raise ValueError(f"purpose must be service, session or direct; got {purpose!r}")


def detect(require: bool = False) -> SupabaseConfig | None:
    """Work out which Supabase this process should use.

    Resolution order, most explicit first:

    1. `SUPABASE_URL` + `SUPABASE_DB_URL` — you said exactly what you want.
    2. `SUPABASE_PROJECT_REF` + `SUPABASE_DB_PASSWORD` — cloud, URL assembled for you.
    3. A reachable local stack on the ports in `supabase/config.toml`.

    Returns `None` when nothing is configured and nothing is running, unless
    `require=True`.
    """
    api_url = os.environ.get("SUPABASE_URL")
    db_url = os.environ.get("SUPABASE_DB_URL") or os.environ.get("LIVINGEVAL_DATABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    ref = os.environ.get("SUPABASE_PROJECT_REF")

    if api_url and db_url:
        local = "127.0.0.1" in api_url or "localhost" in api_url
        return SupabaseConfig(
            api_url=api_url, db_url=db_url,
            service_role_key=key or (LOCAL_SERVICE_ROLE_KEY if local else None),
            is_local=local, project_ref=ref,
        )

    password = os.environ.get("SUPABASE_DB_PASSWORD")
    if ref and password:
        return SupabaseConfig(
            api_url=api_url or f"https://{ref}.supabase.co",
            db_url=connection_url(ref, password,
                                  purpose=os.environ.get("SUPABASE_DB_PURPOSE", "service"),
                                  region=os.environ.get("SUPABASE_REGION", "ap-south-1")),
            service_role_key=key, is_local=False, project_ref=ref,
        )

    if _local_is_running():
        return SupabaseConfig(
            api_url=LOCAL["api_url"], db_url=LOCAL["db_url"],
            service_role_key=key or LOCAL_SERVICE_ROLE_KEY,
            is_local=True, studio_url=LOCAL["studio_url"],
        )

    if require:
        raise RuntimeError(
            "no Supabase found.\n"
            "  local:  npx supabase start          (or: python scripts/dev.py up)\n"
            "  cloud:  export SUPABASE_PROJECT_REF=<ref> SUPABASE_DB_PASSWORD=<password>\n"
            "          export SUPABASE_SERVICE_ROLE_KEY=<key>   # only for Storage"
        )
    return None


def _local_is_running(timeout: float = 0.6) -> bool:
    """Is the local stack up? A TCP connect to the database port, not an HTTP call:
    it answers in microseconds and does not depend on the API gateway being ready."""
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(LOCAL["db_url"])
    try:
        with socket.create_connection(
            (parsed.hostname or "127.0.0.1", parsed.port or 54422), timeout=timeout
        ):
            return True
    except OSError:
        return False
