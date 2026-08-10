"""Every setting, in one place, from the environment.

A deployed service needs its configuration to be *auditable*: one object you can print,
one place to look when something is wrong, and no `os.environ.get` scattered through
request handlers. `Settings.load()` reads the environment once and validates it, and
`/v1/status` renders it with the secrets redacted.

Three rules the validation enforces, because each one is a way a deployment quietly
becomes unsafe:

1. **Binding to anything other than localhost requires an API key.** The local platform
   is deliberately unauthenticated (DECISIONS.md #25). The moment it listens on
   `0.0.0.0` it is not local any more, and shipping it open is a mistake you make once.
   `LIVINGEVAL_ALLOW_INSECURE=1` overrides it, deliberately noisily.
2. **Secrets are never read from a file, only the environment.** Same rule as everywhere
   else in this library.
3. **A Supabase pooler URL is detected and its constraints applied**, rather than left to
   fail at the first concurrent request.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any
from urllib.parse import urlparse

__all__ = ["Settings", "load", "redact"]

#: Substrings that mark a value as secret when a settings dict is rendered.
_SECRET_HINTS = ("key", "secret", "password", "token", "dsn", "url", "credential")


def redact(value: Any, key: str = "") -> Any:
    """Blank anything that looks like a credential, keeping enough to identify it."""
    if value is None or not isinstance(value, str) or not value:
        return value
    if not any(h in key.lower() for h in _SECRET_HINTS):
        return value
    # sqlite and file URLs carry no credentials, and showing the path is what makes
    # "which database am I actually pointed at" answerable from /v1/status.
    if value.startswith(("sqlite:", "file:")):
        return value
    if "://" in value:
        # Keep the shape of a connection string, drop the credentials in it.
        try:
            parsed = urlparse(value)
            host = parsed.hostname or "?"
            port = f":{parsed.port}" if parsed.port else ""
            return f"{parsed.scheme}://***@{host}{port}{parsed.path}"
        except ValueError:
            return "***"
    return value[:4] + "…" + value[-2:] if len(value) > 10 else "***"


def _bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


@dataclass
class Settings:
    """Resolved configuration for a livingeval process."""

    # -- storage ---------------------------------------------------------------
    database_url: str = "sqlite:///livingeval.db"
    artifact_url: str = "file://./results"

    # -- what this process is about --------------------------------------------
    suite_name: str = "default"
    judge_spec: str = "oracle"
    embed_spec: str | None = None
    seed: int = 0
    trace_limit: int = 20_000

    # -- the server ------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None
    allow_insecure: bool = False
    cors_origins: tuple[str, ...] = ()
    log_json: bool = False
    log_level: str = "info"
    metrics: bool = True
    request_timeout_s: float = 120.0

    # -- the worker ------------------------------------------------------------
    worker_interval_s: float = 900.0
    worker_mine: int = 20
    worker_ingest: str | None = None

    # -- provenance ------------------------------------------------------------
    environment: str = "local"
    release: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    # -- construction ----------------------------------------------------------

    @classmethod
    def load(cls, **overrides) -> Settings:
        """Read the environment, apply overrides, then validate."""
        origins = os.environ.get("LIVINGEVAL_CORS_ORIGINS", "")
        settings = cls(
            database_url=os.environ.get("LIVINGEVAL_DATABASE_URL")
            or os.environ.get("DATABASE_URL")
            or "sqlite:///livingeval.db",
            artifact_url=os.environ.get("LIVINGEVAL_ARTIFACT_URL", "file://./results"),
            suite_name=os.environ.get("LIVINGEVAL_SUITE", "default"),
            judge_spec=os.environ.get("LIVINGEVAL_JUDGE", "oracle"),
            embed_spec=os.environ.get("LIVINGEVAL_EMBED") or None,
            seed=_int("LIVINGEVAL_SEED", 0),
            trace_limit=_int("LIVINGEVAL_TRACE_LIMIT", 20_000),
            host=os.environ.get("LIVINGEVAL_HOST", "127.0.0.1"),
            port=_int("LIVINGEVAL_PORT", 8000),
            api_key=os.environ.get("LIVINGEVAL_API_KEY") or None,
            allow_insecure=_bool("LIVINGEVAL_ALLOW_INSECURE"),
            cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
            log_json=_bool("LIVINGEVAL_LOG_JSON"),
            log_level=os.environ.get("LIVINGEVAL_LOG_LEVEL", "info").lower(),
            metrics=_bool("LIVINGEVAL_METRICS", True),
            request_timeout_s=float(os.environ.get("LIVINGEVAL_REQUEST_TIMEOUT_S", "120")),
            worker_interval_s=float(os.environ.get("LIVINGEVAL_WORKER_INTERVAL_S", "900")),
            worker_mine=_int("LIVINGEVAL_WORKER_MINE", 20),
            worker_ingest=os.environ.get("LIVINGEVAL_WORKER_INGEST") or None,
            environment=os.environ.get("LIVINGEVAL_ENV", "local"),
            release=os.environ.get("LIVINGEVAL_RELEASE") or None,
        )
        if overrides:
            settings = replace(settings, **{k: v for k, v in overrides.items() if v is not None})
        settings.validate()
        return settings

    # -- derived ---------------------------------------------------------------

    @property
    def is_local_only(self) -> bool:
        return self.host in ("127.0.0.1", "localhost", "::1")

    @property
    def uses_postgres(self) -> bool:
        return self.database_url.startswith(("postgres://", "postgresql://"))

    @property
    def is_supabase(self) -> bool:
        return "supabase" in self.database_url

    @property
    def uses_pooler(self) -> bool:
        """Supabase's transaction pooler listens on 6543 and its host contains `pooler`.

        It matters because pgBouncer in transaction mode does not keep session state:
        server-side prepared statements, `SET` that outlives a statement, and `LISTEN`
        all break. Detecting it here means the store can adapt instead of failing on the
        first concurrent request.
        """
        if not self.uses_postgres:
            return False
        # Port 6543 only: Supabase's *session* pooler is also a "pooler" host but keeps
        # session state, so hostname matching would misclassify it. See store/dsn.py.
        return urlparse(self.database_url).port == 6543

    # -- validation ------------------------------------------------------------

    def validate(self) -> Settings:
        if self.log_level not in ("critical", "error", "warning", "info", "debug", "trace"):
            raise ValueError(f"LIVINGEVAL_LOG_LEVEL={self.log_level!r} is not a log level")
        if not self.database_url.startswith(("sqlite:", "postgres://", "postgresql://")):
            raise ValueError(
                f"LIVINGEVAL_DATABASE_URL={redact(self.database_url, 'url')} is not a "
                "store URL; use sqlite:///path.db or postgresql://..."
            )
        if not self.artifact_url.startswith(("file://", "s3://", "supabase://")):
            raise ValueError(
                f"LIVINGEVAL_ARTIFACT_URL={self.artifact_url!r} must start with "
                "file://, s3:// or supabase://"
            )
        if not self.is_local_only and not self.api_key and not self.allow_insecure:
            raise ValueError(
                f"refusing to bind {self.host} with no API key.\n"
                "The platform is unauthenticated by design when it is local. Once it "
                "listens on a public interface that is a hole, so either:\n"
                "  set LIVINGEVAL_API_KEY=<a long random string>   (recommended)\n"
                "  or set LIVINGEVAL_ALLOW_INSECURE=1              (you have a reason)"
            )
        if self.api_key and len(self.api_key) < 16:
            raise ValueError(
                "LIVINGEVAL_API_KEY is shorter than 16 characters. Generate one with "
                "`python -c \"import secrets;print(secrets.token_urlsafe(32))\"`."
            )
        return self

    # -- rendering -------------------------------------------------------------

    @property
    def db_sslmode(self) -> str | None:
        """What sslmode this URL will actually connect with.

        Derived from the same function that builds the connection, so /v1/status
        reports what is true rather than what was intended.
        """
        if not self.uses_postgres:
            return None
        from urllib.parse import parse_qsl, urlparse

        from livingeval.store.dsn import normalise

        query = dict(parse_qsl(urlparse(normalise(self.database_url)).query))
        return query.get("sslmode", "prefer")

    def as_dict(self, reveal: bool = False) -> dict:
        out = {
            "environment": self.environment,
            "release": self.release,
            "database": self.database_url if reveal else redact(self.database_url, "url"),
            "database_kind": "postgres" if self.uses_postgres else "sqlite",
            "supabase": self.is_supabase,
            "pooler": self.uses_pooler,
            "db_sslmode": self.db_sslmode or ("require" if self.uses_postgres else None),
            "artifacts": self.artifact_url if reveal else redact(self.artifact_url, "url"),
            "suite": self.suite_name,
            "judge": self.judge_spec,
            "embedder": self.embed_spec or "tfidf+svd (default)",
            "host": self.host,
            "port": self.port,
            "auth": "api-key" if self.api_key else ("insecure" if not self.is_local_only else "local-only"),
            "cors_origins": list(self.cors_origins),
            "metrics": self.metrics,
            "log_json": self.log_json,
            "worker_interval_s": self.worker_interval_s,
        }
        return out

    def summary(self) -> str:  # pragma: no cover - display only
        d = self.as_dict()
        width = max(len(k) for k in d)
        return "\n".join(f"  {k:<{width}}  {v}" for k, v in d.items())


def load(**overrides) -> Settings:
    return Settings.load(**overrides)
