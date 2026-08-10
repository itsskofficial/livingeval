"""What has to be true before this listens on anything but localhost.

Four pieces, each closing a hole the local version could afford to leave open.

**Authentication.** A bearer token compared with `hmac.compare_digest`, not `==`, so the
comparison does not leak the key's prefix through timing. Health endpoints stay open
because a load balancer cannot present a token, and `/` and `/review` stay open when no
key is set so the local experience is unchanged.

**Request ids and structured logs.** Every request gets an id, it comes back in
`X-Request-ID`, and it is in the log line. Without it, a log in CloudWatch is a list of
sentences with no way to group them.

**Metrics.** A Prometheus text endpoint with request counts, latency histogram and the
online-scorer numbers. Deliberately hand-rolled: `prometheus_client` would be a
dependency and a registry-lifecycle problem for about sixty lines of formatting.

**Body limits and timeouts.** `POST /v1/traces` takes a list; without a cap it takes
your memory.
"""

from __future__ import annotations

import hmac
import json
import logging
import sys
import time
import uuid
from collections import defaultdict

__all__ = ["Metrics", "configure_logging", "install"]

log = logging.getLogger("livingeval.serve")

#: Never require a token for these. A load balancer health check cannot send one, and
#: refusing it means the task never becomes healthy and the deployment rolls back
#: forever with a green application.
OPEN_PATHS = frozenset({"/healthz", "/readyz", "/metrics"})

#: Latency buckets in seconds. Sized for this workload: `/v1/score` lives in the first
#: bucket, an analysis endpoint in the last two.
BUCKETS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 30.0)


class Metrics:
    """A small Prometheus text exposition, without the client library."""

    def __init__(self) -> None:
        self.started = time.time()
        self.requests: dict[tuple[str, str, int], int] = defaultdict(int)
        self.latency_sum: dict[str, float] = defaultdict(float)
        self.latency_count: dict[str, int] = defaultdict(int)
        self.latency_buckets: dict[tuple[str, float], int] = defaultdict(int)
        self.errors: int = 0

    def observe(self, method: str, path: str, status: int, seconds: float) -> None:
        self.requests[(method, path, status)] += 1
        self.latency_sum[path] += seconds
        self.latency_count[path] += 1
        for bucket in BUCKETS:
            if seconds <= bucket:
                self.latency_buckets[(path, bucket)] += 1
        if status >= 500:
            self.errors += 1

    def render(self, extra: dict | None = None) -> str:
        out = [
            "# HELP livingeval_uptime_seconds Seconds since process start.",
            "# TYPE livingeval_uptime_seconds gauge",
            f"livingeval_uptime_seconds {time.time() - self.started:.1f}",
            "# HELP livingeval_requests_total HTTP requests by method, path and status.",
            "# TYPE livingeval_requests_total counter",
        ]
        for (method, path, status), n in sorted(self.requests.items()):
            out.append(
                f'livingeval_requests_total{{method="{method}",path="{path}",status="{status}"}} {n}'
            )
        out += [
            "# HELP livingeval_request_duration_seconds Request latency.",
            "# TYPE livingeval_request_duration_seconds histogram",
        ]
        for path, count in sorted(self.latency_count.items()):
            cumulative = 0
            for bucket in BUCKETS:
                cumulative = self.latency_buckets.get((path, bucket), 0)
                out.append(
                    f'livingeval_request_duration_seconds_bucket{{path="{path}",le="{bucket}"}} {cumulative}'
                )
            out.append(
                f'livingeval_request_duration_seconds_bucket{{path="{path}",le="+Inf"}} {count}'
            )
            out.append(
                f'livingeval_request_duration_seconds_sum{{path="{path}"}} {self.latency_sum[path]:.6f}'
            )
            out.append(f'livingeval_request_duration_seconds_count{{path="{path}"}} {count}')

        for key, value in (extra or {}).items():
            if isinstance(value, int | float):
                out += [f"# TYPE livingeval_{key} gauge", f"livingeval_{key} {value}"]
        return "\n".join(out) + "\n"


class _JsonFormatter(logging.Formatter):
    """One JSON object per line. CloudWatch Logs Insights can query it; a human can
    still read it."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("request_id", "method", "path", "status", "duration_ms", "client"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "info", json_output: bool = False) -> None:
    """Set up the root logger once. Idempotent, so calling it from both the CLI and an
    ASGI worker does not double every line."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _JsonFormatter() if json_output
        else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s  %(message)s", "%H:%M:%S")
    )
    root.addHandler(handler)
    # uvicorn's access log duplicates ours, with less in it.
    logging.getLogger("uvicorn.access").disabled = True


def install(app, settings, metrics: Metrics) -> None:
    """Attach auth, request-id/logging, body limits and CORS to a FastAPI app."""
    from fastapi import Request
    from fastapi.responses import JSONResponse

    if settings.cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    max_body = 32 * 1024 * 1024

    @app.middleware("http")
    async def _observe(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        start = time.perf_counter()

        # -- body limit ----------------------------------------------------
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > max_body:
            return JSONResponse(
                status_code=413,
                content={"detail": f"request body exceeds {max_body // (1024 * 1024)} MB"},
                headers={"X-Request-ID": request_id},
            )

        # -- auth ----------------------------------------------------------
        path = request.url.path
        if settings.api_key and path not in OPEN_PATHS:
            header = request.headers.get("authorization", "")
            token = header[7:] if header.lower().startswith("bearer ") else (
                request.headers.get("x-api-key", "")
            )
            # `compare_digest` rather than `==`: a plain comparison returns as soon as
            # two bytes differ, which leaks the prefix through response timing.
            if not token or not hmac.compare_digest(token, settings.api_key):
                metrics.observe(request.method, path, 401, time.perf_counter() - start)
                log.warning(
                    "unauthorised request",
                    extra={"request_id": request_id, "method": request.method, "path": path,
                           "client": request.client.host if request.client else None},
                )
                return JSONResponse(
                    status_code=401,
                    content={"detail": "missing or invalid credentials. Send "
                                       "`Authorization: Bearer <LIVINGEVAL_API_KEY>`."},
                    headers={"WWW-Authenticate": "Bearer", "X-Request-ID": request_id},
                )

        # -- the request ---------------------------------------------------
        try:
            response = await call_next(request)
        except Exception:
            duration = time.perf_counter() - start
            metrics.observe(request.method, path, 500, duration)
            log.exception(
                "unhandled error",
                extra={"request_id": request_id, "method": request.method, "path": path,
                       "duration_ms": round(duration * 1000, 2)},
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "internal error", "request_id": request_id},
                headers={"X-Request-ID": request_id},
            )

        duration = time.perf_counter() - start
        metrics.observe(request.method, path, response.status_code, duration)
        response.headers["X-Request-ID"] = request_id
        # Health checks fire every few seconds; logging them at info buries everything.
        level = logging.DEBUG if path in OPEN_PATHS else logging.INFO
        log.log(
            level, "%s %s -> %s", request.method, path, response.status_code,
            extra={"request_id": request_id, "method": request.method, "path": path,
                   "status": response.status_code, "duration_ms": round(duration * 1000, 2)},
        )
        return response
