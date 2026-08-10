# livingeval — the API, the dashboard and the worker in one image.
#
# Multi-stage so the runtime layer carries no compiler and no build cache. The result is
# ~350 MB with the serve + postgres + aws extras, which is small enough that an ECS
# rolling deployment is quick and large enough that pinning matters.
#
#   docker build -f deploy/Dockerfile -t livingeval .
#   docker run --rm -p 8000:8000 -e LIVINGEVAL_ALLOW_INSECURE=1 livingeval
#
# The same image runs both processes; the compose file and the ECS task definition just
# override the command:
#
#   api    -> uvicorn livingeval.serve.asgi:app   (the default CMD)
#   worker -> livingeval worker

# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# psycopg2-binary ships wheels, but scipy/scikit-learn need a toolchain on some
# platforms. Confined to the builder stage so none of it reaches the runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

# Install into a virtualenv we can copy wholesale. `[all]` deliberately excludes
# `unsloth`, which needs CUDA and would fail on a CPU build host.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install ".[serve,postgres,aws,figures,openai,anthropic]"

# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Containers get their configuration from the environment, and a container
    # filesystem does not survive a redeploy — so JSON logs and a non-local bind are
    # the defaults here rather than in the library.
    LIVINGEVAL_HOST=0.0.0.0 \
    LIVINGEVAL_PORT=8000 \
    LIVINGEVAL_LOG_JSON=1 \
    LIVINGEVAL_ENV=container \
    LIVINGEVAL_CACHE=/tmp/livingeval-cache

# `curl` for the container health check. Nothing else: every package in a runtime image
# is a CVE you inherit.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 livingeval

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
USER livingeval

EXPOSE 8000

# Hits /healthz, which deliberately does not touch the database — a liveness probe that
# fails on a database blip gets a healthy process killed.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

CMD ["uvicorn", "livingeval.serve.asgi:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--timeout-keep-alive", "65"]
