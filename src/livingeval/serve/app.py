"""The local platform: a FastAPI app over the store, the analyses and the scorer.

    livingeval serve --db livingeval.db --demo
    # -> http://127.0.0.1:8000

What it is for. Two things a library cannot do on its own:

1. **Per-turn online scoring as a service.** `POST /v1/score` answers in microseconds
   from a fitted rung held in memory. That is the endpoint the whole judge-complexity
   ladder exists to justify — the survey finding is that nobody runs per-turn
   evaluation, and the reason was always that the judge had to come along.
2. **Human-in-the-loop review that a human will actually do.** The mining loop produces
   proposals; approving them from a Python REPL is a chore nobody repeats. `/review`
   is two buttons and a keyboard shortcut.

Deliberately local. It binds `127.0.0.1` by default, has no authentication, and stores
everything in one SQLite file. It is a control surface for your own machine, not a
multi-tenant service, and pretending otherwise would be the point at which this became
a platform competing with four funded companies.

Analyses run **on request and are cached in the store**, not on a background schedule.
Coverage over 20k traces takes a couple of seconds; a scheduler would add a
configuration surface and a class of "why is this number stale" bug in exchange for
nothing at this size.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from livingeval.mine.cluster import Clustering
    from livingeval.mine.space import Space
    from livingeval.serve.scorer import DeployedScorer

from livingeval._version import __version__
from livingeval.config import Settings

log = logging.getLogger("livingeval.serve")

__all__ = ["AppState", "create_app"]


def _require_fastapi():
    try:
        from fastapi import FastAPI  # noqa: F401
    except ImportError as e:  # pragma: no cover - optional extra
        raise ImportError("pip install 'livingeval[serve]'") from e


class AppState:
    """Everything the request handlers share.

    The fitted scorer lives here rather than being rebuilt per request, which is the
    difference between a microsecond and a second. `refit()` rebuilds it, and the
    dashboard exposes that as a button so a demo can show the loop closing.

    **One scorer per process.** Running multiple uvicorn workers gives each its own
    copy, fitted independently from the same store, so they agree but are not shared. A
    `POST /v1/scorer/refit` reaches exactly one of them. That is a real limitation, it
    is in ROADMAP.md, and the honest fix is a shared model cache rather than pretending
    a per-process object is global.
    """

    def __init__(self, db: str | None = None, suite_name: str | None = None,
                 judge_spec: str | None = None, embed_spec: str | None = None,
                 seed: int | None = None, settings: Settings | None = None):
        from livingeval.artifacts import open_artifacts
        from livingeval.store import open_store

        self.settings = settings or Settings.load(
            database_url=db, suite_name=suite_name, judge_spec=judge_spec,
            embed_spec=embed_spec, seed=seed,
        )
        self.store = open_store(self.settings.database_url)
        self.artifacts = open_artifacts(self.settings.artifact_url)
        # Kept as attributes because the handlers and the tests read them by name.
        self.db = self.settings.database_url
        self.suite_name = self.settings.suite_name
        self.judge_spec = self.settings.judge_spec
        self.embed_spec = self.settings.embed_spec
        self.seed = self.settings.seed
        self.scorer: DeployedScorer | None = None
        self.started_at = time.time()
        self.ready = False
        self._space: Space | None = None
        self._clustering: Clustering | None = None
        self._space_traces = 0

    # -- lazily-built shared objects -------------------------------------------

    def judge(self):
        from livingeval.judge import from_spec

        return from_spec(self.judge_spec)

    def traces(self, limit: int | None = None):
        return self.store.get_traces(
            limit=limit or self.settings.trace_limit,
            source=self.settings.trace_source,
        )

    def space_and_clustering(self, traces, force: bool = False):
        """Fit the space and clustering once and reuse them.

        Invalidated when the trace count changes, because a clustering computed over
        400 traces and reported against 4,000 would be a different partition wearing
        the same labels.
        """
        from livingeval import mine

        if force or self._space is None or self._space_traces != len(traces):
            self._space = mine.Space.fit(traces, seed=self.seed, embed=self.embed_spec)
            self._clustering = mine.cluster(traces, space=self._space, seed=self.seed)
            self._space_traces = len(traces)
        return self._space, self._clustering

    def suite(self):
        return self.store.get_suite(self.suite_name)

    def refit_scorer(self, **kwargs) -> dict:
        from livingeval.serve.scorer import DeployedScorer

        traces = self.traces()
        if len(traces) < 40:
            raise ValueError(f"need at least 40 traces to fit a scorer, have {len(traces)}")
        self.scorer = DeployedScorer.from_ladder(traces, self.judge(), **kwargs)
        stats = self.scorer.stats()
        self.store.put_record("scorer_deployed", stats, label=self.scorer.rung_name)
        return stats

    # -- health ----------------------------------------------------------------

    def health(self) -> dict:
        """Liveness: is this process itself alive? Deliberately touches nothing
        external — a health check that fails when the database blips causes the
        orchestrator to kill a process that was fine."""
        return {
            "status": "ok",
            "version": __version__,
            "uptime_s": round(time.time() - self.started_at, 1),
            "environment": self.settings.environment,
            "release": self.settings.release,
        }

    def readiness(self) -> tuple[bool, dict]:
        """Readiness: can this process serve? Checks the database round-trip, which is
        the dependency whose absence makes every real endpoint fail."""
        checks: dict = {}
        ok = True
        try:
            checks["database"] = {"ok": True, "ping_ms": round(self.store.ping() * 1000, 2)}
        except Exception as e:
            ok = False
            checks["database"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        checks["artifacts"] = {"ok": True, "backend": self.artifacts.backend,
                               "base": self.artifacts.base}
        checks["scorer"] = {"deployed": self.scorer is not None}
        return ok, {"status": "ready" if ok else "not ready", "checks": checks}

    def close(self) -> None:
        self.store.close()


def create_app(state: AppState | None = None, **kwargs) -> Any:
    """Build the FastAPI app. `state` is injectable so tests can pass an in-memory DB."""
    _require_fastapi()
    import contextlib

    from fastapi import Body, FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

    from livingeval.serve.middleware import Metrics, install

    st = state or AppState(**kwargs)
    metrics = Metrics()

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        # Readiness flips only after the store answers, so a rolling deployment does
        # not shift traffic onto a task whose database is unreachable.
        ok, _ = st.readiness()
        st.ready = ok
        log.info("livingeval %s starting in %s", __version__, st.settings.environment)
        yield
        log.info("shutting down")
        with contextlib.suppress(Exception):
            st.close()

    app = FastAPI(
        title="livingeval",
        version=__version__,
        description="Coverage, detection power and judge-depth for LLM agent evals.",
        lifespan=lifespan,
    )
    app.state.le = st
    app.state.metrics = metrics
    install(app, st.settings, metrics)

    static = Path(__file__).parent / "static"

    # -- health and metrics ----------------------------------------------------

    @app.get("/healthz", tags=["meta"], include_in_schema=False)
    def healthz():
        """Liveness. Never touches the database - see `AppState.health`."""
        return st.health()

    @app.get("/readyz", tags=["meta"], include_in_schema=False)
    def readyz():
        ok, payload = st.readiness()
        st.ready = ok
        return JSONResponse(status_code=200 if ok else 503, content=payload)

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics():
        if not st.settings.metrics:
            raise HTTPException(404, "metrics are disabled (LIVINGEVAL_METRICS=0)")
        extra: dict = {"traces_total": st.store.count_traces()}
        online = st.store.online_score_stats()
        if online.get("n"):
            extra["online_scores_total"] = online["n"]
            extra["online_latency_us_mean"] = online["avg_us"]
            extra["online_latency_us_p95"] = online["p95_us"]
        if st.scorer is not None and st.scorer.kappa_vs_judge is not None:
            extra["scorer_kappa_vs_judge"] = st.scorer.kappa_vs_judge
        return PlainTextResponse(metrics.render(extra), media_type="text/plain; version=0.0.4")

    # -- pages -----------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard():
        return (static / "dashboard.html").read_text(encoding="utf-8")

    @app.get("/review", response_class=HTMLResponse, include_in_schema=False)
    def review_page():
        return (static / "review.html").read_text(encoding="utf-8")

    # -- the hot path ----------------------------------------------------------

    @app.post("/v1/score", tags=["online scoring"])
    def score(payload: dict = Body(...)):
        """Score one turn or trace.

        Accepts `{"text": "..."}` or `{"turns": [{"role": ..., "content": ...}]}`.
        Answers from the deployed rung in memory: microseconds, no tokens, no network.
        """
        if st.scorer is None:
            raise HTTPException(
                409,
                "no scorer deployed. POST /v1/scorer/refit first, or start the server "
                "with --demo which does it for you.",
            )
        if "text" in payload:
            result = st.scorer.score_text(str(payload["text"]))
        elif "turns" in payload:
            from livingeval.trace.types import Trace, TraceSet

            trace = Trace(
                trace_id=str(payload.get("trace_id") or f"live-{int(time.time() * 1e6)}"),
                turns=payload["turns"],
                ts=float(payload.get("ts") or time.time()),
                session_id=payload.get("session_id"),
            )
            result = st.scorer.score(trace)
            if payload.get("store", True):
                st.store.put_traces(TraceSet([trace]), source="live")
        else:
            raise HTTPException(422, "send either {'text': ...} or {'turns': [...]}")

        st.store.put_online_score(result.scorer, result.label, result.latency_us, result.trace_id)
        return result.as_dict()

    @app.post("/v1/scorer/refit", tags=["online scoring"])
    def refit(payload: dict = Body(default={})):
        """Fit the rung the ladder recommends on everything in the store."""
        try:
            return st.refit_scorer(**payload)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/v1/scorer", tags=["online scoring"])
    def scorer_stats():
        if st.scorer is None:
            return {"deployed": False}
        return {"deployed": True, **st.scorer.stats(), **st.store.online_score_stats()}

    # -- ingestion -------------------------------------------------------------

    @app.post("/v1/traces", tags=["traces"])
    def put_traces(payload: dict = Body(...)):
        """Ingest traces: `{"traces": [ ... ]}` in native or loose shape."""
        from livingeval.trace.ingest import from_records

        records = payload.get("traces") or payload.get("records") or []
        if not records:
            raise HTTPException(422, "send {'traces': [...]}")
        traces = from_records(records)
        n = st.store.put_traces(traces, source=payload.get("source", "api"))
        st._space = None  # the partition changed
        return {"ingested": n, "total": st.store.count_traces()}

    @app.get("/v1/traces", tags=["traces"])
    def get_traces(limit: int = Query(20, le=500)):
        traces = st.store.get_traces(limit=limit)
        return {"n": len(traces), "traces": traces.as_dicts()}

    # -- the three measurements ------------------------------------------------

    @app.get("/v1/coverage", tags=["measurements"])
    def coverage(refresh: bool = False):
        from livingeval import mine

        suite, traces = st.suite(), st.traces()
        if suite is None:
            raise HTTPException(404, f"no suite named {st.suite_name!r}")
        space, clustering = st.space_and_clustering(traces, force=refresh)
        result = mine.coverage(suite, traces, space=space, clustering=clustering, seed=st.seed)
        st.store.put_record("coverage", result.as_dict(), label=suite.name)
        return result.as_dict()

    @app.get("/v1/blindspots", tags=["measurements"])
    def blindspots(top: int = 8, refresh: bool = False):
        from livingeval import mine

        suite, traces = st.suite(), st.traces()
        if suite is None:
            raise HTTPException(404, f"no suite named {st.suite_name!r}")
        space, clustering = st.space_and_clustering(traces, force=refresh)
        report = mine.blindspots(suite, traces, top=top, space=space,
                                 clustering=clustering, seed=st.seed)
        payload = report.as_dict()
        payload["coverage"] = report.coverage.as_dict()
        st.store.put_record("blindspots", payload, label=suite.name)
        return payload

    @app.get("/v1/ladder", tags=["measurements"])
    def ladder_endpoint(n_boot: int = 400, finetune: bool = False):
        from livingeval import scorer as scorer_mod

        traces = st.traces()
        result = scorer_mod.ladder(traces, st.judge(), n_boot=n_boot,
                                   seed=st.seed, finetune=finetune)
        payload = result.as_dict()  # type: ignore[union-attr]
        st.store.put_record("ladder", payload, label=str(payload.get("judge_depth")))
        return payload

    @app.get("/v1/power", tags=["measurements"])
    def power_endpoint(effect: float = 0.4, n_sim: int = 200, top: int = 6):
        from livingeval import power as power_mod
        from livingeval.gate import run_suite

        suite, traces = st.suite(), st.traces()
        if suite is None:
            raise HTTPException(404, f"no suite named {st.suite_name!r}")
        _, clustering = st.space_and_clustering(traces)
        baseline = run_suite(suite, st.judge(), "baseline")
        expected = power_mod.expected_power(suite, baseline, clustering, effect=effect,
                                           top=top, n_sim=n_sim, seed=st.seed)
        false_alarm = power_mod.false_alarm_rate(suite, baseline, n_sim=n_sim, seed=st.seed)
        payload = {**expected.as_dict(), "false_alarm": false_alarm.power.as_dict()}
        st.store.put_record("power", payload, label=suite.name)
        return payload

    @app.get("/v1/judge/validate", tags=["measurements"])
    def validate_judge(n_boot: int = 400):
        from livingeval import judge as judge_mod

        traces = st.traces()
        result = judge_mod.validate(st.judge(), traces.labelled(), n_boot=n_boot, seed=st.seed)
        st.store.put_record("judge_validation", result.as_dict(), label=result.status)
        return result.as_dict()

    @app.get("/v1/gate", tags=["measurements"])
    def gate_endpoint(mode: str = "paired", min_coverage: float = 0.70):
        from livingeval import gate as gate_mod
        from livingeval import mine

        suite, traces = st.suite(), st.traces()
        if suite is None:
            raise HTTPException(404, f"no suite named {st.suite_name!r}")
        space, clustering = st.space_and_clustering(traces)
        cov = mine.coverage(suite, traces, space=space, clustering=clustering, seed=st.seed)
        current = gate_mod.run_suite(suite, st.judge(), "current")
        result = gate_mod.evaluate(current, current, mode=mode, coverage=cov.coverage,
                                   min_coverage=min_coverage)
        payload = result.as_dict()
        st.store.put_record("gate", payload, label=result.decision)
        return payload

    # -- the review queue ------------------------------------------------------

    @app.post("/v1/mine", tags=["review"])
    def mine_proposals(n: int = 20):
        """Run the mining loop and queue proposals for review."""
        from livingeval import mine as mine_mod

        suite, traces = st.suite(), st.traces()
        if suite is None:
            raise HTTPException(404, f"no suite named {st.suite_name!r}")
        space, clustering = st.space_and_clustering(traces)
        report = mine_mod.blindspots(suite, traces, space=space,
                                     clustering=clustering, seed=st.seed)
        proposals = mine_mod.propose(traces, suite, n=n, judge=st.judge(),
                                     report=report, space=space, seed=st.seed)
        rows = [
            {
                "trace_id": p.trace.trace_id, "cluster": p.cluster, "reason": p.reason,
                "suggested_expected": p.suggested_expected, "judge_rationale": p.judge_rationale,
            }
            for p in proposals
        ]
        queued = st.store.put_proposals(suite.name, rows)
        return {"proposed": len(rows), "queued": queued,
                "pending": len(st.store.get_proposals(suite.name, "pending"))}

    @app.get("/v1/proposals", tags=["review"])
    def get_proposals(status: str = "pending", limit: int = 50):
        rows = st.store.get_proposals(st.suite_name, status=status)[:limit]
        return {"n": len(rows), "proposals": rows}

    @app.post("/v1/proposals/{proposal_id}/confirm", tags=["review"])
    def confirm(proposal_id: int, payload: dict = Body(...)):
        """Confirm a proposal. `reviewer` is required — that is the guard-rail."""
        reviewer = str(payload.get("reviewer") or "").strip()
        if not reviewer:
            raise HTTPException(422, "a review must record who did it: send {'reviewer': ...}")
        if "expected" not in payload:
            raise HTTPException(422, "send {'expected': 0 or 1}")
        try:
            return st.store.confirm_proposal(proposal_id, reviewer, int(payload["expected"]))
        except KeyError as e:
            raise HTTPException(404, str(e)) from e

    @app.post("/v1/proposals/{proposal_id}/reject", tags=["review"])
    def reject(proposal_id: int, payload: dict = Body(...)):
        reviewer = str(payload.get("reviewer") or "").strip()
        if not reviewer:
            raise HTTPException(422, "a review must record who did it: send {'reviewer': ...}")
        try:
            return st.store.reject_proposal(proposal_id, reviewer)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e

    @app.post("/v1/promote", tags=["review"])
    def promote(payload: dict = Body(default={})):
        """Merge confirmed proposals into the suite."""
        from livingeval import feedback

        try:
            result = feedback.promote(st.store, st.suite_name,
                                      max_cases=payload.get("max_cases"))
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        st.store.put_record("promotion", result.as_dict(), label=st.suite_name)
        return result.as_dict()

    @app.post("/v1/export/finetune", tags=["review"])
    def export_finetune(payload: dict = Body(default={})):
        """Write human-confirmed labels as fine-tuning data."""
        from livingeval import feedback

        suite = st.suite()
        if suite is None:
            raise HTTPException(404, f"no suite named {st.suite_name!r}")
        return feedback.export_finetune_data(
            suite,
            payload.get("path", f"data/{st.suite_name}-finetune.jsonl"),
            fmt=payload.get("format", "chat"),
            split=payload.get("split"),
        )

    # -- meta ------------------------------------------------------------------

    @app.get("/v1/status", tags=["meta"])
    def status():
        lo, hi = st.store.trace_time_span()
        # `as_dict()` redacts: a status endpoint that echoes a database password is a
        # credential leak wearing a diagnostics label.
        return {
            "version": __version__,
            "uptime_s": round(time.time() - st.started_at, 1),
            "config": st.settings.as_dict(),
            "db": st.settings.as_dict()["database"],
            "suite": st.suite_name,
            "judge": st.judge_spec,
            "embedder": st.embed_spec or "tfidf+svd (default)",
            "artifacts": {"backend": st.artifacts.backend, "base": st.artifacts.base},
            "migrations": st.store.applied_migrations(),
            "store": dict(st.store.stats()),
            "trace_span": {"from": lo, "to": hi},
            "scorer": st.scorer.stats() if st.scorer else None,
            "online": st.store.online_score_stats(),
            "suites": st.store.list_suites(),
        }

    @app.get("/v1/artifacts", tags=["meta"])
    def list_artifacts(prefix: str = ""):
        return {"backend": st.artifacts.backend, "base": st.artifacts.base,
                "keys": st.artifacts.list(prefix)}

    @app.get("/v1/records", tags=["meta"])
    def records(kind: str | None = None, limit: int = 20):
        return {"records": st.store.get_records(kind=kind, limit=limit)}

    @app.exception_handler(ValueError)
    def value_error_handler(_request, exc: ValueError):  # pragma: no cover - defensive
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app
