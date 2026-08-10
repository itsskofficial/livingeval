"""The local platform: a REST API, a review queue and a dashboard.

    livingeval serve --demo          # seeds a drifting corpus and opens on :8000

Two things a library cannot do on its own:

1. **Per-turn online scoring as a service.** `POST /v1/score` answers from a fitted
   rung held in memory - microseconds, no tokens, no network. That endpoint is what the
   whole judge-complexity ladder exists to justify.
2. **Review a human will actually do.** Approving mined proposals from a REPL is a
   chore nobody repeats; `/review` is two buttons and a keyboard shortcut.

Local by design: binds `127.0.0.1`, no authentication, one SQLite file. It is a control
surface for your own machine. Making it multi-tenant is the point at which this becomes
a platform competing with four funded companies, which is not the pitch.
"""

from livingeval.serve.scorer import DeployedScorer, ScoreResult

__all__ = ["AppState", "DeployedScorer", "ScoreResult", "create_app", "run", "seed_demo"]


def __getattr__(name: str):
    """`create_app` and `AppState` are lazy so fastapi stays an optional extra."""
    if name in ("create_app", "AppState"):
        from livingeval.serve import app as _app

        return getattr(_app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def seed_demo(store, n: int = 2400, suite_size: int = 200, seed: int = 0,
              suite_name: str = "default") -> dict:
    """Fill a store with a drifting corpus and a suite curated from its first window.

    This is the demo the dashboard is built around, and it is the honest one: the suite
    is genuinely curated from traffic six windows old, so coverage genuinely falls and
    the blind spot genuinely is the intent that appeared later. Nothing is staged.
    """
    from livingeval import synthetic
    from livingeval.suite import EvalSuite

    traces = synthetic.drifting(n=n, seed=seed)
    store.put_traces(traces, source="synthetic")
    windows = traces.sorted_by_time().windows(6)
    suite = EvalSuite.from_traces(
        windows[0].sample(suite_size, seed=seed), name=suite_name
    )
    store.put_suite(suite)
    return {
        "traces": len(traces),
        "suite": suite.name,
        "suite_cases": len(suite),
        "windows": 6,
        "note": "suite curated from window 0; traffic runs to window 5",
    }


def run(host: str | None = None, port: int | None = None, db: str | None = None,
        suite: str | None = None, judge: str | None = None, embed: str | None = None,
        demo: bool = False, refit: bool = True, reload: bool = False,
        workers: int = 1) -> None:
    """Start the server. Called by `livingeval serve`.

    Every argument defaults to its environment variable, so the same command works
    unchanged in a container where configuration arrives as env vars.
    """
    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover - optional extra
        raise ImportError("pip install 'livingeval[serve]'") from e

    from livingeval.config import Settings
    from livingeval.serve.app import AppState, create_app
    from livingeval.serve.middleware import configure_logging

    settings = Settings.load(host=host, port=port, database_url=db, suite_name=suite,
                             judge_spec=judge, embed_spec=embed)
    configure_logging(settings.log_level, settings.log_json)
    host, port = settings.host, settings.port

    state = AppState(settings=settings)
    if demo and state.store.count_traces() == 0:
        info = seed_demo(state.store, suite_name=settings.suite_name)
        print(f"  seeded {info['traces']} traces and a {info['suite_cases']}-case suite "
              f"({info['note']})")
    if refit and state.store.count_traces() >= 40:
        try:
            stats = state.refit_scorer()
            print(f"  deployed scorer: {stats['scorer']} "
                  f"(kappa {stats['kappa_vs_judge']:.3f}, {stats['mean_latency_us'] or 0:.0f} us/turn)")
        except (ValueError, TypeError) as e:
            print(f"  no scorer deployed: {e}")

    print(f"\n  livingeval  ->  http://{host}:{port}")
    print(f"  review queue ->  http://{host}:{port}/review")
    print(f"  api docs     ->  http://{host}:{port}/docs\n")
    uvicorn.run(create_app(state), host=host, port=port, log_level="warning", reload=reload)
