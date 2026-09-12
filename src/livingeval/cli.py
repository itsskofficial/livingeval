"""The `livingeval` command.

argparse, no framework dependency: `pip install livingeval` should give a working
command with nothing else pulled in.

`livingeval gate` is the one designed to be wired into CI, and it exits 0 on PASS,
1 on FAIL and **2 on BLIND**, so a pipeline can treat "the suite has decayed" as its
own branch rather than folding it into green.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from livingeval._version import __version__

__all__ = ["build_parser", "main"]


def _settings(**overrides):
    """Resolved settings, with a readable message when the environment is wrong."""
    from livingeval.config import Settings

    try:
        return Settings.load(**overrides)
    except ValueError as e:
        raise SystemExit(f"configuration error:\n{e}") from e


def _redact(value: str) -> str:
    from livingeval.config import redact

    return redact(value, "url")


def _load_judge(spec: str):
    """Resolve a `--judge` string. The parser lives in `judge.spec` so that the CLI and
    the platform share it without the web layer importing the CLI."""
    from livingeval.judge import from_spec

    try:
        return from_spec(spec)
    except ValueError as e:
        raise SystemExit(str(e)) from e


def _load_traces(spec: str):
    """Resolve a `--traces` string. Shared parser in `livingeval.sources`."""
    from livingeval.sources import load_traces

    try:
        return load_traces(spec)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(str(e)) from e


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_init(args) -> int:
    """Scan, plan, explain, and write a suite. The front door."""
    from livingeval.wizard import run_init
    return run_init(Path(args.path), assume_yes=args.yes, dry_run=args.dry_run,
                    include_tests=args.include_tests)


def cmd_scan(args) -> int:
    """Report what is there without writing anything."""
    from livingeval.discover import scan
    from livingeval.plan import build_plan

    sites = scan(Path(args.path), include_tests=args.include_tests)
    if not sites:
        print("no LLM call sites found")
        return 1
    plan = build_plan(sites)
    print(plan.summary())
    print()
    for site in sites:
        print(f"{site.path}:{site.line}  {site.function}()")
        print(f"    {site.provider} -> {site.archetype} ({site.confidence:.0%} confident)")
        if args.explain:
            for reason in site.rationale:
                print(f"      - {reason}")
    if args.explain:
        print()
        for pipeline in plan.pipelines:
            print(f"{pipeline.slug}  [{pipeline.level.value}/{pipeline.risk.value}]")
            for metric in pipeline.metrics:
                flag = " " if metric.automatable else "*"
                print(f"   {flag} {metric.name:<24} {metric.catches}")
        print()
        print("   * needs a reference answer you write")
    return 0


def cmd_coverage(args) -> int:
    from livingeval import mine
    from livingeval.suite import EvalSuite

    traces = _load_traces(args.traces)
    suite = EvalSuite.load(args.suite)
    space = mine.Space.fit(traces, seed=args.seed)
    clustering = mine.cluster(traces, space=space, seed=args.seed)
    result = mine.coverage(suite, traces, space=space, clustering=clustering, q=args.q)
    print(result.summary())
    if args.out:
        from livingeval import report

        report.save([result, clustering.as_dict() | {"kind": "clustering"}], args.out)
        print(f"\nwrote {args.out}")
    return 0


def cmd_blindspots(args) -> int:
    from livingeval import mine
    from livingeval.suite import EvalSuite

    traces = _load_traces(args.traces)
    suite = EvalSuite.load(args.suite)
    result = mine.blindspots(suite, traces, top=args.top, seed=args.seed)
    print(result.summary())
    if args.out:
        from livingeval import report

        report.save([result, result.coverage], args.out)
        print(f"\nwrote {args.out}")
    return 0


def cmd_ladder(args) -> int:
    from livingeval import scorer

    traces = _load_traces(args.traces)
    judge = _load_judge(args.judge)
    result = scorer.ladder(
        traces, judge, threshold=args.threshold, cv=args.cv, view=args.view, seed=args.seed
    )
    assert isinstance(result, scorer.LadderResult)  # `by=` is not exposed on the CLI
    print(result.summary())
    if args.out:
        from livingeval import report

        report.save([result], args.out)
        print(f"\nwrote {args.out}")
    return 0


def cmd_validate(args) -> int:
    from livingeval import judge as J

    traces = _load_traces(args.traces)
    judge = _load_judge(args.judge)
    result = J.validate(judge, traces.labelled() if args.labelled_only else traces)
    print(result.summary())
    if args.out:
        from livingeval import report

        report.save([result], args.out)
    return 0


def cmd_power(args) -> int:
    from livingeval import mine, power
    from livingeval.gate import run_suite
    from livingeval.suite import EvalSuite

    traces = _load_traces(args.traces)
    suite = EvalSuite.load(args.suite)
    judge = _load_judge(args.judge)
    baseline = run_suite(suite, judge, "baseline")
    clustering = mine.cluster(traces, seed=args.seed)
    result = power.expected_power(
        suite, baseline, clustering, effect=args.effect, n_sim=args.n_sim, seed=args.seed
    )
    print(result.summary())
    fa = power.false_alarm_rate(suite, baseline, n_sim=args.n_sim, seed=args.seed)
    print(f"\n  false-alarm rate of this gate: {fa.power}")
    if args.out:
        from livingeval import report

        report.save([result, fa], args.out)
    return 0


def cmd_gate(args) -> int:
    from livingeval import gate as G
    from livingeval.suite import EvalSuite

    suite = EvalSuite.load(args.suite)
    judge = _load_judge(args.judge)
    current = G.run_suite(suite, judge, "current")

    baseline = None
    if args.baseline:
        from livingeval import report

        record = report.load(args.baseline)
        stored = record.first("suite_run")
        if stored is None:
            raise SystemExit(f"{args.baseline} contains no suite_run record")
        import numpy as np

        baseline = G.SuiteRun(
            stored["suite"], stored["suite_hash"], stored["judge"], stored["case_ids"],
            np.asarray(stored["outcomes"], dtype=int), current.expected, current.observed,
            current.groups, current.gateable, "baseline", 0.0,
        )

    coverage = None
    if args.traces:
        from livingeval import mine

        traces = _load_traces(args.traces)
        coverage = mine.coverage(suite, traces, seed=args.seed).coverage

    result = G.evaluate(
        current, baseline, mode=args.mode, threshold=args.threshold, coverage=coverage,
        min_coverage=args.min_coverage, min_cases=args.min_cases,
    )
    print(result.summary())
    if args.out:
        from livingeval import report

        report.save([result, current], args.out)
        print(f"\nwrote {args.out}")
    return result.exit_code


def cmd_audit(args) -> int:
    """Run the bundled audit. Lives in `scripts/` so the engine stays scenario-free."""
    root = Path(__file__).resolve().parents[3]
    script = root / "scripts" / "run_audit.py"
    if not script.exists():
        raise SystemExit(
            "the audit script ships with the repository, not the wheel; clone "
            "https://github.com/itsskofficial/livingeval and run scripts/run_audit.py"
        )
    import runpy

    sys.argv = [str(script), *([] if args.quick is False else ["--quick"])]
    runpy.run_path(str(script), run_name="__main__")
    return 0


def cmd_figures(args) -> int:
    from livingeval import report

    record = report.load(args.record)
    made = report.figures(record, args.out)
    for p in made:
        print(f"wrote {p}")
    if not made:
        print("nothing in that record can be plotted")
    return 0


def cmd_synth(args) -> int:
    traces = _load_traces(f"synthetic:{args.generator},n={args.n},seed={args.seed}")
    traces.to_jsonl(args.out)
    print(f"wrote {len(traces)} traces to {args.out}")
    return 0



def cmd_serve(args) -> int:
    from livingeval import serve

    serve.run(host=args.host, port=args.port, db=args.db, suite=args.suite,
              judge=args.judge, embed=args.embed, demo=args.demo,
              refit=not args.no_refit, workers=args.workers)
    return 0


def cmd_ingest(args) -> int:
    """Load traces into a store, once or by following a live Langfuse project."""
    from livingeval.store import open_store

    store = open_store(args.db)
    if args.follow:
        from livingeval.trace.ingest.langfuse import poll

        print(f"polling Langfuse every {args.interval}s into {args.db}  (ctrl-c to stop)")
        total = poll(store, interval_seconds=args.interval, limit=args.limit)
        print(f"wrote {total} traces")
        return 0
    if not args.traces:
        raise SystemExit("pass --traces, or --follow to poll Langfuse")
    traces = _load_traces(args.traces)
    n = store.put_traces(traces, source=args.source)
    print(f"ingested {n} traces; store now holds {store.count_traces()}")
    if args.suite_from_first_window:
        from livingeval.suite import EvalSuite

        windows = traces.sorted_by_time().windows(args.windows)
        suite = EvalSuite.from_traces(
            windows[0].sample(args.suite_size, seed=args.seed), name=args.suite
        )
        store.put_suite(suite)
        print(f"curated suite {suite.name!r} with {len(suite)} case(s) from window 0 of "
              f"{args.windows} - i.e. a realistically stale one")
    return 0


def cmd_promote(args) -> int:
    from livingeval import feedback
    from livingeval.store import open_store

    result = feedback.promote(open_store(args.db), args.suite, max_cases=args.max_cases)
    print(result.summary())
    return 0


def cmd_export(args) -> int:
    from livingeval import feedback
    from livingeval.store import open_store

    store = open_store(args.db)
    suite = store.get_suite(args.suite)
    if suite is None:
        raise SystemExit(f"no suite named {args.suite!r} in {args.db}")
    info = feedback.export_finetune_data(
        suite, args.out, fmt=args.format, split=args.split,
        confirmed_only=not args.include_unconfirmed,
    )
    for k, v in info.items():
        print(f"  {k}: {v}")
    return 0


def cmd_finetune(args) -> int:
    """Run the ladder with the fine-tuned rung appended."""
    from livingeval import scorer

    traces = _load_traces(args.traces)
    judge = _load_judge(args.judge)
    result = scorer.ladder(
        traces, judge, seed=args.seed, n_boot=args.n_boot,
        finetune={"backend": args.backend, **({"model": args.model} if args.model else {})},
    )
    assert isinstance(result, scorer.LadderResult)
    print(result.summary())
    if args.out:
        from livingeval import report

        report.save([result], args.out)
        print(f"wrote {args.out}")
    return 0


def cmd_compare_spaces(args) -> int:
    """Does the representation change the conclusions? Measure rather than assume."""
    from livingeval import embed, mine
    from livingeval.suite import EvalSuite

    traces = _load_traces(args.traces)
    if args.suite:
        suite = EvalSuite.load(args.suite)
    else:
        windows = traces.sorted_by_time().windows(6)
        suite = EvalSuite.from_traces(windows[0].sample(args.suite_size, seed=args.seed),
                                      name="window0")
        print(f"no --suite given; curated {len(suite)} cases from window 0 of 6")

    rows = []
    for spec in ["tfidf+svd", *args.embed]:
        space = mine.Space.fit(
            traces, seed=args.seed, embed=None if spec == "tfidf+svd" else embed.resolve(spec)
        )
        clustering = mine.cluster(traces, space=space, seed=args.seed)
        rep = mine.blindspots(suite, traces, space=space, clustering=clustering, seed=args.seed)
        worst = rep.spots[0]
        rows.append({
            "space": space.name,
            "coverage": rep.coverage.coverage,
            "k": clustering.k,
            "silhouette": clustering.silhouette,
            "worst_cluster_share": worst.traffic_share,
            "worst_cluster_coverage": worst.coverage,
            "worst_terms": worst.terms,
        })
        print(f"{space.name:<46} coverage {rep.coverage.coverage:.3f}  k={clustering.k:<3} "
              f"worst: {worst.traffic_share:5.1%} traffic at {worst.coverage:5.1%} covered")

    covs = [float(r["coverage"]) for r in rows if r["coverage"] is not None]
    spread = (max(covs) - min(covs)) if covs else float("nan")
    print(f"\n  coverage spread across representations: {spread:.3f}")
    print("  A small spread means the cheap default is not costing you a conclusion.")
    print("  A large one means every cross-project comparison needs the representation pinned.")
    if args.out:
        from livingeval import report as report_mod

        report_mod.save([{"kind": "space_comparison", "rows": rows, "spread": spread}], args.out)
        print(f"wrote {args.out}")
    return 0



def cmd_db(args) -> int:
    """Inspect, migrate or dump the schema."""
    from livingeval.store import open_store
    from livingeval.store.migrations import MIGRATIONS, postgres_sql, sqlite_sql

    if args.db_command == "sql":
        sql = sqlite_sql() if args.dialect == "sqlite" else postgres_sql()
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(sql, encoding="utf-8")
            print(f"wrote {args.out}")
        else:
            print(sql)
        return 0

    url = args.url or _settings().database_url

    if args.db_command == "check":
        print(f"database  {_redact(url)}")
        if url.startswith(("postgres://", "postgresql://")):
            from livingeval.store import dsn as dsn_mod

            print(dsn_mod.parse(url).describe())
        try:
            store = open_store(url)
        except Exception as e:
            print(f"\n  UNREACHABLE: {type(e).__name__}: {e}")
            return 1
        applied = store.applied_migrations()
        pending = [m.id for m in MIGRATIONS if m.id not in set(applied)]
        print(f"  ping      {store.ping() * 1000:.1f} ms")
        print(f"  applied   {applied or 'none'}")
        print(f"  pending   {pending or 'none'}")
        print(f"  contents  {store.stats().summary()}")
        store.close()
        return 0 if not pending else 2

    if args.db_command == "migrate":
        store = open_store(url)
        applied = store.migrate()
        print(f"applied {len(applied)} migration(s): {applied or 'nothing to do'}")
        print(f"  now at: {store.applied_migrations()}")
        store.close()
        return 0

    raise SystemExit(f"unknown db command {args.db_command!r}")


def cmd_worker(args) -> int:
    """Run the measurement loop on an interval, or once."""
    from livingeval.serve.app import AppState
    from livingeval.serve.middleware import configure_logging
    from livingeval.serve.worker import run_forever, run_once

    settings = _settings(database_url=args.db, suite_name=args.suite, judge_spec=args.judge,
                         embed_spec=args.embed)
    configure_logging(settings.log_level, settings.log_json)
    state = AppState(settings=settings)
    try:
        if args.once:
            result = run_once(state, tick=1, mine_n=args.mine, ingest_spec=args.ingest)
            print(result.summary())
            for err in result.errors:
                print(f"  ! {err}")
            return 1 if result.errors else 0
        run_forever(state, interval_s=args.interval, mine_n=args.mine,
                    ingest_spec=args.ingest, max_ticks=args.max_ticks)
        return 0
    finally:
        state.close()


def cmd_artifacts(args) -> int:
    """List or fetch what the deployment has written."""
    from livingeval.artifacts import open_artifacts

    store = open_artifacts(args.url or _settings().artifact_url)
    if args.get:
        data = store.get_bytes(args.get)
        if args.out:
            Path(args.out).write_bytes(data)
            print(f"wrote {args.out} ({len(data)} bytes)")
        else:
            sys.stdout.write(data.decode("utf-8", "replace"))
        return 0
    keys = store.list(args.prefix)
    print(f"{store.backend}  {store.base}   {len(keys)} object(s)")
    for k in keys[: args.limit]:
        print(f"  {k}")
    if len(keys) > args.limit:
        print(f"  ... and {len(keys) - args.limit} more")
    return 0


def cmd_config(args) -> int:
    """Show the resolved configuration, with secrets redacted."""
    settings = _settings()
    print("livingeval configuration")
    print(settings.summary())
    if args.check:
        problems = []
        if settings.uses_postgres and settings.uses_pooler:
            problems.append(
                "database URL is a transaction pooler (6543): correct for the service, "
                "but run `livingeval db migrate --url <direct 5432 URL>` for migrations"
            )
        if not settings.is_local_only and not settings.api_key:
            problems.append("binding non-locally with no LIVINGEVAL_API_KEY")
        if settings.artifact_url.startswith("file://") and settings.environment != "local":
            problems.append(
                f"artifacts go to a local path but LIVINGEVAL_ENV={settings.environment}; "
                "a container filesystem does not survive a redeploy - use s3:// or supabase://"
            )
        print("\nchecks")
        for p in problems:
            print(f"  ! {p}")
        if not problems:
            print("  all good")
        return 1 if problems else 0
    return 0



def cmd_supabase(args) -> int:
    """Point livingeval at Supabase — the local stack or a cloud project."""

    from livingeval.store import supabase as sb

    if args.supabase_command == "url":
        if not (args.project_ref and args.password):
            raise SystemExit(
                "need --project-ref and --password.\n"
                "  Supabase dashboard -> Project Settings -> Database -> Connection string"
            )
        url = sb.connection_url(args.project_ref, args.password,
                                purpose=args.purpose, region=args.region)
        print(url)
        return 0

    config = sb.detect(require=True)
    assert config is not None  # require=True raises rather than returning None

    if args.supabase_command == "status":
        print(f"supabase [{'local' if config.is_local else 'cloud'}]")
        for k, v in config.as_dict().items():
            print(f"  {k:<18} {v}")
        try:
            from livingeval.store import open_store

            store = open_store(config.db_url)
            print(f"  {'database':<18} reachable, {store.ping() * 1000:.1f} ms")
            print(f"  {'migrations':<18} {store.applied_migrations()}")
            print(f"  {'contents':<18} {store.stats().summary()}")
            store.close()
        except Exception as e:
            print(f"  {'database':<18} UNREACHABLE: {type(e).__name__}: {e}")
            return 1
        return 0

    if args.supabase_command == "env":
        for k, v in config.env().items():
            print(f"{'set ' if args.shell == 'cmd' else 'export '}{k}={v}"
                  if args.shell != "dotenv" else f"{k}={v}")
        if args.bucket:
            print(("export " if args.shell != "dotenv" else "")
                  + f"LIVINGEVAL_ARTIFACT_URL=supabase://{args.bucket}/livingeval")
        return 0

    if args.supabase_command == "init":
        from livingeval.artifacts import SupabaseArtifacts
        from livingeval.store import open_store

        print(f"supabase [{'local' if config.is_local else 'cloud'}]  {config.api_url}")

        store = open_store(config.db_url)
        applied = store.migrate()
        print(f"  migrations   {applied or 'already up to date'} "
              f"-> {store.applied_migrations()}")

        # RLS is applied by migration 0002; verify rather than assume, because a table
        # in `public` without it is reachable with the anon key that ships in a browser.
        try:
            unprotected = _unprotected_tables(store)
            if unprotected:
                print(f"  ! RLS OFF on: {unprotected} - the Data API can read these")
            else:
                print("  rls          on for every livingeval table")
        except Exception as e:
            print(f"  rls          could not verify ({type(e).__name__})")
        store.close()

        if config.service_role_key:
            artifacts = SupabaseArtifacts(args.bucket, "livingeval",
                                          url=config.api_url,
                                          service_key=config.service_role_key)
            if not artifacts.wait_ready():
                print("  bucket       Storage did not come up in 60s; skipping")
                return 0
            result = artifacts.create_bucket(public=False)
            print(f"  bucket       {result['bucket']} "
                  f"({'created' if result['created'] else 'already exists'}, private)")
            probe = artifacts.put("_healthcheck.json", {"ok": True})
            print(f"  storage      write+read ok -> {probe.uri}")
        else:
            print("  bucket       skipped: no SUPABASE_SERVICE_ROLE_KEY")

        print("\n  export these to use it:")
        for k, v in config.env().items():
            print(f"    export {k}={v}")
        print(f"    export LIVINGEVAL_ARTIFACT_URL=supabase://{args.bucket}/livingeval")
        return 0

    raise SystemExit(f"unknown supabase command {args.supabase_command!r}")


def _unprotected_tables(store) -> list[str]:
    """livingeval tables in `public` without row-level security."""
    names = ("traces", "verdicts", "suites", "proposals", "records", "online_scores")
    with store._cur(dict_rows=False) as cur:
        cur.execute(
            "SELECT tablename FROM pg_tables "
            "WHERE schemaname = 'public' AND rowsecurity = false AND tablename = ANY(%s)",
            (list(names),),
        )
        return [r[0] for r in cur.fetchall()]


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="livingeval",
        description="Eval suites that tell you when they've gone blind.",
    )
    p.add_argument("--version", action="version", version=f"livingeval {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, traces=True, suite=False, judge=False):
        if traces:
            sp.add_argument("--traces", required=True,
                            help="a .jsonl glob, an OTel/Langfuse .json, or synthetic:drifting,n=1200")
        if suite:
            sp.add_argument("--suite", required=True, help="path to a suite JSON")
        if judge:
            sp.add_argument("--judge", default="oracle",
                            help="rule:mod:fn | oracle[:noise] | openai:model | anthropic:model")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--out", default=None, help="write a result record here")

    sp = sub.add_parser("init", help="scan a codebase and write an eval suite")
    sp.add_argument("--path", default=".", help="repository root (default: .)")
    sp.add_argument("--yes", action="store_true",
                    help="take every default; report what was assumed")
    sp.add_argument("--dry-run", action="store_true", help="plan and explain, write nothing")
    sp.add_argument("--include-tests", action="store_true",
                    help="scan test files too (usually mocks, so off by default)")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("scan", help="list the LLM call sites and what they need")
    sp.add_argument("--path", default=".")
    sp.add_argument("--explain", action="store_true", help="show the reasoning")
    sp.add_argument("--include-tests", action="store_true")
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser("coverage", help="how much of your traffic does this suite represent")
    common(sp, suite=True)
    sp.add_argument("--q", type=float, default=50.0, help="radius percentile (default 50)")
    sp.set_defaults(func=cmd_coverage)

    sp = sub.add_parser("blindspots", help="rank the traffic your suite cannot see")
    common(sp, suite=True)
    sp.add_argument("--top", type=int, default=5)
    sp.set_defaults(func=cmd_blindspots)

    sp = sub.add_parser("ladder", help="what is the cheapest model that reproduces your judge")
    common(sp, judge=True)
    sp.add_argument("--threshold", type=float, default=0.80, help="kappa bar (default 0.80)")
    sp.add_argument("--cv", default="session", choices=["session", "random"])
    sp.add_argument("--view", default="full", choices=["full", "request", "response", "last_exchange"])
    sp.set_defaults(func=cmd_ladder)

    sp = sub.add_parser("validate", help="measure a judge against human labels")
    common(sp, judge=True)
    sp.add_argument("--labelled-only", action="store_true", default=True)
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("power", help="would this suite catch a regression")
    common(sp, suite=True, judge=True)
    sp.add_argument("--effect", type=float, default=0.40)
    sp.add_argument("--n-sim", type=int, default=300)
    sp.set_defaults(func=cmd_power)

    sp = sub.add_parser("gate", help="PASS (0), FAIL (1) or BLIND (2)")
    sp.add_argument("--suite", required=True)
    sp.add_argument("--judge", default="oracle")
    sp.add_argument("--baseline", default=None, help="a record written by a previous gate run")
    sp.add_argument("--traces", default=None, help="production traces, to compute coverage")
    sp.add_argument("--mode", default="paired", choices=["paired", "threshold"])
    sp.add_argument("--threshold", type=float, default=0.90)
    sp.add_argument("--min-coverage", type=float, default=0.70)
    sp.add_argument("--min-cases", type=int, default=20)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_gate)

    sp = sub.add_parser("audit", help="reproduce the three headline findings")
    sp.add_argument("--quick", action="store_true", help="fewer simulations, same shape")
    sp.set_defaults(func=cmd_audit)

    sp = sub.add_parser("figures", help="draw figures from a saved record")
    sp.add_argument("record")
    sp.add_argument("--out", default="figs/")
    sp.set_defaults(func=cmd_figures)

    sp = sub.add_parser("synth", help="write synthetic traces to JSONL")
    sp.add_argument("generator", choices=["shortcut", "lexical", "morphology", "deep", "drifting", "stable"])
    sp.add_argument("--n", type=int, default=1200)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", default="traces.jsonl")
    sp.set_defaults(func=cmd_synth)

    sp = sub.add_parser("serve", help="the local platform: dashboard, review queue, /v1/score")
    sp.add_argument("--host", default=None, help="defaults to LIVINGEVAL_HOST")
    sp.add_argument("--port", type=int, default=None, help="defaults to LIVINGEVAL_PORT")
    sp.add_argument("--db", default=None, help="defaults to LIVINGEVAL_DATABASE_URL")
    sp.add_argument("--suite", default=None)
    sp.add_argument("--judge", default=None)
    sp.add_argument("--embed", default=None,
                    help="hashing | hf:<model> | st:<model> | openai:<model>")
    sp.add_argument("--demo", action="store_true",
                    help="seed a drifting corpus and a stale suite if the store is empty")
    sp.add_argument("--no-refit", action="store_true", help="do not deploy a scorer at startup")
    sp.add_argument("--workers", type=int, default=1,
                    help="uvicorn workers; each holds its own fitted scorer")
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("ingest", help="load traces into a store, once or by following Langfuse")
    sp.add_argument("--traces", default=None, help="jsonl glob / otel json / synthetic:drifting")
    sp.add_argument("--db", default="sqlite:///livingeval.db")
    sp.add_argument("--source", default="file")
    sp.add_argument("--follow", action="store_true", help="poll the Langfuse API continuously")
    sp.add_argument("--interval", type=float, default=30.0)
    sp.add_argument("--limit", type=int, default=200)
    sp.add_argument("--suite", default="default")
    sp.add_argument("--suite-from-first-window", action="store_true",
                    help="also curate a suite from the oldest window, i.e. a stale one")
    sp.add_argument("--suite-size", type=int, default=200)
    sp.add_argument("--windows", type=int, default=6)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("promote", help="merge confirmed proposals into the suite")
    sp.add_argument("--db", default="sqlite:///livingeval.db")
    sp.add_argument("--suite", default="default")
    sp.add_argument("--max-cases", type=int, default=None, help="retire oldest beyond this")
    sp.set_defaults(func=cmd_promote)

    sp = sub.add_parser("export", help="write human-confirmed labels as fine-tuning data")
    sp.add_argument("--db", default="sqlite:///livingeval.db")
    sp.add_argument("--suite", default="default")
    sp.add_argument("--out", default="data/finetune.jsonl")
    sp.add_argument("--format", default="chat",
                    choices=["chat", "prompt_completion", "classification"])
    sp.add_argument("--split", type=float, default=None, help="session-aware holdout fraction")
    sp.add_argument("--include-unconfirmed", action="store_true",
                    help="train on judge labels too (makes the agreement number circular)")
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("finetune", help="run the ladder with a fine-tuned rung appended")
    sp.add_argument("--traces", required=True)
    sp.add_argument("--judge", default="oracle")
    sp.add_argument("--backend", default="local", choices=["local", "unsloth", "fireworks"])
    sp.add_argument("--model", default=None)
    sp.add_argument("--n-boot", type=int, default=400)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_finetune)

    sp = sub.add_parser("compare-spaces",
                        help="does the embedding change the conclusions? measure it")
    sp.add_argument("--traces", required=True)
    sp.add_argument("--suite", default=None)
    sp.add_argument("--embed", nargs="*", default=["hashing"],
                    help="e.g. hashing hf:sentence-transformers/all-MiniLM-L6-v2")
    sp.add_argument("--suite-size", type=int, default=200)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_compare_spaces)

    sp = sub.add_parser("db", help="migrate, check or dump the database schema")
    sp.add_argument("db_command", choices=["check", "migrate", "sql"])
    sp.add_argument("--url", default=None, help="defaults to LIVINGEVAL_DATABASE_URL")
    sp.add_argument("--dialect", default="postgres", choices=["postgres", "sqlite"],
                    help="for `sql`")
    sp.add_argument("--out", default=None, help="for `sql`: write to a file")
    sp.set_defaults(func=cmd_db)

    sp = sub.add_parser("worker", help="run the measurement loop on an interval")
    sp.add_argument("--db", default=None)
    sp.add_argument("--suite", default=None)
    sp.add_argument("--judge", default=None)
    sp.add_argument("--embed", default=None)
    sp.add_argument("--interval", type=float, default=900.0)
    sp.add_argument("--mine", type=int, default=20)
    sp.add_argument("--ingest", default=None, help="trace source to pull each tick")
    sp.add_argument("--once", action="store_true", help="one tick, then exit")
    sp.add_argument("--max-ticks", type=int, default=None)
    sp.set_defaults(func=cmd_worker)

    sp = sub.add_parser("artifacts", help="list or fetch what a deployment has written")
    sp.add_argument("--url", default=None, help="defaults to LIVINGEVAL_ARTIFACT_URL")
    sp.add_argument("--prefix", default="")
    sp.add_argument("--get", default=None, help="fetch one key")
    sp.add_argument("--out", default=None)
    sp.add_argument("--limit", type=int, default=50)
    sp.set_defaults(func=cmd_artifacts)

    sp = sub.add_parser("config", help="show the resolved configuration")
    sp.add_argument("--check", action="store_true", help="also warn about deployment mistakes")
    sp.set_defaults(func=cmd_config)

    sp = sub.add_parser("supabase", help="local or cloud Supabase: status, init, env, url")
    sp.add_argument("supabase_command", choices=["status", "init", "env", "url"])
    sp.add_argument("--bucket", default="livingeval-artifacts",
                    help="Storage bucket for records and figures")
    sp.add_argument("--shell", default="bash", choices=["bash", "cmd", "dotenv"],
                    help="output format for `env`")
    sp.add_argument("--project-ref", default=None, help="for `url`: your project ref")
    sp.add_argument("--password", default=None, help="for `url`: the database password")
    sp.add_argument("--purpose", default="service", choices=["service", "session", "direct"],
                    help="for `url`: service=pooler 6543, session=pooler 5432, direct=db host")
    sp.add_argument("--region", default="ap-south-1", help="for `url`: pooler region")
    sp.set_defaults(func=cmd_supabase)

    return p


def load_dotenv(path: str = ".env") -> int:
    """Read `.env` from the working directory into the environment, if it exists.

    Without this the library and the CLI disagree about configuration: a `.env` holding
    `LIVINGEVAL_JUDGE_MODEL` works when a script loads it and is silently ignored by
    `livingeval gate`, which then falls back to a default model the user never chose and
    fails with a 404 from Ollama. Same repository, same file, different answer.

    Real environment variables always win, so `LIVINGEVAL_JUDGE_MODEL=x livingeval gate`
    and CI secrets both behave as expected.
    """
    import os
    from pathlib import Path

    file = Path(path)
    if not file.is_file():
        return 0
    loaded = 0
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")
            loaded += 1
    return loaded


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
