"""One command for the whole local stack.

    python scripts/dev.py up        # Supabase + migrations + bucket + seed + scorer
    python scripts/dev.py serve     # ...and run the API against it
    python scripts/dev.py status
    python scripts/dev.py env       # print the exports, for your own shell
    python scripts/dev.py reset     # wipe livingeval's data, keep the stack
    python scripts/dev.py down

The point of running the **real** Supabase locally rather than SQLite is that the only
difference between here and production becomes credentials. Same Postgres, same
pgBouncer semantics, same Storage API, same RLS, same migrations. "It worked locally"
then means something.

Nothing here needs a key. The local stack's service-role key is a published constant
that only signs against a local secret. The keys you do eventually need — OpenAI,
Anthropic, Langfuse — are for things with no local equivalent, and everything else runs
without them.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BUCKET = "livingeval-artifacts"


def supabase_cmd() -> list[str]:
    """Prefer a real `supabase` on PATH; fall back to npx, which needs no install.

    On Windows the npm-installed `npx` is `npx.CMD`, a **batch file**. `subprocess`
    calls `CreateProcess`, which cannot execute a batch file at all — not by bare name
    and not by absolute path. It fails with a bare `FileNotFoundError` that names
    neither npx nor the reason, which is a genuinely confusing five minutes.

    So `_executable` runs a `.cmd`/`.bat` through the command interpreter and passes
    everything else straight through. `shell=True` would also work, and would turn every
    argument into a quoting hazard.
    """
    resolved = shutil.which("supabase")
    if resolved:
        return _executable(resolved)
    resolved = shutil.which("npx")
    if resolved:
        return [*_executable(resolved), "--yes", "supabase@2.113.0"]
    raise SystemExit(
        "the Supabase CLI is not available.\n"
        "  npm install -g supabase       (or) scoop install supabase\n"
        "  https://supabase.com/docs/guides/local-development"
    )


def _executable(path: str) -> list[str]:
    """Wrap Windows batch files in the command interpreter; pass anything else through."""
    if os.name == "nt" and path.lower().endswith((".cmd", ".bat")):
        return [os.environ.get("COMSPEC", "cmd.exe"), "/c", path]
    return [path]


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=ROOT, **kw)


def heading(text: str) -> None:
    print(f"\n{'=' * 74}\n  {text}\n{'=' * 74}")


def apply_env() -> dict:
    """Resolve Supabase and put it in this process's environment."""
    from livingeval.store import supabase as sb

    config = sb.detect(require=True)
    assert config is not None
    env = config.env()
    env["LIVINGEVAL_ARTIFACT_URL"] = f"supabase://{BUCKET}/livingeval"
    os.environ.update(env)
    return {"config": config, "env": env}


# ---------------------------------------------------------------------------


def cmd_up(args) -> int:
    heading("1. Local Supabase")
    from livingeval.store.supabase import _local_is_running

    if _local_is_running():
        print("  already running")
    else:
        # The excluded services are the ones livingeval does not use. Skipping them
        # saves a few hundred MB of images and a good chunk of start-up time.
        result = run([*supabase_cmd(), "start", "-x",
                      "realtime,imgproxy,edge-runtime,logflare,vector,supavisor,mailpit"])
        if result.returncode != 0:
            return result.returncode

    info = apply_env()
    config = info["config"]
    print(f"  api      {config.api_url}")
    print(f"  db       {config.db_url}")
    print(f"  studio   {config.studio_url or 'http://127.0.0.1:54423'}")

    heading("2. Schema, RLS and the Storage bucket")
    from livingeval.cli import main as cli_main

    cli_main(["supabase", "init", "--bucket", BUCKET])

    heading("3. Seed the drifting corpus")
    from livingeval.serve import seed_demo
    from livingeval.store import open_store

    store = open_store(os.environ["LIVINGEVAL_DATABASE_URL"])
    if store.count_traces() >= 100 and not args.force:
        print(f"  store already holds {store.count_traces()} traces; --force to reseed")
    else:
        result = seed_demo(store, n=args.n, suite_size=args.suite_size)
        print(f"  {result['traces']} traces, suite '{result['suite']}' "
              f"with {result['suite_cases']} cases ({result['note']})")
    print(f"  {store.stats().summary()}")
    store.close()

    heading("Ready")
    print("  python scripts/dev.py serve      # the dashboard on :8000")
    print("  python scripts/dev.py env        # exports for your own shell")
    print(f"  studio: {config.studio_url or 'http://127.0.0.1:54423'}\n")
    return 0


def cmd_serve(args) -> int:
    apply_env()
    from livingeval import serve

    # The suite and judge have to be passed through, not left to default. Without them
    # the platform comes up on the suite literally named "default" with the oracle
    # judge, so the dashboard reads zero and the review queue looks empty even when
    # there are proposals waiting under the real suite name - a confusing five minutes
    # that looks like the mining step silently failed.
    serve.run(port=args.port, suite=args.suite, judge=args.judge, demo=False, refit=True)
    return 0


def cmd_worker(args) -> int:
    apply_env()
    from livingeval.cli import main as cli_main

    return cli_main(["worker", "--interval", str(args.interval), "--mine", str(args.mine)])


def cmd_status(args) -> int:
    from livingeval.store.supabase import _local_is_running

    if not _local_is_running():
        print("local Supabase is not running.  python scripts/dev.py up")
        return 1
    apply_env()
    from livingeval.cli import main as cli_main

    return cli_main(["supabase", "status"])


def cmd_env(args) -> int:
    info = apply_env()
    prefix = "" if args.shell == "dotenv" else "export "
    for k, v in info["env"].items():
        print(f"{prefix}{k}={v}")
    return 0


def cmd_reset(args) -> int:
    """Drop livingeval's rows, keep the stack and the schema."""
    apply_env()
    from livingeval.store import open_store

    store = open_store(os.environ["LIVINGEVAL_DATABASE_URL"])
    with store._cur(dict_rows=False) as cur:
        for table in ("online_scores", "records", "proposals", "suites", "verdicts", "traces"):
            cur.execute(f"TRUNCATE TABLE {table} CASCADE")
    print("  truncated livingeval tables")
    print(f"  {store.stats().summary()}")
    store.close()
    return 0


def cmd_down(args) -> int:
    return run([*supabase_cmd(), "stop"] + (["--no-backup"] if args.wipe else [])).returncode


def cmd_test(args) -> int:
    """Run the Supabase-marked tests against the running local stack."""
    apply_env()
    os.environ["LIVINGEVAL_TEST_POSTGRES_URL"] = os.environ["LIVINGEVAL_DATABASE_URL"]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-m", "postgres or supabase_stack", "-v"],
        cwd=ROOT, env=env,
    ).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("up", help="start Supabase, migrate, create the bucket, seed")
    sp.add_argument("--n", type=int, default=2400)
    sp.add_argument("--suite-size", type=int, default=200)
    sp.add_argument("--force", action="store_true", help="reseed even if data exists")
    sp.set_defaults(func=cmd_up)

    sp = sub.add_parser("serve", help="run the API against local Supabase")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--suite", default="support-week1")
    sp.add_argument("--judge", default="ollama")
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("worker", help="run the measurement loop against local Supabase")
    sp.add_argument("--interval", type=float, default=300.0)
    sp.add_argument("--mine", type=int, default=20)
    sp.set_defaults(func=cmd_worker)

    sp = sub.add_parser("status", help="what is running and what is in it")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("env", help="print the environment to export")
    sp.add_argument("--shell", default="bash", choices=["bash", "dotenv"])
    sp.set_defaults(func=cmd_env)

    sp = sub.add_parser("reset", help="truncate livingeval's tables")
    sp.set_defaults(func=cmd_reset)

    sp = sub.add_parser("down", help="stop the local Supabase stack")
    sp.add_argument("--wipe", action="store_true", help="also delete the data volume")
    sp.set_defaults(func=cmd_down)

    sp = sub.add_parser("test", help="run the Supabase-backed tests against the local stack")
    sp.set_defaults(func=cmd_test)

    args = ap.parse_args()
    t0 = time.time()
    code = int(args.func(args) or 0)
    if args.command in ("up", "reset"):
        print(f"  ({time.time() - t0:.0f}s)")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
