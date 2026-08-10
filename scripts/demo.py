"""The end-to-end demo — the whole loop in one run, ~3 minutes, no network.

    python scripts/demo.py                 # everything except the fine-tuned rung
    python scripts/demo.py --finetune      # + rung 5 (downloads a small encoder)
    python scripts/demo.py --embed hashing # + a second representation, compared

Then, for the video:

    livingeval serve --db demo.db

This is the script to record. It walks the loop in the order the story needs:

    1. traffic arrives and lands in a store
    2. a suite curated six windows ago still scores 0.95
    3. coverage says it represents 43% of today's traffic
    4. the blind-spot report names the clusters with zero cases
    5. detection power against a regression there is 0.01 — the gate is asleep
    6. the ladder says what the judge is actually reading, and what to deploy
    7. the online scorer answers a turn in microseconds
    8. mining proposes cases, a reviewer confirms them, the suite grows
    9. power is re-measured — and the confirmed labels export as training data
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import livingeval as le  # noqa: E402
from livingeval.serve import seed_demo  # noqa: E402
from livingeval.store import open_store  # noqa: E402

STEP = 0


def step(title: str) -> None:
    global STEP
    STEP += 1
    print(f"\n{'=' * 78}\n  {STEP}. {title}\n{'=' * 78}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(ROOT / "demo.db"))
    ap.add_argument("--n", type=int, default=2400)
    ap.add_argument("--suite-size", type=int, default=200)
    ap.add_argument("--finetune", action="store_true", help="add rung 5 (downloads a model)")
    ap.add_argument("--embed", default=None, help="also compare a second representation")
    ap.add_argument("--fresh", action="store_true", help="delete the database first")
    args = ap.parse_args()

    if args.fresh and Path(args.db).exists():
        Path(args.db).unlink()
    t_start = time.time()

    # -----------------------------------------------------------------------
    step("Production traffic arrives and lands in a store")
    store = open_store(f"sqlite:///{args.db}")
    if store.count_traces() == 0:
        info = seed_demo(store, n=args.n, suite_size=args.suite_size)
        print(f"  {info['traces']} traces ingested into {args.db}")
        print(f"  suite '{info['suite']}' curated with {info['suite_cases']} cases "
              f"from window 0 of 6 - i.e. a realistically stale one")
    traces = store.get_traces()
    suite = store.get_suite("default")
    judge = le.judge.oracle(noise=0.05, seed=7)
    windows = traces.sorted_by_time().windows(6)
    recent = windows[-1]
    print(f"  {store.stats().summary()}")

    # -----------------------------------------------------------------------
    step("The only number most tools report")
    run = le.run_suite(suite, judge, "today")
    print(run.summary())
    print(f"\n  {run.score:.2f}. Ship it?")

    # -----------------------------------------------------------------------
    step("Coverage - is that score about the traffic you serve?")
    space = le.mine.Space.fit(recent, seed=0)
    clustering = le.mine.cluster(recent, space=space, seed=0)
    coverage = le.mine.coverage(suite, recent, space=space, clustering=clustering)
    print(coverage.summary())

    # -----------------------------------------------------------------------
    step("Blind spots - which traffic, specifically")
    report = le.mine.blindspots(suite, recent, space=space, clustering=clustering, top=6)
    print(report.summary())

    # -----------------------------------------------------------------------
    step("Detection power - would the gate fire if that traffic broke?")
    regression = le.power.on_meta("is_new_intent", True, effect=0.5)
    power_before = le.power.estimate(suite, run, regression, n_sim=400)
    print(power_before.summary())

    step("The gate now has three answers, not two")
    decision = le.gate.evaluate(run.with_outcomes(run.outcomes, "current"), run,
                                coverage=coverage.coverage, power=power_before.power.point)
    print(decision.summary())
    print(f"\n  CI exits {decision.exit_code}. A plain threshold gate would have exited 0.")

    # -----------------------------------------------------------------------
    step("Is the judge a measurement?")
    print(le.judge.validate(judge, traces, n_boot=400).summary())

    step("The judge-complexity ladder")
    ladder = le.scorer.ladder(traces, judge, n_boot=400,
                              finetune={"epochs": 3, "batch_size": 16} if args.finetune else False)
    print(ladder.summary())

    # -----------------------------------------------------------------------
    if args.embed:
        step(f"Does the representation change the conclusion? ({args.embed})")
        from livingeval import embed as embed_mod

        alt_space = le.mine.Space.fit(recent, embed=embed_mod.resolve(args.embed), seed=0)
        alt_clustering = le.mine.cluster(recent, space=alt_space, seed=0)
        alt_cov = le.mine.coverage(suite, recent, space=alt_space, clustering=alt_clustering)
        print(f"  tf-idf + SVD     coverage {coverage.coverage:.3f}   k={clustering.k}")
        print(f"  {args.embed:<16} coverage {alt_cov.coverage:.3f}   k={alt_clustering.k}")
        print(f"\n  spread: {abs(coverage.coverage - alt_cov.coverage):.3f}")
        print("  A small spread means the cheap default is not costing you a conclusion.")

    # -----------------------------------------------------------------------
    step("Deploy the recommended rung and score a turn online")
    from livingeval.serve.scorer import DeployedScorer

    scorer = DeployedScorer.from_ladder(traces, judge, result=ladder)
    print(scorer.summary())
    for _ in range(200):
        scorer.score(recent[0])
    print(f"\n  200 turns scored at {scorer.mean_latency_us:.0f} us each.")
    print(f"  The original spec asked for under 50 ms. This is "
          f"{50_000 / max(scorer.mean_latency_us, 1e-9):,.0f}x under it.")

    # -----------------------------------------------------------------------
    step("Close the loop - mine, review, promote")
    proposals = le.mine.propose(recent, suite, n=40, judge=judge, report=report,
                                space=space, seed=0)
    rows = [{"trace_id": p.trace.trace_id, "cluster": p.cluster, "reason": p.reason,
             "suggested_expected": p.suggested_expected} for p in proposals]
    print(f"  queued {store.put_proposals('default', rows)} proposal(s) for review")

    # A reviewer confirms, correcting the judge where the ground truth disagrees.
    # In the real workflow this is the /review page; here it is scripted so the demo
    # runs unattended.
    corrections = 0
    for p in store.get_proposals("default", status="pending"):
        truth = p.get("human_label")
        expected = int(truth) if truth is not None else int(p["suggested"] or 0)
        if truth is not None and truth != p["suggested"]:
            corrections += 1
        store.confirm_proposal(p["id"], reviewer="demo-reviewer", expected=expected)
    print(f"  reviewer confirmed them, correcting the judge on {corrections}")

    promotion = le.feedback.promote(store, "default", max_cases=400)
    print("\n" + promotion.summary())

    # -----------------------------------------------------------------------
    step("Re-measure - did the loop buy anything?")
    grown = promotion.suite
    power_after = le.power.estimate(grown, le.run_suite(grown, judge), regression, n_sim=400)
    cov_after = le.mine.coverage(grown, recent, space=space, clustering=clustering)
    print(f"  coverage  {coverage.coverage:.3f}  ->  {cov_after.coverage:.3f}")
    print(f"  power     {power_before.power.point:.3f}  ->  {power_after.power.point:.3f}"
          f"   ({power_before.n_affected_cases} -> {power_after.n_affected_cases} affected cases)")

    step("Export the human-confirmed labels as training data")
    info = le.feedback.export_finetune_data(grown, ROOT / "data" / "demo-finetune.jsonl",
                                            fmt="chat", split=0.2)
    for k, v in info.items():
        print(f"  {k}: {v}")

    # -----------------------------------------------------------------------
    le.report.save(
        [coverage, report, power_before, power_after, ladder, decision, cov_after],
        ROOT / "results" / "demo.json",
        meta={"demo": True, "db": args.db, "seconds": round(time.time() - t_start, 1)},
    )
    print(f"\n{'=' * 78}")
    print(f"  done in {time.time() - t_start:.0f}s.  wrote results/demo.json")
    print("\n  Now run the platform against the same database:\n")
    print(f"      livingeval serve --db sqlite:///{args.db}\n")
    print("  Dashboard on :8000, review queue on /review, API docs on /docs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
