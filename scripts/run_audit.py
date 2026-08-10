"""Reproduce the three headline findings.

    python scripts/run_audit.py            # ~6 minutes on a laptop, CPU only
    python scripts/run_audit.py --quick    # ~1 minute, same shape, wider intervals

Writes `results/audit.json` and prints the markdown tables that appear in the README.
No network, no API key, no GPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


import livingeval as le  # noqa: E402
import suites  # noqa: E402


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


# ---------------------------------------------------------------------------
# Finding 1 - a frozen suite's blind spot grows while its score does not move
# ---------------------------------------------------------------------------


def finding_one(cfg) -> dict:
    banner("FINDING 1  -  what a frozen suite stops being able to see")

    traces = le.synthetic.drifting(n=cfg.n_traces, seed=0)
    judge = le.judge.oracle(noise=0.05, seed=7)
    windows = traces.sorted_by_time().windows(cfg.n_windows)

    frozen = le.EvalSuite.from_traces(windows[0].sample(cfg.n_cases, seed=0), name="frozen")

    # The mined arm runs the library's actual loop rather than resampling at random:
    # blind spots -> proposals from the least-covered clusters -> reviewer confirms,
    # correcting the judge where it was wrong -> a bounded, rolling suite. Random
    # resampling would be a weaker comparison *and* a dishonest one, since it is not
    # what the tool does.
    state = {"suite": le.EvalSuite(list(frozen.cases), name="mined")}

    def mined(window, i):
        current = state["suite"]
        if i > 0:
            proposals = le.mine.propose(window, current, n=cfg.n_mine, judge=judge, seed=i)
            corrections = {
                p.trace.trace_id: int(p.trace.label)
                for p in proposals
                if p.trace.label is not None and p.trace.label != p.suggested_expected
            }
            confirmed = proposals.confirm(reviewer="audit", expected=corrections)
            current = current.extend(confirmed, name="mined")
            if len(current) > cfg.n_cases:  # a rolling suite, oldest cases retired
                current = le.EvalSuite(current.cases[-cfg.n_cases:], name="mined")
            state["suite"] = current
            print(f"  window {i}: mined {len(confirmed)} case(s), "
                  f"{len(corrections)} judge label(s) corrected on review")
        return current

    results = []
    for label, kwargs in (("frozen", {"suite": frozen}), ("mined", {"make_suite": mined})):
        curve = le.power.staleness_curve(
            traces, judge, label=label, n_windows=cfg.n_windows, n_sim=cfg.n_sim,
            effect=0.8, top_clusters=8, **kwargs,
        )
        print(f"\n{curve.summary()}")
        results.append(curve)

    # The crisp version: a regression confined to the intent that appeared mid-stream.
    banner("FINDING 1b  -  the same thing said in one number")
    recent = windows[-1]
    mined_now = state["suite"]
    regression = le.power.on_meta("is_new_intent", True, effect=0.5)

    localised = []
    for suite in (frozen, mined_now):
        base = le.run_suite(suite, judge, "baseline")
        res = le.power.estimate(suite, base, regression, n_sim=cfg.n_sim * 2, seed=3)
        print(f"\n{res.summary()}")
        localised.append(res)

    banner("FINDING 1c  -  what the blind-spot report says about the frozen suite")
    space = le.mine.Space.fit(recent, seed=0)
    clustering = le.mine.cluster(recent, space=space, seed=0)
    report = le.mine.blindspots(frozen, recent, space=space, clustering=clustering)
    print(report.summary())

    # And the score that did not move while all of that happened.
    first_score = le.run_suite(frozen, judge, "w0").score
    last_score = le.run_suite(frozen, judge, "wN").score
    print(
        f"\n  Throughout all of that, the frozen suite's own score went from "
        f"{first_score:.4f} to {last_score:.4f}. It is the same cases, so of course it did. "
        "That is exactly the problem: the only number anyone looks at is the one number\n"
        "  that cannot move when the traffic moves."
    )

    return {
        "curves": results,
        "localised": localised,
        "coverage": [report.coverage],
        "blindspots": [report],
        "frozen_score": first_score,
        "table": le.report.decay_table([c.as_dict() for c in results],
                                       "Coverage and detection power, window by window"),
        "coverage_table": le.report.coverage_table([report.coverage.as_dict()],
                                                   "Frozen suite against the final window"),
    }


# ---------------------------------------------------------------------------
# Finding 2 - the false-alarm rate of the gate everyone ships
# ---------------------------------------------------------------------------


def finding_two(cfg) -> dict:
    banner("FINDING 2  -  how often each gate design fires with nothing wrong")

    traces = le.synthetic.stable(n=cfg.n_traces, seed=1)
    judge = le.judge.oracle(noise=0.05, seed=11)

    rows = []
    for n_cases in cfg.sizes:
        suite = le.EvalSuite.from_traces(traces.sample(n_cases, seed=5), name=f"n{n_cases}")
        base = le.run_suite(suite, judge, "baseline")
        # Two points under the score a rerun is *expected* to produce - not two points
        # under the score observed once. Setting it against the observed score puts the
        # line above the rerun mean and the gate fires for reasons unrelated to the
        # model, which would make this table a measurement of that mistake instead.
        thresh = round(le.power.expected_rerun_score(base, flake=0.05) - 0.02, 4)

        paired = le.power.false_alarm_rate(suite, base, n_sim=cfg.n_sim * 3, seed=2)
        legacy = le.power.false_alarm_rate(
            suite, base, n_sim=cfg.n_sim * 3, seed=2, mode="threshold", threshold=thresh
        )
        rows.append({
            "kind": "false_alarm",
            "n_cases": n_cases,
            "baseline_score": base.score,
            "threshold": thresh,
            "threshold_rate": legacy.power.point,
            "threshold_ci": [legacy.power.lo, legacy.power.hi],
            "paired_rate": paired.power.point,
            "paired_ci": [paired.power.lo, paired.power.hi],
            "flake": 0.05,
        })
        print(
            f"  n={n_cases:<4} baseline={base.score:.3f} threshold={thresh:.3f}   "
            f"threshold gate {legacy.power.point:.3f}   paired gate {paired.power.point:.3f}"
        )

    print(
        "\n  Both arms are reruns of an unchanged system with 5% per-case instability.\n"
        "  Every fire in this table is a false alarm."
    )
    return {"rows": rows, "table": le.report.false_alarm_table(rows, "False-alarm rate under no regression")}


# ---------------------------------------------------------------------------
# Finding 3 - how deep the judges actually are
# ---------------------------------------------------------------------------


def finding_three(cfg) -> dict:
    banner("FINDING 3  -  validating the ladder against judges of known depth")

    ladders = []
    validations = []
    for scenario in suites.SCENARIOS.values():
        traces = scenario.traces(n=cfg.n_ladder, seed=2)
        judge = scenario.judge()
        result = le.scorer.ladder(traces, judge, n_boot=cfg.n_boot, seed=0)
        result.meta.update(scenario.as_dict())
        ladders.append(result)

        validation = le.judge.validate(judge, traces, n_boot=cfg.n_boot)
        validations.append(validation)

        agree = "matches" if (result.depth or "none") == (scenario.designed_depth or "none") else "DIFFERS FROM"
        print(f"\n{result.summary()}")
        print(f"  designed depth = {scenario.designed_depth or 'none'}  ->  measured {agree} design")

    banner("FINDING 3b  -  a global rung hides a per-cluster one")
    scenario = suites.SCENARIOS["support_triage"]
    traces = scenario.traces(n=cfg.n_ladder, seed=4)
    clustering = le.mine.cluster(traces, seed=0)
    clustering.label_traces(traces)
    per_cluster = le.scorer.ladder(traces, scenario.judge(), by="cluster", n_boot=200, seed=0)
    depths = {k: (v.depth or "none") for k, v in per_cluster.items()}
    print(f"  per-cluster depths: {depths}")

    return {
        "ladders": ladders,
        "validations": validations,
        "per_cluster_depths": depths,
        "table": le.report.ladder_table([x.as_dict() for x in ladders],
                                        "Judge depth: designed against measured"),
    }


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "audit.json"))
    ap.add_argument("--figs", default=str(ROOT / "figs"))
    args = ap.parse_args()

    class Cfg:
        quick = args.quick
        n_traces = 900 if args.quick else 2400
        n_windows = 4 if args.quick else 6
        n_cases = 120 if args.quick else 200
        n_sim = 60 if args.quick else 200
        n_ladder = 400 if args.quick else 900
        n_boot = 200 if args.quick else 600
        n_mine = 20 if args.quick else 40
        sizes = (30, 100) if args.quick else (25, 50, 100, 200, 400, 800)

    cfg = Cfg()
    t0 = time.time()

    one = finding_one(cfg)
    two = finding_two(cfg)
    three = finding_three(cfg)

    records = [
        *one["curves"], *one["localised"], *one["coverage"], *one["blindspots"],
        *two["rows"], *three["ladders"], *three["validations"],
    ]
    out = le.report.save(
        records,
        args.out,
        meta={
            "audit": "livingeval staleness audit",
            "quick": cfg.quick,
            "seconds": round(time.time() - t0, 1),
            "config": {k: getattr(cfg, k) for k in
                       ("n_traces", "n_windows", "n_cases", "n_sim", "n_ladder", "sizes")},
            "per_cluster_depths": three["per_cluster_depths"],
            "frozen_score": one["frozen_score"],
        },
    )

    banner("TABLES  (these are the ones in the README)")
    tables = [one["table"], one["coverage_table"], two["table"], three["table"]]
    for t in tables:
        print(f"\n{t}\n")

    try:
        made = le.report.figures(le.report.load(out), args.figs)
        for p in made:
            print(f"figure: {p}")
    except ImportError:
        print("(matplotlib not installed; skipping figures)")

    print(f"\nwrote {out}   ({time.time() - t0:.0f}s)")
    Path(ROOT / "results" / "audit_tables.md").write_text("\n\n".join(tables) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
