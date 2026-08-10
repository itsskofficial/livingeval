"""The background worker: keep the measurements current without anyone clicking.

The local platform computes on request, and that is right for a laptop
([DECISIONS.md #25](../../DECISIONS.md)). A deployment is different in one specific way:
nobody is watching. Coverage that is only computed when someone opens a dashboard is
coverage that is only computed when someone was already worried, which is exactly the
wrong sampling.

So a deployed stack runs a second process on a fixed interval:

    ingest new traces  ->  coverage  ->  blind spots  ->  power  ->  mine proposals
                                                                      |
                                       records + artifacts <----------+

It writes results into the store *and* the artifact store, so the history survives the
container. It **does not** promote anything: mined proposals sit in the review queue
until a named human confirms them, which is the same guard-rail as everywhere else and
the reason an unattended loop is safe to run at all.

One tick at a time, single-threaded, no scheduler library. If a tick fails, it is logged
and the next one runs; a worker that dies on one bad window is worse than one that skips
it.
"""

from __future__ import annotations

import contextlib
import logging
import signal
import time
from dataclasses import dataclass, field

from livingeval._version import __version__

__all__ = ["WorkerResult", "run_forever", "run_once"]

log = logging.getLogger("livingeval.worker")


@dataclass
class WorkerResult:
    """What one tick did."""

    tick: int
    seconds: float
    ingested: int = 0
    traces_total: int = 0
    coverage: float | None = None
    worst_cluster: dict | None = None
    power: float | None = None
    proposals_queued: int = 0
    pending_review: int = 0
    artifacts: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "kind": "worker_tick", "tick": self.tick, "seconds": round(self.seconds, 2),
            "ingested": self.ingested, "traces_total": self.traces_total,
            "coverage": self.coverage, "worst_cluster": self.worst_cluster,
            "power": self.power, "proposals_queued": self.proposals_queued,
            "pending_review": self.pending_review, "artifacts": self.artifacts,
            "errors": self.errors, "version": __version__,
        }

    def summary(self) -> str:  # pragma: no cover - display only
        bits = [f"tick {self.tick}", f"{self.traces_total} traces"]
        if self.ingested:
            bits.append(f"+{self.ingested} new")
        if self.coverage is not None:
            bits.append(f"coverage {self.coverage:.3f}")
        if self.power is not None:
            bits.append(f"power {self.power:.3f}")
        if self.proposals_queued:
            bits.append(f"queued {self.proposals_queued}")
        if self.pending_review:
            bits.append(f"{self.pending_review} awaiting review")
        if self.errors:
            bits.append(f"{len(self.errors)} error(s)")
        return f"[worker] {'  '.join(bits)}  ({self.seconds:.1f}s)"


def run_once(state, tick: int = 0, mine_n: int = 20, ingest_spec: str | None = None,
             power_effect: float = 0.4, power_sims: int = 200) -> WorkerResult:
    """One pass of the loop. Every stage is independently guarded.

    A failure in, say, the power simulation must not cost you the coverage number that
    already succeeded — so each stage records its error and the tick continues.
    """
    from livingeval import mine as mine_mod

    t0 = time.perf_counter()
    result = WorkerResult(tick=tick, seconds=0.0)

    # -- ingest ---------------------------------------------------------------
    if ingest_spec:
        try:
            from livingeval.sources import load_traces

            fresh = load_traces(ingest_spec)
            result.ingested = state.store.put_traces(fresh, source="worker")
            state._space = None  # the partition may have changed
        except Exception as e:
            result.errors.append(f"ingest: {type(e).__name__}: {e}")
            log.warning("worker ingest failed: %s", e)

    traces = state.traces()
    result.traces_total = len(traces)
    suite = state.suite()
    if suite is None or len(traces) < 40:
        result.seconds = time.perf_counter() - t0
        result.errors.append("no suite, or too few traces to measure")
        return result

    space = clustering = None
    # -- coverage and blind spots --------------------------------------------
    try:
        space, clustering = state.space_and_clustering(traces, force=True)
        report = mine_mod.blindspots(suite, traces, space=space, clustering=clustering,
                                     seed=state.seed)
        payload = report.as_dict()
        payload["coverage"] = report.coverage.as_dict()
        result.coverage = report.coverage.coverage
        if report.spots:
            worst = report.spots[0]
            result.worst_cluster = {
                "cluster": worst.cluster, "traffic_share": worst.traffic_share,
                "coverage": worst.coverage, "n_suite_cases": worst.n_suite_cases,
                "terms": worst.terms,
            }
        state.store.put_record("blindspots", payload, label=suite.name)
        state.store.put_record("coverage", report.coverage.as_dict(), label=suite.name)
        result.artifacts.append(
            state.artifacts.put_versioned(f"{suite.name}/blindspots.json", payload)["latest"]["uri"]
        )
    except Exception as e:
        result.errors.append(f"coverage: {type(e).__name__}: {e}")
        log.warning("worker coverage failed: %s", e)

    # -- power ----------------------------------------------------------------
    try:
        from livingeval import power as power_mod
        from livingeval.gate import run_suite

        if clustering is not None:
            baseline = run_suite(suite, state.judge(), "worker")
            expected = power_mod.expected_power(
                suite, baseline, clustering, effect=power_effect,
                n_sim=power_sims, seed=state.seed,
            )
            result.power = expected.expected
            state.store.put_record("power", expected.as_dict(), label=suite.name)
            result.artifacts.append(
                state.artifacts.put_versioned(
                    f"{suite.name}/power.json", expected.as_dict()
                )["latest"]["uri"]
            )
    except Exception as e:
        result.errors.append(f"power: {type(e).__name__}: {e}")
        log.warning("worker power failed: %s", e)

    # -- mine -----------------------------------------------------------------
    try:
        if mine_n and space is not None:
            proposals = mine_mod.propose(traces, suite, n=mine_n, judge=state.judge(),
                                         space=space, seed=state.seed)
            rows = [
                {"trace_id": p.trace.trace_id, "cluster": p.cluster, "reason": p.reason,
                 "suggested_expected": p.suggested_expected,
                 "judge_rationale": p.judge_rationale}
                for p in proposals
            ]
            result.proposals_queued = state.store.put_proposals(suite.name, rows)
    except Exception as e:
        result.errors.append(f"mine: {type(e).__name__}: {e}")
        log.warning("worker mining failed: %s", e)

    result.pending_review = len(state.store.get_proposals(suite.name, status="pending"))
    result.seconds = time.perf_counter() - t0
    state.store.put_record("worker_tick", result.as_dict(), label=str(tick))
    return result


def run_forever(state, interval_s: float = 900.0, mine_n: int = 20,
                ingest_spec: str | None = None, max_ticks: int | None = None) -> int:
    """Tick until told to stop. Returns the number of completed ticks.

    Handles `SIGTERM` so that an ECS task draining mid-tick finishes the tick and exits
    cleanly rather than being killed 30 seconds later with a half-written record.
    """
    stopping = {"flag": False}

    def _stop(signum, _frame):  # pragma: no cover - signal path
        log.info("received signal %s; finishing this tick then exiting", signum)
        stopping["flag"] = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        # Not the main thread, or a platform without the signal: the loop still runs,
        # it just cannot drain gracefully.
        with contextlib.suppress(ValueError, OSError):
            signal.signal(sig, _stop)

    log.info("worker starting: interval %.0fs, mine %d, ingest %s",
             interval_s, mine_n, ingest_spec or "none")
    ticks = 0
    while not stopping["flag"] and (max_ticks is None or ticks < max_ticks):
        ticks += 1
        result = run_once(state, tick=ticks, mine_n=mine_n, ingest_spec=ingest_spec)
        log.info("%s", result.summary())
        if stopping["flag"] or (max_ticks is not None and ticks >= max_ticks):
            break
        # Sleep in slices so a SIGTERM during a 15-minute wait is noticed promptly
        # instead of after the whole interval.
        remaining = interval_s
        while remaining > 0 and not stopping["flag"]:
            time.sleep(min(1.0, remaining))
            remaining -= 1.0
    log.info("worker stopped after %d tick(s)", ticks)
    return ticks
