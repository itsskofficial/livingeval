"""livingeval - eval suites that tell you when they've gone blind.

Three questions an eval score does not answer, and this library does:

    does this suite still represent my traffic?    ->  mine.coverage, mine.blindspots
    would it catch the regression I fear?          ->  power.estimate, power.decay
    is the judge behind it a measurement at all?   ->  judge.validate, scorer.ladder

Quickstart, with no model, no network and no API key:

    import livingeval as le

    traces = le.synthetic.drifting(n=1200)          # a new intent appears at week 4
    judge  = le.judge.oracle()                      # stands in for your LLM judge
    suite  = le.EvalSuite.from_traces(traces[:120].sorted_by_time())

    print(le.mine.blindspots(suite, traces).summary())
    print(le.scorer.ladder(traces, judge).summary())

And a local platform - dashboard, review queue and a per-turn scoring endpoint:

    livingeval serve --demo          # http://127.0.0.1:8000
"""

from livingeval import (
    embed,
    feedback,
    gate,
    judge,
    mine,
    power,
    report,
    scorer,
    stats,
    store,
    suite,
    synthetic,
    trace,
)
from livingeval._version import SCHEMA_VERSION, __version__
from livingeval.gate import GateResult, SuiteRun, run_suite
from livingeval.suite import EvalCase, EvalSuite
from livingeval.trace import Trace, TraceSet, Turn, ingest

#: `le.load(...)` reads a saved result record; suites load with `EvalSuite.load`.
load = report.load
save = report.save

__all__ = [
    "SCHEMA_VERSION",
    "EvalCase",
    "EvalSuite",
    "GateResult",
    "SuiteRun",
    "Trace",
    "TraceSet",
    "Turn",
    "__version__",
    "embed",
    "feedback",
    "gate",
    "ingest",
    "judge",
    "load",
    "mine",
    "power",
    "report",
    "run_suite",
    "save",
    "scorer",
    "serve",
    "stats",
    "store",
    "suite",
    "synthetic",
    "trace",
]


def __getattr__(name: str):
    """`serve` is resolved lazily: importing it pulls in fastapi, and `import
    livingeval` must stay a plain offline import with no web stack. Enforced by an
    import-linter contract and by a runtime test."""
    if name == "serve":
        import livingeval.serve as _serve

        return _serve
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
