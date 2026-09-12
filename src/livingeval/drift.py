"""`livingeval drift` -- has the suite stopped describing the traffic?

A generated suite is written from source code, and source code says nothing
about what users actually send. The probe sets are synthetic by construction and
the reference stubs start as placeholders, so on day one the suite describes what
the *application* can do rather than what anyone *does* with it. That gap is
invisible from inside the suite: every metric can be green while most of the
traffic sits somewhere none of the cases go.

This is the same measurement `mine.coverage` already performs, pointed at the
generated golden sets instead of a hand-written suite. The bridge is small --
golden inputs become single-turn traces, which is the shape the embedding space
already understands -- and the payoff is that `init` and the analysis half stop
being two separate tools.

Two honest limits, stated because the number is useless if they are not:

**Coverage is about inputs, not correctness.** A trace sitting near a golden case
means the suite contains something like it, not that the answer was right. It
bounds what the suite *could* have caught.

**Synthetic probes flatter it.** Safety probes are written to be adversarial and
share little vocabulary with ordinary traffic, so they contribute almost nothing
to coverage of a normal day and drag the number down. The report separates
probe-derived cases from real ones for exactly that reason.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from livingeval.suite.case import EvalCase
from livingeval.suite.suite import EvalSuite
from livingeval.trace.types import Trace, TraceSet, Turn

__all__ = ["DriftReport", "goldens_as_suite", "measure_drift"]


@dataclass
class DriftReport:
    coverage: float | None
    n_traces: int
    n_cases: int
    n_probe_cases: int
    clusters: list[dict] = field(default_factory=list)
    proposals: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.coverage is None:
            return "UNKNOWN"
        return "BLIND" if self.coverage < 0.70 else "OK"

    def render(self) -> str:
        out = []
        if self.coverage is None:
            out.append("-> UNKNOWN  coverage could not be computed")
        else:
            out.append(f"-> {self.verdict}  coverage {self.coverage:.1%} of "
                       f"{self.n_traces} traces")
        for note in self.notes:
            out.append(f"   {note}")
        if self.clusters:
            out += ["", "  traffic this suite cannot see:", ""]
            for cluster in self.clusters:
                out.append(f"    {cluster['share']:6.1%} of traffic, "
                           f"{cluster['coverage']:5.1%} covered  "
                           f"{cluster['label']}")
        if self.proposals:
            out += ["", f"  {len(self.proposals)} clusters are below the coverage "
                        f"threshold. `livingeval serve` queues cases mined from "
                        f"them for review; `livingeval promote` merges the ones a "
                        f"human confirms."]
        return "\n".join(out)


def _as_trace(case_id: str, text: str, ts: float = 0.0) -> Trace:
    """One golden input as a single-turn trace.

    Single-turn because that is what a golden input is: a request, with no reply
    yet. The embedding space views requests by default, so this lines up with how
    production traces are measured without any special-casing.
    """
    return Trace(trace_id=case_id, turns=[Turn(role="user", content=text)], ts=ts)


def goldens_as_suite(package: Path, include_probes: bool = False) -> EvalSuite:
    """Every golden input in a generated suite, as an `EvalSuite`.

    `include_probes` is off by default. Safety probes are adversarial by
    construction and share little vocabulary with ordinary traffic; counting
    them as coverage of a normal day overstates how much the suite represents.
    """
    cases: list[EvalCase] = []
    directory = package / "goldens"
    if not directory.exists():
        return EvalSuite([], name="generated")

    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        metric = data.get("metric", path.stem)
        for row in data.get("cases", []):
            is_probe = row.get("kind") in {"adversarial", "benign", "mixed"}
            if is_probe and not include_probes:
                continue
            # An unanswered stub still carries the placeholder text this tool
            # wrote. Counting it as coverage would measure the boilerplate
            # against itself and report a number for a suite describing nothing.
            if row.get("expected") is None and row.get("note", "").startswith("TODO"):
                continue
            cases.append(EvalCase(
                case_id=f"{metric}:{row['id']}",
                trace=_as_trace(row["id"], row.get("input", "")),
                # The label is irrelevant to coverage, which is a geometry
                # question. 1 keeps the case gateable so nothing downstream
                # silently drops it.
                expected=1,
                provenance="curated",
                tags=(metric, row.get("kind", "sample")),
            ))
    return EvalSuite(cases, name="generated")


def count_probe_cases(package: Path) -> int:
    total = 0
    directory = package / "goldens"
    if not directory.exists():
        return 0
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        total += sum(1 for row in data.get("cases", [])
                     if row.get("kind") in {"adversarial", "benign", "mixed"})
    return total


def measure_drift(package: Path, traces: TraceSet, top: int = 5,
                  include_probes: bool = False, seed: int = 0) -> DriftReport:
    """Coverage of the generated suite over real traffic, plus what it misses."""
    from livingeval import mine

    suite = goldens_as_suite(package, include_probes=include_probes)
    probes = count_probe_cases(package)
    notes: list[str] = []

    if len(suite) == 0:
        notes.append(
            "every reference case is still an unanswered placeholder, so there is "
            "nothing real for traffic to sit near. Replace the inputs in "
            "livingeval_evals/goldens/ with questions your users actually ask -- "
            "from production traffic if you have it -- and write the answers. "
            "Or pass --include-probes to measure against the synthetic probes, "
            "which will understate coverage.")
        return DriftReport(None, len(traces), 0, probes, notes=notes)

    if len(traces) == 0:
        notes.append("no traces supplied, so drift cannot be measured")
        return DriftReport(None, 0, len(suite), probes, notes=notes)

    report = mine.blindspots(suite, traces, top=top, seed=seed)
    fraction = report.coverage.coverage

    clusters = [
        {"label": spot.terms or f"cluster {spot.cluster}",
         "share": spot.traffic_share,
         "coverage": spot.coverage,
         "example": spot.example_trace_id}
        for spot in report.spots
    ]

    if probes and not include_probes:
        notes.append(f"{probes} synthetic safety probes excluded: they are "
                     f"adversarial by construction and would overstate coverage")
    if fraction is not None and fraction < 0.70:
        notes.append("below 70%: most of your traffic is unrepresented, so a green "
                     "suite is not evidence the application is working")

    return DriftReport(
        coverage=fraction, n_traces=len(traces), n_cases=len(suite),
        n_probe_cases=probes, clusters=clusters,
        proposals=[spot.as_dict() for spot in report.spots
                   if spot.coverage < report.threshold],
        notes=notes)
