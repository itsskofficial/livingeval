"""Measuring what "unchanged" looks like.

A generated suite ships with estimated noise thresholds, and an estimate is not
a measurement. The estimates exist so the registry is not blank on day one; this
replaces them with the spread the suite actually shows on *your* system, against
*your* judge, with *your* golden sets. Until that has happened the gate returns
BLIND rather than PASS, because a comparison against a guessed threshold is not
evidence of stability.

The procedure is the standard one -- run the suite repeatedly with nothing
changed, and take the spread of each metric as its noise floor -- with two
corrections.

**Two sigma, not one, and never on a single pair of runs.** Two runs give a
difference, not a distribution. Below four the standard deviation is so unstable
that the threshold it produces is worse than the estimate it replaces, so this
refuses rather than pretending.

**The threshold is a floor, not the test.** Where per-case outcomes exist the
gate runs McNemar and corrects across metrics; the threshold is what covers the
metrics that have only aggregates. Recording both, and recording which metrics
got which treatment, is why the gate can say *why* it is blind.

Runs cost money. Ten runs of a twenty-metric suite is twenty times the API bill
of one, and `--runs` exists so that is the user's decision rather than a default
they discover on an invoice.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = ["MIN_RUNS", "Measurement", "measure", "update_registry"]

#: Below this a standard deviation is too unstable to be worth trusting.
MIN_RUNS = 4

#: Two sigma covers ~95% of run-to-run movement under an unchanged system.
SIGMA = 2.0


@dataclass
class Measurement:
    key: str
    mean: float
    stdev: float
    noise: float
    values: list[float]

    @property
    def line(self) -> str:
        return (f"  {self.key:<38} mean {self.mean:7.4f}  "
                f"sd {self.stdev:7.4f}  noise {self.noise:7.4f}")


def _run_once(package: Path, out: Path) -> dict:
    """One full pass of the generated suite, into a temporary artifact."""
    result = subprocess.run(
        [sys.executable, "-m", f"{package.name}.run_suite", "--out", str(out)],
        cwd=package.parent, capture_output=True, text=True,
        # The suite prints DeepEval's progress, which contains emoji, and the
        # generated harness already sets its own stdout to utf-8. Decoding it
        # here with the Windows default instead kills the reader thread after
        # every API call in the run has been paid for.
        encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(
            f"the suite failed to run:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    return json.loads(out.read_text(encoding="utf-8"))


def measure(package: Path, runs: int, scratch: Path | None = None,
            progress=print) -> tuple[dict[str, Measurement], dict]:
    """Run the suite `runs` times unchanged and measure each metric's spread.

    Returns the measurements and the last run, which becomes the baseline --
    using the *last* rather than the mean keeps `baseline.json` a real artifact
    that a real run produced, so its per-case rows pair correctly against a
    future candidate.
    """
    if runs < MIN_RUNS:
        raise ValueError(
            f"{runs} runs cannot produce a usable noise threshold. Below {MIN_RUNS} "
            f"the standard deviation is less reliable than the shipped estimate, "
            f"and a threshold nobody should trust is worse than one labelled as "
            f"a guess.")

    scratch = scratch or package / ".baseline_runs"
    scratch.mkdir(parents=True, exist_ok=True)

    collected: dict[str, list[float]] = {}
    last: dict = {}
    for index in range(1, runs + 1):
        progress(f"  run {index}/{runs}")
        last = _run_once(package, scratch / f"run{index}.json")
        for key, value in last.get("metrics", {}).items():
            collected.setdefault(key, []).append(float(value))

    measurements = {}
    for key, values in sorted(collected.items()):
        if len(values) < MIN_RUNS:
            # A metric that only appeared in some runs -- usually a golden set
            # that was skipped as incomplete. Not measurable, and saying so is
            # better than averaging over whatever did appear.
            continue
        stdev = statistics.stdev(values)
        measurements[key] = Measurement(
            key=key, mean=statistics.fmean(values), stdev=stdev,
            noise=round(SIGMA * stdev, 6), values=values)
    return measurements, last


def update_registry(registry_path: Path,
                    measurements: dict[str, Measurement]) -> int:
    """Write measured thresholds into the generated registry.

    Rewrites the literal rather than the file: the registry's docstring explains
    what the numbers mean and why direction matters, and regenerating it from a
    template would throw that away every time somebody measured.
    """
    source = registry_path.read_text(encoding="utf-8")
    namespace: dict = {}
    exec(compile(source, str(registry_path), "exec"), namespace)
    registry = namespace["REGISTRY"]

    updated = 0
    for key, entry in registry.items():
        if key in measurements:
            entry["noise"] = measurements[key].noise
            entry["measured"] = True
            entry["runs"] = len(measurements[key].values)
            updated += 1

    head = source[:source.index("REGISTRY = ")]
    body = json.dumps(registry, indent=4)
    body = (body.replace(": true", ": True").replace(": false", ": False")
                .replace(": null", ": None"))
    registry_path.write_text(f"{head}REGISTRY = {body}\n", encoding="utf-8")
    return updated
