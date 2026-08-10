"""Record the Layer B golden fixtures.

Run deliberately, never automatically:

    python scripts/record_goldens.py

A test suite that rewrites its own expectations when they fail verifies nothing. The
whole value of a golden file is that changing it is a separate, visible, reviewable
act - so this is a script you run and a diff someone reads, not a `--update` flag on
pytest.

Golden numbers here come from a **pinned deterministic config**, not from a
publication. For catching numerical drift from a refactor or a scikit-learn release,
a synthetic fixture is strictly better than a real corpus: faster, no download, no
licence question, no GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import livingeval as le  # noqa: E402
from livingeval import synthetic as syn  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "golden.json"


def build() -> dict:
    traces = syn.shortcut(n=300, seed=42)
    judge = le.judge.oracle(noise=0.1, seed=42)
    suite = le.EvalSuite.from_traces(traces.sample(80, seed=42), name="golden")

    ladder = le.scorer.ladder(traces, judge, n_boot=200, seed=42)
    validation = le.judge.validate(judge, traces, n_boot=200, seed=42)
    space = le.mine.Space.fit(traces, seed=42)
    clustering = le.mine.cluster(traces, space=space, k=5, seed=42)
    coverage = le.mine.coverage(suite, traces, space=space, clustering=clustering)
    run = le.run_suite(suite, judge)
    fa = le.power.false_alarm_rate(suite, run, n_sim=200, seed=42)

    return {
        "_config": {
            "generator": "shortcut(n=300, seed=42)",
            "judge": "oracle(noise=0.1, seed=42)",
            "suite": "sample(80, seed=42)",
            "note": "regenerate with scripts/record_goldens.py; never automatically",
        },
        "trace_content_hash": traces[0].content_hash(),
        "suite_content_hash": suite.content_hash(),
        "ladder_depth": ladder.depth,
        "ladder_kappa": {r.name: round(r.kappa.point, 6) for r in ladder.rungs},
        "judge_kappa": round(validation.kappa.point, 6),
        "judge_status": validation.status,
        "clustering_sizes": {str(k): v for k, v in clustering.sizes.items()},
        "coverage": round(coverage.coverage, 6),
        "coverage_radius": round(coverage.radius, 6),
        "suite_score": round(run.score, 6),
        "false_alarm_paired": round(fa.power.point, 6),
    }


if __name__ == "__main__":
    payload = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT}")
