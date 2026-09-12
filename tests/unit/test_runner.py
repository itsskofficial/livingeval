"""The generated runner, executed.

These are the only tests that run the emitted package as a program. They exist
because the two worst bugs this suite has had were invisible to every test that
only read the generated source: one merged each pipeline's result envelope into
the results map, so the file ended up with two keys in it and every score but
the last pipeline's discarded, and the other left whole metrics with no dataset,
so they vanished from the report rather than failing.

Both looked like a working run. That is the failure mode worth a subprocess.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from livingeval.discover import scan
from livingeval.generate import build_goldens, emit
from livingeval.plan import build_plan

APP = '''
    from openai import OpenAI

    def answer(question, store):
        docs = store.similarity_search(question, k=5)
        return OpenAI().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": str(docs) + question}],
        )
'''

STUB = '''
def run() -> dict:
    return {{
        "metrics": {{"{name}.score": 0.5}},
        "cases": {{"{name}.score": [{{"id": "a", "score": 0.5, "passed": True}}]}},
        "unmeasured": {{"{name}.missing": "nothing wired it"}},
    }}
'''


@pytest.fixture
def generated(tmp_path):
    (tmp_path / "rag.py").write_text(textwrap.dedent(APP), encoding="utf-8")
    plan = build_plan(scan(tmp_path))
    goldens = build_goldens([m for p in plan.pipelines for m in p.metrics],
                            scope="course questions")
    emit(plan, goldens, tmp_path, scope="course questions")
    return tmp_path, plan


def _stub_out_the_pipelines(package: Path, plan) -> list[str]:
    """Replace every eval module with one that returns a known envelope.

    The point is the runner's bookkeeping, not the judge's opinion, so nothing
    here needs an API key.
    """
    names = [p.slug for p in plan.pipelines]
    for name in names:
        (package / f"{name}.py").write_text(STUB.format(name=name), encoding="utf-8")
    return names


def test_every_pipelines_scores_survive_into_the_result_file(generated):
    """Merging the envelope instead of its contents leaves a file with two keys
    in it, which still looks like a successful run."""
    root, plan = generated
    names = _stub_out_the_pipelines(root / "livingeval_evals", plan)
    assert len(names) > 1, "this test needs more than one pipeline to mean anything"

    done = subprocess.run([sys.executable, "-m", "livingeval_evals.run_suite"],
                          cwd=root, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr

    written = json.loads((root / "livingeval_evals" / "baseline.json").read_text())
    assert set(written["metrics"]) == {f"{n}.score" for n in names}
    assert set(written["cases"]) == {f"{n}.score" for n in names}
    assert set(written["unmeasured"]) == {f"{n}.missing" for n in names}


def test_metrics_that_produced_no_number_are_named_in_the_output(generated):
    root, plan = generated
    _stub_out_the_pipelines(root / "livingeval_evals", plan)
    done = subprocess.run([sys.executable, "-m", "livingeval_evals.run_suite"],
                          cwd=root, capture_output=True, text=True)
    assert "produced no number" in done.stdout
    assert "nothing wired it" in done.stdout


def test_the_second_run_writes_a_candidate_rather_than_overwriting_the_baseline(generated):
    root, plan = generated
    _stub_out_the_pipelines(root / "livingeval_evals", plan)
    for _ in range(2):
        subprocess.run([sys.executable, "-m", "livingeval_evals.run_suite"],
                       cwd=root, capture_output=True, text=True, check=True)
    assert (root / "livingeval_evals" / "baseline.json").exists()
    assert (root / "livingeval_evals" / "candidate.json").exists()


def test_a_raising_application_is_a_measurement_not_a_crashed_run(generated):
    """error_rate is the metric. An exception that kills the process instead
    discards every judge call already paid for on the cases before it."""
    root, _ = generated
    harness = root / "livingeval_evals" / "harness.py"
    source = harness.read_text(encoding="utf-8")
    source = source.replace(
        "    result = answer(question)",
        "    raise RuntimeError('the app is down')\n    result = answer(question)")
    harness.write_text(source, encoding="utf-8")

    probe = textwrap.dedent("""
        import json, sys
        sys.path.insert(0, ".")
        from livingeval_evals.harness import call_once, finalise
        call_once("hi")
        call_once("hello")
        scores, unmeasured = finalise(["application.error_rate",
                                       "application.cost_per_query"])
        print(json.dumps({"scores": scores, "unmeasured": unmeasured}))
    """)
    (root / "probe.py").write_text(probe, encoding="utf-8")
    done = subprocess.run([sys.executable, "probe.py"], cwd=root,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["scores"]["application.error_rate"] == 1.0
    assert out["scores"]["application.latency_p95"] >= 0
    # And a metric it genuinely cannot see says why, rather than reporting zero.
    assert "cost" in out["unmeasured"]["application.cost_per_query"]


def test_telemetry_over_no_calls_is_unmeasured_not_a_clean_bill_of_health(generated):
    """The operational pipeline sends no cases of its own. Measured per
    pipeline, error_rate there is zero errors over zero calls -- which reads
    exactly like a healthy application."""
    root, _ = generated
    probe = textwrap.dedent("""
        import json, sys
        sys.path.insert(0, ".")
        from livingeval_evals.harness import finalise
        scores, unmeasured = finalise(["application.error_rate"])
        print(json.dumps({"scores": scores, "unmeasured": unmeasured}))
    """)
    (root / "probe.py").write_text(probe, encoding="utf-8")
    done = subprocess.run([sys.executable, "probe.py"], cwd=root,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert "application.error_rate" not in out["scores"]
    assert "application.error_rate" in out["unmeasured"]
