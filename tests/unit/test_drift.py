"""Coverage of a generated suite over real traffic."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from livingeval.discover import scan
from livingeval.drift import count_probe_cases, goldens_as_suite, measure_drift
from livingeval.generate import build_goldens, emit
from livingeval.plan import build_plan
from livingeval.trace.types import Trace, TraceSet, Turn

RAG = '''
    from openai import OpenAI

    def answer(question, store):
        docs = store.similarity_search(question, k=5)
        return OpenAI().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": str(docs) + question}],
        )
'''


def suite_at(tmp_path: Path) -> Path:
    (tmp_path / "rag.py").write_text(textwrap.dedent(RAG), encoding="utf-8")
    sites = scan(tmp_path)
    plan = build_plan(sites)
    goldens = build_goldens([m for p in plan.pipelines for m in p.metrics],
                            scope="course questions")
    emit(plan, goldens, tmp_path, scope="course questions")
    return tmp_path / "livingeval_evals"


def traces(texts: list[str]) -> TraceSet:
    return TraceSet([
        Trace(trace_id=f"t{i}", ts=float(i),
              turns=[Turn(role="user", content=text),
                     Turn(role="assistant", content="...")])
        for i, text in enumerate(texts)
    ])


def answer_the_goldens(package: Path, texts: list[str]) -> None:
    """Stand in for a human filling in the correctness set."""
    path = package / "goldens" / "application_correctness.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for case, text in zip(data["cases"], texts, strict=False):
        case["input"] = text
        case["expected"] = "an answer"
        case.pop("note", None)
    path.write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# the bridge
# ---------------------------------------------------------------------------


def test_probes_are_excluded_from_the_suite_by_default(tmp_path):
    """Adversarial probes share little vocabulary with ordinary traffic;
    counting them as coverage of a normal day overstates it."""
    package = suite_at(tmp_path)
    assert count_probe_cases(package) > 0
    without = goldens_as_suite(package, include_probes=False)
    with_probes = goldens_as_suite(package, include_probes=True)
    assert len(with_probes) > len(without)


def test_golden_inputs_become_single_turn_traces(tmp_path):
    package = suite_at(tmp_path)
    suite = goldens_as_suite(package, include_probes=True)
    assert len(suite) > 0
    for case in suite:
        assert len(case.trace.turns) == 1
        assert case.trace.turns[0].role == "user"


def test_cases_carry_their_metric_as_a_tag(tmp_path):
    package = suite_at(tmp_path)
    suite = goldens_as_suite(package, include_probes=True)
    assert any("application.toxicity" in case.tags for case in suite)


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------


def test_traffic_matching_the_goldens_is_covered(tmp_path):
    package = suite_at(tmp_path)
    shared = ["how long is the machine learning course",
              "what are the prerequisites for the deep learning track",
              "can I get a refund if I drop out midway",
              "does the programme include live classes",
              "what is the fee for the advanced module"]
    answer_the_goldens(package, shared)
    report = measure_drift(package, traces(shared * 8))
    assert report.coverage is not None and report.coverage > 0.5
    assert report.verdict == "OK"


def test_traffic_the_suite_never_saw_is_reported_blind(tmp_path):
    package = suite_at(tmp_path)
    answer_the_goldens(package, ["how long is the machine learning course"] * 5)
    unseen = ["my payment failed three times what should I do",
              "the invoice has the wrong VAT number on it",
              "can I transfer this subscription to a colleague",
              "billing charged me twice this month"]
    report = measure_drift(package, traces(unseen * 10))
    assert report.verdict == "BLIND"
    assert report.coverage is not None and report.coverage < 0.5


def test_blind_spots_name_the_traffic_that_is_missed(tmp_path):
    """A number tells you to act; the cluster table tells you what to write."""
    package = suite_at(tmp_path)
    answer_the_goldens(package, ["how long is the course"] * 5)
    report = measure_drift(package, traces(
        ["refund for the annual plan please"] * 20 +
        ["the invoice VAT number is wrong"] * 20))
    assert report.clusters
    labels = " ".join(c["label"] for c in report.clusters)
    assert "refund" in labels or "invoice" in labels or "vat" in labels.lower()


def test_a_suite_with_only_placeholders_says_so_rather_than_scoring(tmp_path):
    """Placeholder inputs are not traffic. Reporting a number from them would
    be measuring the tool's own boilerplate."""
    package = suite_at(tmp_path)
    report = measure_drift(package, traces(["a real question"] * 20))
    assert report.coverage is None
    assert report.verdict == "UNKNOWN"
    assert any("placeholder" in note for note in report.notes)


def test_no_traces_is_reported_not_divided_by_zero(tmp_path):
    package = suite_at(tmp_path)
    answer_the_goldens(package, ["a question"] * 5)
    report = measure_drift(package, TraceSet([]))
    assert report.coverage is None
    assert any("no traces" in note for note in report.notes)


def test_report_renders_without_raising(tmp_path):
    package = suite_at(tmp_path)
    answer_the_goldens(package, ["how long is the course"] * 5)
    text = measure_drift(package, traces(["something else entirely"] * 20)).render()
    assert "coverage" in text
