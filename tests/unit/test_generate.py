"""What gets written, and the promises the written files have to keep."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from livingeval.discover.sites import CallSite, Evidence
from livingeval.generate import build_goldens, emit
from livingeval.plan import build_plan
from livingeval.plan.taxonomy import Archetype, Reference


@pytest.fixture
def suite(tmp_path):
    site = CallSite(path=Path("myapp/rag.py"), line=14, function="answer",
                    provider="openai", evidence=Evidence(retrieval=["similarity_search"]),
                    archetype=Archetype.RAG.value, confidence=0.6,
                    rationale=("retrieval before generation",))
    plan = build_plan([site])
    metrics = [m for p in plan.pipelines for m in p.metrics]
    goldens = build_goldens(metrics, scope="course questions")
    emission = emit(plan, goldens, tmp_path)
    return tmp_path, plan, goldens, emission


# ---------------------------------------------------------------------------
# the code
# ---------------------------------------------------------------------------


def test_every_generated_module_is_valid_python(suite):
    root, *_ = suite
    for path in (root / "livingeval_evals").glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_every_eval_module_explains_its_metrics(suite):
    """The explanation is the product. A file that cannot say why it exists
    gets deleted the first time it disagrees with its owner."""
    root, plan, *_ = suite
    for pipeline in plan.pipelines:
        source = (root / "livingeval_evals" / f"{pipeline.slug}.py").read_text(encoding="utf-8")
        doc = ast.get_docstring(ast.parse(source)) or ""
        assert "Why these metrics" in doc
        for metric in pipeline.metrics:
            assert metric.name in doc, f"{pipeline.slug} does not explain {metric.name}"
            assert metric.catches[:30] in doc


def test_harness_wires_to_the_discovered_entry_point(suite):
    """Guessing an interface produces a suite that runs against nothing and
    reports a confident zero."""
    root, *_ = suite
    source = (root / "livingeval_evals" / "harness.py").read_text(encoding="utf-8")
    assert "from myapp.rag import answer" in source
    assert "myapp/rag.py:14" in source or "myapp\\rag.py:14" in source


def test_harness_refuses_rather_than_faking_when_the_entry_point_is_unclear(tmp_path):
    site = CallSite(path=Path("x.py"), line=1, function="<module>", provider="openai",
                    evidence=Evidence(), archetype=Archetype.GENERIC.value)
    plan = build_plan([site])
    emit(plan, {}, tmp_path)
    source = (tmp_path / "livingeval_evals" / "harness.py").read_text(encoding="utf-8")
    assert "NotImplementedError" in source


def test_registry_records_direction_and_noise_for_every_metric(suite):
    """Getting direction backwards reports every safety fix as a regression."""
    root, plan, *_ = suite
    source = (root / "livingeval_evals" / "metric_registry.py").read_text(encoding="utf-8")
    namespace: dict = {}
    exec(compile(source, "metric_registry.py", "exec"), namespace)
    registry = namespace["REGISTRY"]
    for pipeline in plan.pipelines:
        for metric in pipeline.metrics:
            assert metric.key in registry
            entry = registry[metric.key]
            assert entry["direction"] == ("higher" if metric.higher_is_better else "lower")
            assert entry["noise"] >= 0


def test_registry_directions_match_the_catalogue(suite):
    """Latency counts down; a judged safety score counts up, because G-Eval and
    DeepEval 4.x both score 1 for the good outcome."""
    root, *_ = suite
    source = (root / "livingeval_evals" / "metric_registry.py").read_text(encoding="utf-8")
    namespace: dict = {}
    exec(compile(source, "metric_registry.py", "exec"), namespace)
    registry = namespace["REGISTRY"]
    assert registry["application.latency_p95"]["direction"] == "lower"
    assert registry["application.toxicity"]["direction"] == "higher"


# ---------------------------------------------------------------------------
# the datasets
# ---------------------------------------------------------------------------


def test_probe_sets_are_complete_and_carry_all_three_case_kinds(suite):
    root, *_ = suite
    path = root / "livingeval_evals" / "goldens" / "application_scope_adherence.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["complete"] is True
    kinds = {c["kind"] for c in data["cases"]}
    assert {"adversarial", "benign", "mixed"} <= kinds


def test_benign_cases_exist_for_every_probe_set(suite):
    """Without them a system that refuses everything scores perfectly."""
    _root, _, goldens, _ = suite
    for key, golden in goldens.items():
        if not any(c.kind == "adversarial" for c in golden.cases):
            continue
        assert any(c.kind == "benign" for c in golden.cases), f"{key} is all attacks"


def test_reference_sets_ship_incomplete_with_no_answers(suite):
    """The central promise: never fill these from the current output."""
    _root, _, goldens, _ = suite
    for golden in goldens.values():
        if golden.reference == Reference.BASED.value:
            assert golden.complete is False
            assert all(c.expected is None for c in golden.cases)
            assert all(c.note.startswith("TODO") for c in golden.cases)


def test_incomplete_sets_are_skipped_by_the_harness_not_scored(suite):
    root, *_ = suite
    source = (root / "livingeval_evals" / "harness.py").read_text(encoding="utf-8")
    assert 'if not data.get("complete", False)' in source


def test_scope_is_substituted_into_the_probes(suite):
    root, *_ = suite
    data = json.loads((root / "livingeval_evals" / "goldens" /
                       "application_scope_adherence.json").read_text(encoding="utf-8"))
    assert any("course questions" in c["input"] for c in data["cases"])
    assert not any("{scope}" in c["input"] for c in data["cases"])


def test_every_dataset_says_why_it_exists(suite):
    _root, _, goldens, _ = suite
    assert all(len(g.why) > 40 for g in goldens.values())


# ---------------------------------------------------------------------------
# the explanation
# ---------------------------------------------------------------------------


def test_why_document_covers_sites_pipelines_and_datasets(suite):
    root, plan, _goldens, _ = suite
    why = (root / "livingeval_evals" / "WHY.md").read_text(encoding="utf-8")
    assert "answer()" in why or "`answer()`" in why
    for pipeline in plan.pipelines:
        assert pipeline.slug in why
    assert "BLIND" in why


def test_why_document_states_the_human_boundary(suite):
    root, *_ = suite
    why = (root / "livingeval_evals" / "WHY.md").read_text(encoding="utf-8")
    assert "awaiting your answers" in why


# ---------------------------------------------------------------------------
# the metrics that own no dataset
# ---------------------------------------------------------------------------


def test_reference_free_metrics_get_inputs_rather_than_nothing(suite):
    """A metric with no cases does not fail loudly -- it drops out of the
    report, and the suite silently covers whatever was easy to generate."""
    _root, plan, goldens, _ = suite
    from livingeval.plan.taxonomy import Mechanism
    for pipeline in plan.pipelines:
        for metric in pipeline.metrics:
            if metric.mechanism is Mechanism.TELEMETRY:
                continue
            assert metric.key in goldens, f"{metric.key} would never run"
            assert goldens[metric.key].cases


def test_generated_input_sets_are_complete_and_owe_nobody_an_answer(suite):
    _root, _, goldens, _ = suite
    generated = [g for g in goldens.values()
                 if any(c.kind == "synthetic" for c in g.cases)]
    assert generated
    for golden in generated:
        assert golden.complete is True
        assert all(c.expected is None for c in golden.cases)
        assert not any("{scope}" in c.input for c in golden.cases)


def test_input_sets_are_shaped_by_what_the_metric_watches(suite):
    """Instruction-following scored on inputs carrying no instruction measures
    nothing, and scores well doing it."""
    from livingeval.generate.goldens import _INPUT_SETS
    texts = " ".join(_INPUT_SETS["instruction_following"][0]).lower()
    assert "exactly three" in texts and "one sentence" in texts
    vague = " ".join(_INPUT_SETS["clarification"][0])
    assert all(len(t) < 90 for t in _INPUT_SETS["clarification"][0]), vague


def test_the_structured_metrics_arrive_already_configured(tmp_path):
    """These are the metrics that make a generated suite worth something on
    day one. Requiring a person to retype the field names first means they do
    nothing on day one."""
    from livingeval.discover import scan

    (tmp_path / "bill.py").write_text(
        'from typing import Literal\n'
        'from pydantic import BaseModel\n'
        'from openai import OpenAI\n\n'
        'class Invoice(BaseModel):\n'
        '    vendor: str\n'
        '    currency: Literal["USD", "EUR"]\n\n'
        'def extract(text):\n'
        '    return OpenAI().responses.create(model="gpt-4o", input=text,\n'
        '                                     response_format=Invoice)\n',
        encoding="utf-8")
    plan = build_plan(scan(tmp_path))
    emit(plan, {}, tmp_path)
    source = (tmp_path / "livingeval_evals" / "harness.py").read_text(encoding="utf-8")
    assert "REQUIRED_FIELDS: list = ['vendor', 'currency']" in source
    assert "'currency': ['USD', 'EUR']" in source
    # And a parsed object is recognised without the user wrapping it.
    assert "model_dump" in source


def test_latency_mean_is_registered_with_the_direction_it_actually_has(suite):
    """The harness reports it beside the p95. Unregistered it took the default
    direction, and the first live gate reported a run that got slower as an
    improvement."""
    root, *_ = suite
    source = (root / "livingeval_evals" / "metric_registry.py").read_text(encoding="utf-8")
    namespace: dict = {}
    exec(compile(source, "metric_registry.py", "exec"), namespace)
    assert namespace["REGISTRY"]["application.latency_mean"]["direction"] == "lower"
