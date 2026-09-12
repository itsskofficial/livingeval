"""Confirmed cases reaching the golden sets the suite actually runs on.

Until this existed, `drift` named the blind spots, `serve` queued cases from
them and a person confirmed them -- and the generated suite went on running the
same synthetic inputs it was born with. The loop was described in the README
and stopped one hop short of the files.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from livingeval.discover import scan
from livingeval.generate import build_goldens, emit
from livingeval.generate.adopt import adopt_into_goldens
from livingeval.plan import build_plan
from livingeval.suite.case import EvalCase
from livingeval.trace.types import Trace, Turn

APP = '''
    from openai import OpenAI

    def answer(question, store):
        docs = store.similarity_search(question, k=5)
        return OpenAI().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": str(docs) + question}],
        )
'''


@pytest.fixture
def package(tmp_path) -> Path:
    (tmp_path / "rag.py").write_text(textwrap.dedent(APP), encoding="utf-8")
    plan = build_plan(scan(tmp_path))
    goldens = build_goldens([m for p in plan.pipelines for m in p.metrics],
                            scope="course questions")
    emit(plan, goldens, tmp_path, scope="course questions")
    return tmp_path / "livingeval_evals"


def case(text: str, expected: int = 1, reviewer: str = "priya",
         tags: tuple = (), meta: dict | None = None) -> EvalCase:
    return EvalCase(
        case_id=text[:12].replace(" ", "-"),
        trace=Trace(trace_id=text[:8], turns=[Turn(role="user", content=text),
                                              Turn(role="assistant", content="...")]),
        expected=expected, provenance="confirmed", reviewer=reviewer,
        confirmed_at=1.0, tags=tags, meta=meta or {})


def read(package: Path, key: str) -> dict:
    return json.loads((package / "goldens" / f"{key}.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# what arrives
# ---------------------------------------------------------------------------


def test_a_confirmed_question_replaces_a_placeholder_rather_than_joining_it(package):
    before = read(package, "application_correctness")
    assert all(c["expected"] is None for c in before["cases"] if "expected" in c) or \
        all("expected" not in c for c in before["cases"])

    result = adopt_into_goldens(package, [case("can I get a refund after 40 days")])
    after = read(package, "application_correctness")

    assert result.added == 1
    assert result.replaced_placeholders == 1
    assert len(after["cases"]) == len(before["cases"]), "the set grew instead of filling"
    assert any(c["input"] == "can I get a refund after 40 days" for c in after["cases"])


def test_a_thumbs_up_brings_the_question_but_does_not_invent_the_answer(package):
    """An output a reviewer found acceptable is not the answer a domain expert
    would give, and those come apart on exactly the cases that matter."""
    adopt_into_goldens(package, [case("how do I change my plan")])
    row = next(c for c in read(package, "application_correctness")["cases"]
               if c["input"] == "how do I change my plan")
    assert row.get("expected") is None
    assert row["note"].startswith("TODO")
    assert read(package, "application_correctness")["complete"] is False


def test_a_reviewer_who_wrote_the_answer_has_it_recorded(package):
    adopt_into_goldens(package, [
        case("what is the refund window", meta={"expected_output": "30 days"})])
    row = next(c for c in read(package, "application_correctness")["cases"]
               if c["input"] == "what is the refund window")
    assert row["expected"] == "30 days"
    assert row["reviewer"] == "priya"


def test_a_confirmed_failure_says_so_in_the_note(package):
    """The most useful row in any suite: a question known to break the system."""
    adopt_into_goldens(package, [case("do you support SSO", expected=0)])
    row = next(c for c in read(package, "application_correctness")["cases"]
               if c["input"] == "do you support SSO")
    assert "wrong" in row["note"]


# ---------------------------------------------------------------------------
# what must not happen
# ---------------------------------------------------------------------------


def test_an_answer_a_human_wrote_is_never_overwritten(package):
    path = package / "goldens" / "application_correctness.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["cases"][0] = {"id": "correctness-01", "input": "how long is the course",
                        "kind": "sample", "expected": "twelve weeks"}
    path.write_text(json.dumps(data), encoding="utf-8")

    adopt_into_goldens(package, [case("a totally different question")] * 1)
    after = read(package, "application_correctness")
    kept = next(c for c in after["cases"] if c["id"] == "correctness-01")
    assert kept["expected"] == "twelve weeks"


def test_a_question_already_in_the_set_is_left_alone(package):
    twice = [case("how do I cancel"), case("how do I cancel")]
    result = adopt_into_goldens(package, twice)
    assert result.added == 1
    assert result.skipped_existing == 1


def test_unconfirmed_cases_are_ignored(package):
    """Nothing enters the suite until a person clicks."""
    mined = EvalCase(case_id="m1", trace=Trace(trace_id="m1", turns=[
        Turn(role="user", content="anything")]), expected=1, provenance="mined")
    result = adopt_into_goldens(package, [mined])
    assert result.added == 0
    assert "no confirmed cases" in result.note


def test_cases_carrying_a_metric_tag_go_to_that_metrics_set(package):
    """`goldens_as_suite` tags every case with its metric, so a case that came
    back round the loop knows where it belongs."""
    adopt_into_goldens(package, [
        case("summarise the refund policy", tags=("application.completeness",))])
    assert any(c["input"] == "summarise the refund policy"
               for c in read(package, "application_completeness")["cases"])


def test_a_tag_naming_no_generated_set_falls_back_rather_than_failing(package):
    result = adopt_into_goldens(package, [
        case("a question", tags=("component.nonexistent",))])
    assert result.added == 1
    assert any(c["input"] == "a question"
               for c in read(package, "application_correctness")["cases"])


def test_a_directory_with_no_goldens_reports_rather_than_raising(tmp_path):
    result = adopt_into_goldens(tmp_path / "nothing_here", [case("hello")])
    assert result.added == 0
    assert "no golden sets" in result.note
