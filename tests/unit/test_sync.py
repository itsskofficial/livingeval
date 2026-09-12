"""Keeping the suite in step with the code, without destroying anyone's work.

The merge tests are the important ones. Every other failure here is an
inconvenience; losing a golden answer somebody wrote by hand is the worst thing
this tool can do, so those cases are asserted directly rather than through the
CLI.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from livingeval.discover import scan
from livingeval.generate import build_goldens, emit
from livingeval.generate.goldens import merge_golden
from livingeval.manifest import Manifest, site_digest
from livingeval.plan import build_plan
from livingeval.sync import diff, protected_files

RAG = '''
    from openai import OpenAI

    def answer(question, store):
        docs = store.similarity_search(question, k=5)
        client = OpenAI()
        return client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": str(docs) + question}],
        )
'''

AGENT = '''
    from openai import OpenAI

    def act(task, tools):
        client = OpenAI()
        for _ in range(8):
            resp = client.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": task}], tools=tools,
            )
            if not resp.choices[0].message.tool_calls:
                return resp
        return resp
'''


def project(tmp_path: Path, **modules: str) -> Path:
    for name, source in modules.items():
        (tmp_path / f"{name}.py").write_text(textwrap.dedent(source), encoding="utf-8")
    return tmp_path


def generate(root: Path, scope: str = "test scope"):
    sites = scan(root)
    plan = build_plan(sites)
    goldens = build_goldens([m for p in plan.pipelines for m in p.metrics], scope=scope)
    emit(plan, goldens, root, scope=scope)
    return plan


# ---------------------------------------------------------------------------
# the merge -- never lose an answer
# ---------------------------------------------------------------------------


def test_human_answers_survive_a_merge():
    existing = {"complete": False, "cases": [
        {"id": "correctness-01", "input": "real question",
         "expected": "the answer a human wrote"}]}
    fresh = {"complete": False, "cases": [
        {"id": "correctness-01", "input": "placeholder", "note": "TODO: answer"},
        {"id": "correctness-02", "input": "another placeholder", "note": "TODO: answer"}]}
    merged, preserved = merge_golden(existing, fresh)
    assert preserved == 1
    kept = next(c for c in merged["cases"] if c["id"] == "correctness-01")
    assert kept["expected"] == "the answer a human wrote"
    assert kept["input"] == "real question", "the human's input was replaced"


def test_new_cases_are_appended_not_substituted():
    existing = {"complete": False, "cases": [{"id": "a", "expected": "kept"}]}
    fresh = {"complete": False, "cases": [{"id": "b", "input": "new"}]}
    merged, _ = merge_golden(existing, fresh)
    assert {c["id"] for c in merged["cases"]} == {"a", "b"}


def test_cases_are_never_removed_by_a_merge():
    """A case somebody answered is evidence about the application. The scanner
    deciding it is no longer relevant is not reason enough to discard it."""
    existing = {"complete": True, "cases": [
        {"id": "gone-01", "expected": "an answer"},
        {"id": "kept-01", "expected": "another"}]}
    fresh = {"complete": True, "cases": [{"id": "kept-01", "input": "x"}]}
    merged, _ = merge_golden(existing, fresh)
    assert {c["id"] for c in merged["cases"]} == {"gone-01", "kept-01"}


def test_appending_an_unanswered_case_reopens_a_finished_set():
    existing = {"complete": True, "cases": [{"id": "a", "expected": "answered"}]}
    fresh = {"complete": True, "cases": [
        {"id": "a", "input": "x"},
        {"id": "b", "input": "y", "note": "TODO: the correct answer"}]}
    merged, _ = merge_golden(existing, fresh)
    assert merged["complete"] is False


def test_regeneration_preserves_answers_end_to_end(tmp_path):
    root = project(tmp_path, rag=RAG)
    generate(root)
    path = root / "livingeval_evals" / "goldens" / "application_correctness.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["cases"][0]["expected"] = "an afternoon of labelling"
    path.write_text(json.dumps(data), encoding="utf-8")

    generate(root)  # the whole pipeline again, as `sync` runs it
    after = json.loads(path.read_text(encoding="utf-8"))
    assert any(c.get("expected") == "an afternoon of labelling" for c in after["cases"])


# ---------------------------------------------------------------------------
# the diff
# ---------------------------------------------------------------------------


def test_a_new_call_site_is_reported_added(tmp_path):
    root = project(tmp_path, rag=RAG)
    generate(root)
    manifest = Manifest.load(root / "livingeval_evals")

    project(root, agent=AGENT)
    sites = scan(root)
    change = diff(manifest, sites, build_plan(sites))
    assert any(c.kind == "added" and "act()" in c.subject for c in change.sites)
    assert any(c.kind == "added" and c.subject == "component.tool_selection"
               for c in change.metrics)


def test_a_deleted_call_site_is_reported_removed_not_deleted(tmp_path):
    """Static analysis cannot tell deleted from moved-out-of-view, and guessing
    wrong deletes working evals."""
    root = project(tmp_path, rag=RAG, agent=AGENT)
    generate(root)
    manifest = Manifest.load(root / "livingeval_evals")

    (root / "agent.py").unlink()
    sites = scan(root)
    change = diff(manifest, sites, build_plan(sites))
    removed = [c for c in change.sites if c.kind == "removed"]
    assert removed and "act()" in removed[0].subject
    assert "not deleted" in removed[0].detail


def test_moving_a_call_site_down_the_file_is_not_a_change(tmp_path):
    """A sync that reports every site as modified after adding an import is a
    sync nobody reads."""
    root = project(tmp_path, rag=RAG)
    generate(root)
    manifest = Manifest.load(root / "livingeval_evals")

    (root / "rag.py").write_text(
        "# a new comment\n# and another\n" +
        (root / "rag.py").read_text(encoding="utf-8"), encoding="utf-8")
    sites = scan(root)
    change = diff(manifest, sites, build_plan(sites))
    assert all(c.kind == "unchanged" for c in change.sites)
    assert not change.has_changes


def test_gaining_a_component_is_a_change(tmp_path):
    root = project(tmp_path, rag='''
        from openai import OpenAI

        def answer(question):
            return OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": question}],
            )
    ''')
    generate(root)
    manifest = Manifest.load(root / "livingeval_evals")

    project(root, rag=RAG)  # the same function, now with a retriever
    sites = scan(root)
    change = diff(manifest, sites, build_plan(sites))
    assert any(c.kind == "changed" for c in change.sites)


def test_nothing_changed_reports_nothing_changed(tmp_path):
    root = project(tmp_path, rag=RAG)
    generate(root)
    manifest = Manifest.load(root / "livingeval_evals")
    sites = scan(root)
    change = diff(manifest, sites, build_plan(sites))
    assert not change.has_changes
    assert "nothing has changed" in change.summary()


# ---------------------------------------------------------------------------
# edited files
# ---------------------------------------------------------------------------


def test_an_untouched_suite_reports_no_edits(tmp_path):
    root = project(tmp_path, rag=RAG)
    generate(root)
    package = root / "livingeval_evals"
    assert protected_files(Manifest.load(package), package) == []


def test_an_edited_file_is_detected(tmp_path):
    root = project(tmp_path, rag=RAG)
    generate(root)
    package = root / "livingeval_evals"
    harness = package / "harness.py"
    harness.write_text(harness.read_text(encoding="utf-8") + "\n# mine\n", encoding="utf-8")
    assert "harness.py" in protected_files(Manifest.load(package), package)


def test_protected_files_are_skipped_by_a_rewrite(tmp_path):
    root = project(tmp_path, rag=RAG)
    plan = generate(root)
    package = root / "livingeval_evals"
    harness = package / "harness.py"
    marker = "\n# hand-tuned, must survive\n"
    harness.write_text(harness.read_text(encoding="utf-8") + marker, encoding="utf-8")

    goldens = build_goldens([m for p in plan.pipelines for m in p.metrics], scope="s")
    emission = emit(plan, goldens, root, protect={"harness.py"})
    assert "harness.py" in emission.skipped
    assert marker in harness.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# the manifest
# ---------------------------------------------------------------------------


def test_manifest_round_trips(tmp_path):
    root = project(tmp_path, rag=RAG)
    generate(root, scope="a very specific scope")
    manifest = Manifest.load(root / "livingeval_evals")
    assert manifest is not None
    assert manifest.scope == "a very specific scope"
    assert manifest.sites and manifest.files and manifest.metrics


def test_missing_manifest_is_none_not_an_exception(tmp_path):
    assert Manifest.load(tmp_path) is None


def test_site_fingerprint_ignores_line_numbers(tmp_path):
    root = project(tmp_path, rag=RAG)
    first = scan(root)[0]
    (root / "rag.py").write_text(
        "\n\n\n" + (root / "rag.py").read_text(encoding="utf-8"), encoding="utf-8")
    second = scan(root)[0]
    assert second.line != first.line
    assert site_digest(second) == site_digest(first)
