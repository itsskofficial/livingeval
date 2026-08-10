"""Layer A - the plumbing: traces, rendering, ingest, suites, provenance, records."""

from __future__ import annotations

import json

import numpy as np
import pytest

import livingeval as le
from livingeval import synthetic as syn
from livingeval.trace import ingest, render_view
from livingeval.trace.types import Trace, TraceSet, Turn

# -- traces -----------------------------------------------------------------


def test_turn_rejects_an_unknown_role():
    with pytest.raises(ValueError, match="role must be"):
        Turn(role="narrator", content="hi")


def test_trace_group_falls_back_to_the_trace_id():
    t = Trace("t1", [Turn("user", "hi")])
    assert t.group == "t1"
    assert Trace("t1", [Turn("user", "hi")], session_id="s1").group == "s1"


def test_content_hash_ignores_timestamp_label_and_meta():
    """The judge's verdict is a function of the conversation, so the cache key must
    be too - otherwise every rerun with a new timestamp is a cache miss and a bill."""
    turns = [Turn("user", "hi"), Turn("assistant", "hello")]
    a = Trace("t1", turns, ts=1.0, label=1, meta={"x": 1})
    b = Trace("t2", list(turns), ts=999.0, label=0, meta={"y": 2})
    assert a.content_hash() == b.content_hash()


def test_content_hash_changes_with_the_conversation():
    a = Trace("t1", [Turn("user", "hi")])
    b = Trace("t1", [Turn("user", "hi there")])
    assert a.content_hash() != b.content_hash()


def test_trace_round_trips_through_a_dict():
    t = Trace("t1", [Turn("user", "hi"), Turn("tool", "{}", name="lookup")],
              ts=5.0, session_id="s", label=1, meta={"k": "v"})
    assert Trace.from_dict(t.as_dict()).as_dict() == t.as_dict()


def test_windows_partition_the_stream_without_loss():
    traces = syn.stable(n=400, seed=0)
    windows = traces.sorted_by_time().windows(5)
    assert sum(len(w) for w in windows) == len(traces)
    assert {t.trace_id for w in windows for t in w} == set(traces.ids)


def test_labelled_selects_only_traces_with_ground_truth():
    traces = TraceSet([
        Trace("a", [Turn("user", "x")], label=1),
        Trace("b", [Turn("user", "y")]),
    ])
    assert len(traces.labelled()) == 1


# -- rendering --------------------------------------------------------------


def test_the_request_view_contains_no_assistant_text():
    trace = Trace("t", [Turn("user", "ASKED"), Turn("assistant", "ANSWERED")])
    assert "ASKED" in render_view(trace, "request")
    assert "ANSWERED" not in render_view(trace, "request")


def test_the_response_view_is_the_final_assistant_turn_only():
    trace = Trace("t", [Turn("assistant", "first"), Turn("user", "u"), Turn("assistant", "last")])
    assert render_view(trace, "response") == "last"


def test_the_full_view_excludes_system_turns_by_default():
    trace = Trace("t", [Turn("system", "SECRET PROMPT"), Turn("user", "hi")])
    assert "SECRET PROMPT" not in render_view(trace, "full")


def test_an_unknown_view_raises_rather_than_defaulting():
    with pytest.raises(ValueError, match="unknown view"):
        render_view(Trace("t", []), "vibes")


# -- ingest -----------------------------------------------------------------


def test_jsonl_round_trips(tmp_path):
    traces = syn.stable(n=40, seed=0)
    path = tmp_path / "t.jsonl"
    traces.to_jsonl(path)
    back = ingest.read_jsonl(str(path))
    assert len(back) == len(traces)
    assert back[0].content_hash() == traces[0].content_hash()
    assert back[0].session_id == traces[0].session_id


def test_loose_records_are_coerced():
    records = [{"id": "x1", "messages": [{"role": "human", "content": "hi"},
                                         {"role": "ai", "content": "hello"}],
                "timestamp": "2026-01-01T00:00:00Z", "sessionId": "s9"}]
    traces = ingest.from_records(records)
    assert traces[0].trace_id == "x1"
    assert [t.role for t in traces[0].turns] == ["user", "assistant"]
    assert traces[0].session_id == "s9"
    assert traces[0].ts > 0


def test_otel_spans_group_into_traces():
    spans = [
        {"trace_id": "T1", "name": "llm", "start_time_unix_nano": 1_000_000_000,
         "attributes": {"gen_ai.prompt": json.dumps([{"role": "user", "content": "hi"}]),
                        "gen_ai.completion": "hello", "session.id": "S"}},
        {"trace_id": "T1", "name": "retriever", "start_time_unix_nano": 2_000_000_000,
         "attributes": {"output": "docs"}},
        {"trace_id": "T2", "name": "llm", "start_time_unix_nano": 3_000_000_000,
         "attributes": {"gen_ai.prompt": "other", "gen_ai.completion": "reply"}},
    ]
    traces = ingest.from_spans(spans)
    assert len(traces) == 2
    first = next(t for t in traces if t.trace_id == "T1")
    assert first.session_id == "S"
    assert any(t.role == "tool" and t.name == "retriever" for t in first.turns)


def test_langfuse_export_is_read():
    records = [{"id": "lf1", "sessionId": "s1", "timestamp": "2026-02-01T10:00:00Z",
                "input": [{"role": "user", "content": "hi"}],
                "output": "hello",
                "observations": [{"type": "SPAN", "name": "search", "output": {"hits": 3}}],
                "scores": [{"name": "human", "value": 1}]}]
    traces = ingest.from_export(records)
    assert traces[0].label == 1
    assert [t.role for t in traces[0].turns] == ["user", "tool", "assistant"]


# -- suites and provenance --------------------------------------------------


def test_mined_cases_are_not_gateable_until_confirmed():
    traces = syn.stable(n=10, seed=0)
    case = le.EvalCase.from_trace(traces[0], expected=1, provenance="mined")
    assert not case.gateable
    assert case.confirm(reviewer="sk").gateable


def test_a_confirmed_case_must_name_its_reviewer():
    traces = syn.stable(n=10, seed=0)
    with pytest.raises(ValueError, match="who confirmed"):
        le.EvalCase("c", traces[0], 1, provenance="confirmed")
    with pytest.raises(ValueError, match="reviewer"):
        le.EvalCase.from_trace(traces[0], 1, provenance="mined").confirm(reviewer="")


def test_unconfirmed_cases_do_not_move_the_gate_score():
    traces = syn.stable(n=200, seed=0)
    judge = le.judge.oracle()
    suite = le.EvalSuite.from_traces(traces[:50], name="s")
    clean = le.run_suite(suite, judge).score

    wrong = [le.EvalCase(f"m{i}", t, expected=1 - int(t.label), provenance="mined")
             for i, t in enumerate(traces[50:80])]
    polluted = le.run_suite(suite.extend(wrong), judge)
    assert polluted.score == pytest.approx(clean)
    assert polluted.n_gateable == 50


def test_from_traces_refuses_to_invent_labels():
    traces = TraceSet([Trace("a", [Turn("user", "x")])])
    with pytest.raises(ValueError, match="no label"):
        le.EvalSuite.from_traces(traces)


def test_suite_content_hash_and_diff():
    traces = syn.stable(n=60, seed=0)
    a = le.EvalSuite.from_traces(traces[:30], name="a")
    b = le.EvalSuite.from_traces(traces[:30], name="a")
    assert a.content_hash() == b.content_hash()
    assert a.diff(b).empty

    c = a.extend([le.EvalCase.from_trace(traces[40], 1)])
    assert not a.diff(c).empty
    assert len(a.diff(c).added) == 1


def test_duplicate_case_ids_are_rejected():
    traces = syn.stable(n=10, seed=0)
    case = le.EvalCase.from_trace(traces[0], 1)
    with pytest.raises(ValueError, match="duplicate"):
        le.EvalSuite([case, case])


def test_suite_saves_and_loads(tmp_path):
    traces = syn.stable(n=40, seed=0)
    suite = le.EvalSuite.from_traces(traces[:20], name="s")
    path = suite.save(tmp_path / "s.json")
    assert le.EvalSuite.load(path).content_hash() == suite.content_hash()


# -- judges -----------------------------------------------------------------


def test_a_judge_with_no_human_labels_is_unvalidated():
    traces = TraceSet([Trace(f"t{i}", [Turn("user", "x"), Turn("assistant", "y")])
                       for i in range(20)])
    result = le.judge.validate(le.judge.rule(lambda t: 1), traces)
    assert result.status == "UNVALIDATED"
    assert result.kappa is None


def test_judge_validation_recovers_a_known_noise_level():
    """The oracle flips a known fraction of labels, so kappa has an expected value
    and the validation code can be checked rather than trusted."""
    traces = syn.stable(n=1200, seed=0)
    result = le.judge.validate(le.judge.oracle(noise=0.10, seed=1), traces, n_boot=300)
    assert result.status == "VALIDATED"
    assert 0.55 < result.kappa.point < 0.85
    assert result.kappa.lo <= result.kappa.point <= result.kappa.hi


def test_a_perfect_judge_scores_kappa_one():
    traces = syn.stable(n=300, seed=0)
    result = le.judge.validate(le.judge.oracle(), traces, n_boot=200)
    assert result.kappa.point == pytest.approx(1.0)


def test_a_small_labelled_set_is_flagged_underpowered_not_hidden():
    traces = syn.stable(n=200, seed=0)
    subset = TraceSet(list(traces[:10]))
    result = le.judge.validate(le.judge.oracle(noise=0.2, seed=0), subset, n_boot=100)
    assert result.status == "UNDERPOWERED"
    assert result.kappa is not None


def test_the_keyword_judge_is_reproduced_by_rung_two():
    """A straw man with a known depth: if the ladder cannot recover this, it is
    broken."""
    traces = syn.stable(n=500, seed=0)
    judge = le.judge.keyword_judge(syn.SHORTCUT_PHRASE)
    result = le.scorer.ladder(traces, judge, n_boot=100)
    assert result.depth == "keyword"


def test_verdict_cache_avoids_a_second_call(tmp_path):
    from livingeval.judge.cache import VerdictCache

    cache = VerdictCache("fp", tmp_path)
    cache.put("k", {"label": 1})
    cache.flush()
    assert VerdictCache("fp", tmp_path).get("k") == {"label": 1}
    assert VerdictCache("other-fingerprint", tmp_path).get("k") is None


# -- gate -------------------------------------------------------------------


def test_the_gate_returns_three_states_with_distinct_exit_codes():
    from livingeval.gate import EXIT_CODES

    assert EXIT_CODES == {"PASS": 0, "FAIL": 1, "BLIND": 2}


def test_a_clean_rerun_passes_and_a_real_regression_fails():
    traces = syn.stable(n=400, seed=0)
    judge = le.judge.oracle()
    suite = le.EvalSuite.from_traces(traces.sample(120, seed=0), name="s")
    base = le.run_suite(suite, judge)

    assert le.gate.evaluate(base.with_outcomes(base.outcomes, "current"), base).decision == "PASS"

    broken = base.outcomes.copy()
    broken[np.flatnonzero(broken == 1)[:25]] = 0
    assert le.gate.evaluate(base.with_outcomes(broken, "current"), base).decision == "FAIL"


def test_low_coverage_turns_a_pass_into_blind():
    traces = syn.stable(n=400, seed=0)
    judge = le.judge.oracle()
    suite = le.EvalSuite.from_traces(traces.sample(120, seed=0), name="s")
    base = le.run_suite(suite, judge)
    current = base.with_outcomes(base.outcomes, "current")

    assert le.gate.evaluate(current, base, coverage=0.95).decision == "PASS"
    result = le.gate.evaluate(current, base, coverage=0.40)
    assert result.decision == "BLIND"
    assert result.exit_code == 2
    assert "coverage" in result.blind_reasons[0]


def test_a_detected_regression_beats_blindness():
    """Precedence matters: a suite that caught something has done its job, and
    downgrading that to BLIND would suppress a real signal."""
    traces = syn.stable(n=400, seed=0)
    judge = le.judge.oracle()
    suite = le.EvalSuite.from_traces(traces.sample(120, seed=0), name="s")
    base = le.run_suite(suite, judge)
    broken = base.outcomes.copy()
    broken[np.flatnonzero(broken == 1)[:25]] = 0
    result = le.gate.evaluate(base.with_outcomes(broken, "current"), base, coverage=0.10)
    assert result.decision == "FAIL"


def test_a_gate_without_a_baseline_says_so():
    traces = syn.stable(n=200, seed=0)
    suite = le.EvalSuite.from_traces(traces.sample(60, seed=0), name="s")
    result = le.gate.evaluate(le.run_suite(suite, le.judge.oracle()), None, threshold=0.5)
    assert result.mode == "threshold"
    assert any("no baseline" in r for r in result.reasons)


# -- records ----------------------------------------------------------------


def test_records_round_trip_and_figures_need_no_traces(tmp_path):
    traces = syn.stable(n=300, seed=0)
    suite = le.EvalSuite.from_traces(traces.sample(80, seed=0), name="s")
    cov = le.mine.coverage(suite, traces)
    path = le.report.save([cov], tmp_path / "r.json")
    record = le.report.load(path)
    assert record.first("coverage")["suite"] == "s"
    assert le.report.coverage_table(record).startswith("|")


def test_a_future_schema_is_refused_rather_than_misread(tmp_path):
    path = tmp_path / "future.json"
    path.write_text(json.dumps({"schema": 999, "results": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema 999"):
        le.report.load(path)
