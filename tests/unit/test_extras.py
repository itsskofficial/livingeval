"""Layer A for the optional subsystems: store, embeddings, feedback, and laziness.

The most important test in this file is the first one. Everything else in the library
rests on `pip install livingeval` being a plain offline import — that is what lets the
whole suite run on every commit in seconds instead of being skipped in CI. The
`import-linter` contracts approximate it statically; this checks it for real, in a
subprocess, by looking at what actually got loaded.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

import livingeval as le
from livingeval import embed, feedback
from livingeval.store import SQLiteStore, open_store
from livingeval.suite import EvalCase, EvalSuite

# ---------------------------------------------------------------------------
# the guarantee the whole design rests on
# ---------------------------------------------------------------------------


def test_importing_livingeval_does_not_load_optional_deps():
    """A full analysis run must not pull torch, transformers, fastapi or matplotlib
    into the process.

    Checked in a subprocess against `sys.modules`, because the property is about what
    the import machinery actually did — a static import graph cannot distinguish a
    deferred import from a module-scope one, and this is the thing that matters.
    """
    script = """
import sys, json
import livingeval as le
traces = le.synthetic.drifting(n=200, seed=0)
suite = le.EvalSuite.from_traces(traces[:60], name="s")
le.mine.blindspots(suite, traces, top=2)
le.scorer.ladder(traces, le.judge.oracle(), n_boot=20)
le.gate.evaluate(le.run_suite(suite, le.judge.oracle()), None, threshold=0.5)
le.store.open_store("sqlite://:memory:").put_traces(traces)
le.mine.Space.fit(traces, embed=le.embed.hashing(dim=32, cache=False))
heavy = [m for m in ("torch", "transformers", "fastapi", "matplotlib", "openai",
                     "anthropic", "psycopg2", "uvicorn") if m in sys.modules]
print(json.dumps(heavy))
"""
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    leaked = json.loads(out.stdout.strip().splitlines()[-1])
    assert leaked == [], f"optional dependencies leaked into the default import path: {leaked}"


def test_serve_is_not_imported_until_asked_for():
    script = """
import sys, json
import livingeval as le
print(json.dumps("livingeval.serve" in sys.modules))
"""
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert json.loads(out.stdout.strip().splitlines()[-1]) is False


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------


@pytest.fixture
def store():
    s = SQLiteStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def traces():
    return le.synthetic.drifting(n=300, seed=0)


def test_traces_round_trip_through_the_store(store, traces):
    assert store.put_traces(traces, source="test") == len(traces)
    back = store.get_traces()
    assert len(back) == len(traces)
    assert back[0].content_hash() == traces[0].content_hash()
    assert back[0].session_id == traces[0].session_id
    assert back[0].label == traces[0].label
    assert back[0].meta["intent"] == traces[0].meta["intent"]


def test_put_traces_is_idempotent(store, traces):
    """Replaying a file or re-polling an overlapping window must not duplicate."""
    store.put_traces(traces)
    store.put_traces(traces)
    assert store.count_traces() == len(traces)


def test_limit_returns_the_newest_traces_oldest_first(store, traces):
    """A window analysis reads forward in time, so a reversed result would silently
    invert every drift curve."""
    store.put_traces(traces)
    recent = store.get_traces(limit=50)
    assert len(recent) == 50
    assert list(recent.timestamps) == sorted(recent.timestamps)
    assert recent.tmax == traces.sorted_by_time().tmax


def test_time_window_queries(store, traces):
    store.put_traces(traces)
    lo, hi = store.trace_time_span()
    mid = (lo + hi) / 2
    early, late = store.get_traces(until=mid), store.get_traces(since=mid)
    assert len(early) + len(late) == len(traces)


def test_suites_round_trip(store, traces):
    suite = EvalSuite.from_traces(traces[:60], name="s")
    store.put_suite(suite)
    assert store.get_suite("s").content_hash() == suite.content_hash()
    assert store.get_suite("missing") is None
    assert store.list_suites()[0]["n_cases"] == 60


def test_the_review_queue_records_who_decided(store, traces):
    store.put_traces(traces)
    rows = [{"trace_id": t.trace_id, "cluster": 1, "reason": "r", "suggested_expected": 1}
            for t in traces[:5]]
    assert store.put_proposals("s", rows) == 5
    assert store.put_proposals("s", rows) == 0, "re-mining must not duplicate the queue"

    pending = store.get_proposals("s")
    assert len(pending) == 5
    store.confirm_proposal(pending[0]["id"], reviewer="sk", expected=0)
    store.reject_proposal(pending[1]["id"], reviewer="sk")
    assert len(store.get_proposals("s", status="pending")) == 3

    cases = store.confirmed_cases("s")
    assert len(cases) == 1
    assert cases[0].reviewer == "sk" and cases[0].expected == 0 and cases[0].gateable


def test_a_review_without_a_reviewer_is_refused(store, traces):
    store.put_traces(traces)
    store.put_proposals("s", [{"trace_id": traces[0].trace_id, "suggested_expected": 1}])
    pid = store.get_proposals("s")[0]["id"]
    with pytest.raises(ValueError, match="who did it"):
        store.confirm_proposal(pid, reviewer="", expected=1)


def test_records_and_online_scores(store):
    store.put_record("coverage", {"coverage": 0.42}, label="s")
    assert store.get_records("coverage")[0]["payload"]["coverage"] == 0.42
    for i in range(20):
        store.put_online_score("bow", i % 2, 40.0 + i)
    stats = store.online_score_stats()
    assert stats["n"] == 20 and stats["avg_us"] > 40 and stats["p95_us"] >= stats["avg_us"]


def test_open_store_rejects_an_unknown_url():
    with pytest.raises(ValueError, match="unrecognised store URL"):
        open_store("mongodb://nope")


def test_a_store_is_a_usable_trace_source(store, traces, tmp_path):
    from livingeval.sources import load_traces

    path = tmp_path / "t.db"
    s = SQLiteStore(path)
    s.put_traces(traces)
    s.close()
    assert len(load_traces(f"sqlite:///{path}")) == len(traces)


# ---------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------


def test_the_hashing_embedder_is_deterministic_across_processes():
    """Python's built-in `hash()` is salted per process. If this ever regresses,
    coverage numbers become irreproducible between runs, silently."""
    script = (
        "import json;from livingeval.embed import HashingEmbedder;"
        "print(json.dumps(HashingEmbedder(dim=16)(['refund my invoice'])[0].tolist()))"
    )
    a = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    b = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert json.loads(a.stdout) == json.loads(b.stdout)


def test_hashing_embeddings_are_unit_norm_and_shaped():
    e = embed.hashing(dim=64, cache=False)
    v = e(["a refund please", "where is my parcel", ""])
    assert v.shape == (3, 64)
    assert np.allclose(np.linalg.norm(v[:2], axis=1), 1.0)


def test_similar_text_is_closer_than_unrelated_text():
    from livingeval.mine.space import Space

    e = embed.hashing(dim=512, cache=False)
    v = e([
        "my invoice was charged twice for the annual renewal",
        "my invoice was charged twice for the seat upgrade",
        "the courier never arrived for order 4471",
    ])
    d = Space.cosine_distance(v, v)
    assert d[0, 1] < d[0, 2]


def test_the_embedding_cache_returns_identical_vectors(tmp_path):
    inner = embed.HashingEmbedder(dim=32)
    texts = ["one", "two", "three"]
    first = embed.CachedEmbedder(inner, directory=tmp_path)(texts)
    second = embed.CachedEmbedder(inner, directory=tmp_path)(texts)
    assert np.allclose(first, second)
    assert list(tmp_path.glob("embed-*.npz")), "nothing was written to the cache"


def test_the_cache_survives_a_partially_overlapping_batch(tmp_path):
    inner = embed.HashingEmbedder(dim=32)
    a = embed.CachedEmbedder(inner, directory=tmp_path)
    first = a(["one", "two"])
    mixed = a(["two", "three", "one"])
    assert np.allclose(mixed[0], first[1])
    assert np.allclose(mixed[2], first[0])


def test_a_corrupt_cache_is_a_performance_problem_not_a_correctness_one(tmp_path):
    inner = embed.HashingEmbedder(dim=32)
    e = embed.CachedEmbedder(inner, directory=tmp_path)
    e(["one"])
    e.path.write_bytes(b"not an npz file")
    again = embed.CachedEmbedder(inner, directory=tmp_path)
    assert again(["one"]).shape == (1, 32)


def test_resolve_understands_the_spec_strings():
    assert "hashing" in embed.resolve("hashing:64", cache=False).name
    with pytest.raises(ValueError, match="unknown embedder"):
        embed.resolve("wordvec:glove", cache=False)


def test_coverage_works_in_an_embedding_space_and_records_which_one(traces):
    suite = EvalSuite.from_traces(traces[:80], name="s")
    space = le.mine.Space.fit(traces, embed=embed.hashing(dim=128, cache=False), seed=0)
    assert space.uses_embeddings
    result = le.mine.coverage(suite, traces, space=space)
    assert 0.0 <= result.coverage <= 1.0
    assert result.as_dict()["uses_embeddings"] is True
    assert "hashing" in result.as_dict()["representation"]


def test_cluster_labels_survive_in_an_embedding_space(traces):
    """Geometry from the embedder, human-readable labels from tf-idf. Without this a
    blind-spot report reads "cluster 7", which nobody can act on."""
    space = le.mine.Space.fit(traces, embed=embed.hashing(dim=128, cache=False), seed=0)
    clustering = le.mine.cluster(traces, space=space, seed=0)
    terms = clustering.describe(traces)
    assert any(v.strip() for v in terms.values())


def test_a_suite_still_covers_itself_in_an_embedding_space(traces):
    suite = EvalSuite.from_traces(traces, name="self")
    space = le.mine.Space.fit(traces, embed=embed.hashing(dim=128, cache=False), seed=0)
    assert le.mine.coverage(suite, traces, space=space).coverage == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# feedback: promotion and export
# ---------------------------------------------------------------------------


def test_promote_only_moves_reviewed_cases(store, traces):
    store.put_traces(traces)
    suite = EvalSuite.from_traces(traces[:50], name="s")
    store.put_suite(suite)
    store.put_proposals("s", [
        {"trace_id": t.trace_id, "cluster": 0, "reason": "r", "suggested_expected": 1}
        for t in traces[50:60]
    ])
    queued = store.get_proposals("s")
    for p in queued[:4]:
        store.confirm_proposal(p["id"], reviewer="sk", expected=1)

    result = feedback.promote(store, "s")
    assert result.added == 4
    assert result.skipped_unreviewed == 6
    assert result.reviewers == {"sk": 4}
    assert len(result.suite) == 54
    assert all(c.gateable for c in result.suite)
    assert store.get_suite("s").content_hash() == result.suite.content_hash()


def test_promote_retires_oldest_and_reports_it(store, traces):
    store.put_traces(traces)
    store.put_suite(EvalSuite.from_traces(traces[:50], name="s"))
    store.put_proposals("s", [
        {"trace_id": t.trace_id, "suggested_expected": 1} for t in traces[50:70]
    ])
    for p in store.get_proposals("s"):
        store.confirm_proposal(p["id"], reviewer="sk", expected=1)
    result = feedback.promote(store, "s", max_cases=55)
    assert len(result.suite) == 55
    assert result.retired == 15


def test_promote_on_a_missing_suite_raises(store):
    with pytest.raises(KeyError, match="no suite named"):
        feedback.promote(store, "nope")


@pytest.mark.parametrize("fmt", ["chat", "prompt_completion", "classification"])
def test_export_writes_every_format(tmp_path, traces, fmt):
    suite = EvalSuite.from_traces(traces[:60], name="s")
    info = feedback.export_finetune_data(suite, tmp_path / "d.jsonl", fmt=fmt)
    rows = [json.loads(line) for line in (tmp_path / "d.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 60 == info["n_train"]
    if fmt == "chat":
        assert rows[0]["messages"][-1]["content"] in ("PASS", "FAIL")
    elif fmt == "prompt_completion":
        assert rows[0]["completion"].strip() in ("PASS", "FAIL")
    else:
        assert rows[0]["label"] in (0, 1)


def test_export_drops_unconfirmed_cases_by_default(tmp_path, traces):
    """Training a scorer on judge labels and then scoring it against judge labels is a
    closed loop that reports success regardless of the judge."""
    suite = EvalSuite.from_traces(traces[:40], name="s")
    mined = [EvalCase(f"m{i}", t, expected=1, provenance="mined")
             for i, t in enumerate(traces[40:60])]
    polluted = suite.extend(mined)

    strict = feedback.export_finetune_data(polluted, tmp_path / "a.jsonl")
    loose = feedback.export_finetune_data(polluted, tmp_path / "b.jsonl", confirmed_only=False)
    assert strict["n_train"] == 40 and strict["n_dropped_unconfirmed"] == 20
    assert loose["n_train"] == 60 and loose["n_dropped_unconfirmed"] == 0


def test_the_export_holdout_is_split_by_session(tmp_path, traces):
    suite = EvalSuite.from_traces(traces[:120], name="s")
    info = feedback.export_finetune_data(suite, tmp_path / "d.jsonl", split=0.25,
                                         fmt="classification")
    assert info["n_eval"] > 0 and info["n_train"] > 0
    assert info["n_train"] + info["n_eval"] == 120

    train = {json.loads(x)["text"] for x in
             (tmp_path / "d.jsonl").read_text(encoding="utf-8").splitlines()}
    held = {json.loads(x)["text"] for x in
            (tmp_path / "d.eval.jsonl").read_text(encoding="utf-8").splitlines()}
    assert not (train & held), "a session appeared on both sides of the split"


def test_export_rejects_an_unknown_format(tmp_path, traces):
    with pytest.raises(ValueError, match="unknown format"):
        feedback.export_finetune_data(EvalSuite.from_traces(traces[:5]), tmp_path / "x", fmt="csv")


# ---------------------------------------------------------------------------
# spec resolvers
# ---------------------------------------------------------------------------


def test_judge_specs_resolve():
    from livingeval.judge import from_spec

    assert from_spec("oracle").noise == 0.0
    assert from_spec("oracle:0.1").noise == pytest.approx(0.1)
    assert "keyword" in from_spec("keyword:unable to verify").name
    with pytest.raises(ValueError, match="unrecognised judge spec"):
        from_spec("gpt5:please")
    with pytest.raises(ValueError, match="rule judges look like"):
        from_spec("rule:onlyone")


def test_trace_specs_resolve(tmp_path, traces):
    from livingeval.sources import load_traces

    assert len(load_traces("synthetic:drifting,n=120,seed=1")) == 120
    path = tmp_path / "t.jsonl"
    traces[:30].to_jsonl(path)
    assert len(load_traces(str(path))) == 30
    with pytest.raises(ValueError, match="do not know how to read"):
        load_traces("traces.parquet")
    with pytest.raises(ValueError, match="no synthetic generator"):
        load_traces("synthetic:nonexistent")
