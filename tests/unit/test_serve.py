"""Layer A for the platform: the API contract, the review guard-rail, the hot path.

The tests worth reading are the ones about the guard-rail. Everything else here is
ordinary HTTP plumbing; `test_a_review_without_a_reviewer_is_rejected` and
`test_unreviewed_proposals_never_reach_the_gate` are the two that encode why the loop
is safe to run.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="needs livingeval[serve]")
pytest.importorskip("httpx", reason="TestClient needs httpx")

from fastapi.testclient import TestClient

import livingeval as le
from livingeval.serve import seed_demo
from livingeval.serve.app import AppState, create_app

pytestmark = pytest.mark.serve


@pytest.fixture(scope="module")
def state():
    st = AppState(db="sqlite://:memory:", suite_name="default", judge_spec="oracle:0.05")
    seed_demo(st.store, n=900, suite_size=120, seed=0)
    st.refit_scorer(n_boot=100)
    yield st
    st.close()


@pytest.fixture(scope="module")
def client(state):
    return TestClient(create_app(state))


# -- the hot path ------------------------------------------------------------


def test_scoring_a_turn_is_fast_and_reports_its_own_agreement(client):
    r = client.post("/v1/score", json={"text": "I am unable to verify that right now."})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in (0, 1)
    assert body["kappa_vs_judge"] is not None, "a score with no agreement number is noise"
    assert body["latency_us"] < 50_000, "the original target was 50 ms; this must be far under"


def test_scoring_accepts_turns_as_well_as_text(client):
    r = client.post("/v1/score", json={
        "turns": [{"role": "user", "content": "where is my parcel"},
                  {"role": "assistant", "content": "Checked and consistent across both systems."}],
        "store": False,
    })
    assert r.status_code == 200 and "label" in r.json()


def test_scoring_without_a_deployed_scorer_says_so():
    st = AppState(db="sqlite://:memory:")
    c = TestClient(create_app(st))
    r = c.post("/v1/score", json={"text": "hello"})
    assert r.status_code == 409
    assert "refit" in r.json()["detail"]
    st.close()


def test_a_malformed_score_request_is_rejected(client):
    assert client.post("/v1/score", json={"nope": 1}).status_code == 422


def test_online_latency_is_recorded_for_the_dashboard(client):
    for _ in range(5):
        client.post("/v1/score", json={"text": "a reply"})
    stats = client.get("/v1/scorer").json()
    assert stats["deployed"] is True
    assert stats["n"] >= 5 and stats["avg_us"] > 0


# -- the measurements over HTTP ---------------------------------------------


def test_coverage_endpoint(client):
    body = client.get("/v1/coverage").json()
    assert 0.0 <= body["coverage"] <= 1.0
    assert body["kind"] == "coverage"
    assert "sensitivity" in body


def test_blindspots_endpoint_ranks_and_carries_coverage(client):
    body = client.get("/v1/blindspots?top=4").json()
    spots = body["spots"]
    assert 1 <= len(spots) <= 4
    severities = [s["severity"] for s in spots]
    assert severities == sorted(severities, reverse=True)
    assert body["coverage"]["kind"] == "coverage"


def test_ladder_endpoint_reports_depth_and_a_reading(client):
    body = client.get("/v1/ladder?n_boot=100").json()
    assert {r["rung"] for r in body["rungs"]} == {
        "majority", "length", "keyword", "bow", "charngram"
    }
    assert body["reading"]


def test_power_endpoint_includes_the_false_alarm_rate(client):
    body = client.get("/v1/power?n_sim=40&top=3").json()
    assert 0.0 <= body["expected"] <= 1.0
    assert "false_alarm" in body, "power without its Type-I error is half a number"


def test_gate_endpoint_returns_a_three_valued_decision(client):
    body = client.get("/v1/gate").json()
    assert body["decision"] in ("PASS", "FAIL", "BLIND")
    assert body["exit_code"] == {"PASS": 0, "FAIL": 1, "BLIND": 2}[body["decision"]]


def test_judge_validation_endpoint(client):
    body = client.get("/v1/judge/validate?n_boot=100").json()
    assert body["status"] in ("VALIDATED", "UNDERPOWERED", "UNVALIDATED")


def test_status_reports_the_representation_in_use(client):
    body = client.get("/v1/status").json()
    assert body["store"]["traces"] > 0
    assert body["embedder"]
    assert body["suite"] == "default"


def test_a_missing_suite_is_a_404():
    st = AppState(db="sqlite://:memory:", suite_name="ghost")
    c = TestClient(create_app(st))
    st.store.put_traces(le.synthetic.stable(n=100, seed=0))
    assert c.get("/v1/coverage").status_code == 404
    st.close()


# -- ingestion --------------------------------------------------------------


def test_traces_can_be_posted_in_loose_shape():
    st = AppState(db="sqlite://:memory:")
    c = TestClient(create_app(st))
    r = c.post("/v1/traces", json={"traces": [{
        "id": "x1", "sessionId": "s1", "timestamp": "2026-03-01T09:00:00Z",
        "messages": [{"role": "human", "content": "hi"}, {"role": "ai", "content": "hello"}],
    }]})
    assert r.status_code == 200 and r.json()["ingested"] == 1
    assert c.get("/v1/traces?limit=1").json()["traces"][0]["session_id"] == "s1"
    st.close()


def test_an_empty_ingest_is_rejected(client):
    assert client.post("/v1/traces", json={"traces": []}).status_code == 422


# -- the review guard-rail --------------------------------------------------


def test_mining_queues_proposals_without_duplicating_them(client):
    first = client.post("/v1/mine?n=8", json={}).json()
    assert first["queued"] > 0
    second = client.post("/v1/mine?n=8", json={}).json()
    assert second["queued"] == 0, "re-mining must not refill the queue with the same traces"


def test_a_review_without_a_reviewer_is_rejected(client):
    pid = client.get("/v1/proposals").json()["proposals"][0]["id"]
    assert client.post(f"/v1/proposals/{pid}/confirm", json={"expected": 1}).status_code == 422
    assert client.post(f"/v1/proposals/{pid}/reject", json={}).status_code == 422
    assert client.post(f"/v1/proposals/{pid}/confirm",
                       json={"reviewer": "", "expected": 1}).status_code == 422


def test_confirming_records_the_reviewer_and_the_label(client):
    pid = client.get("/v1/proposals").json()["proposals"][0]["id"]
    body = client.post(f"/v1/proposals/{pid}/confirm",
                       json={"reviewer": "tester", "expected": 0}).json()
    assert body["status"] == "confirmed"
    assert body["reviewer"] == "tester"
    assert body["expected"] == 0


def test_unreviewed_proposals_never_reach_the_gate(client):
    """The whole reason the loop is safe: a mined case cannot block a deployment until
    a named human has looked at it."""
    before = client.get("/v1/coverage").json()["n_cases"]
    pending = client.get("/v1/proposals").json()["proposals"]
    assert pending, "expected leftovers in the queue"

    for p in pending[:3]:
        client.post(f"/v1/proposals/{p['id']}/confirm",
                    json={"reviewer": "tester", "expected": p["suggested"] or 0})

    # Counted from the store rather than assumed, so the test does not depend on what
    # any earlier test in this module happened to confirm.
    n_confirmed = len(client.get("/v1/proposals?status=confirmed").json()["proposals"])
    still_pending = len(client.get("/v1/proposals").json()["proposals"])

    promoted = client.post("/v1/promote", json={}).json()
    assert promoted["added"] == n_confirmed
    assert promoted["skipped_unreviewed"] == still_pending
    assert promoted["n_cases"] == before + n_confirmed
    assert set(promoted["reviewers"]) == {"tester"}
    assert sum(promoted["reviewers"].values()) == n_confirmed


def test_confirming_a_missing_proposal_is_a_404(client):
    r = client.post("/v1/proposals/999999/confirm", json={"reviewer": "x", "expected": 1})
    assert r.status_code == 404


def test_export_writes_confirmed_labels(client, tmp_path):
    r = client.post("/v1/export/finetune",
                    json={"path": str(tmp_path / "ft.jsonl"), "split": 0.2})
    assert r.status_code == 200
    body = r.json()
    assert body["n_train"] > 0 and body["confirmed_only"] is True
    assert (tmp_path / "ft.jsonl").exists()


# -- pages and records ------------------------------------------------------


@pytest.mark.parametrize("path", ["/", "/review", "/docs", "/openapi.json"])
def test_pages_render(client, path):
    assert client.get(path).status_code == 200


def test_the_dashboard_has_no_external_dependencies(client):
    """It has to work with no network. A CDN link would make the demo fail offline."""
    for path in ("/", "/review"):
        html = client.get(path).text
        assert "http://" not in html.replace("http://127.0.0.1", "")
        assert "https://" not in html
        assert "<script src" not in html


def test_every_measurement_is_persisted_as_a_record(client):
    client.get("/v1/coverage")
    kinds = {r["kind"] for r in client.get("/v1/records?limit=50").json()["records"]}
    assert "coverage" in kinds


def test_deploying_a_rung_the_ladder_did_not_clear_is_refused(state):
    with pytest.raises(ValueError, match="below the bar"):
        state.refit_scorer(rung="majority", n_boot=50)
