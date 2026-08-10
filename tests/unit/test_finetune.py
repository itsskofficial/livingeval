"""Layer A for the fine-tuned rung.

Split in two on purpose.

The plumbing tests use a **stub backend** and run on every commit: they check that the
rung integrates into the ladder under the same group-aware CV as the others, that its
number carries the note saying it is a deployment figure rather than a diagnostic, and
that `judge_depth` prefers a cheap rung when one clears the bar.

The real training test is marked `download` and off by default, because it fetches a
checkpoint from the Hugging Face hub and trains once per CV fold. Off because slow, not
because optional — CI runs it on `workflow_dispatch`.
"""

from __future__ import annotations

import numpy as np
import pytest

import livingeval as le
from livingeval.scorer.finetune import FINETUNE_NOTE, FinetunedRung
from livingeval.scorer.rungs import RUNG_ORDER, RUNG_ORDER_WITH_FINETUNE, make_rung


class StubBackend:
    """A perfect learner with no dependencies: memorises training text, else majority.

    Stands in for a fine-tune so the wiring can be checked in milliseconds. It is
    deliberately a *memoriser*, which is the honest caricature of what the rung is
    being asked about — capacity without insight.
    """

    label = "stub"

    def fit(self, texts, labels):
        y = np.asarray(labels, dtype=int)
        self.seen = dict(zip(list(texts), y.tolist(), strict=True))
        self.fallback = int(np.bincount(y, minlength=2).argmax())
        return self

    def predict(self, texts):
        return np.asarray([self.seen.get(t, self.fallback) for t in texts], dtype=int)


# -- the rung in isolation ---------------------------------------------------


def test_the_rung_is_flagged_as_not_a_diagnostic():
    rung = FinetunedRung(backend=StubBackend())
    assert rung.is_diagnostic is False
    assert rung.name == "finetune"


def test_make_rung_resolves_finetune_and_rejects_nonsense():
    assert isinstance(make_rung("finetune", backend=StubBackend()), FinetunedRung)
    with pytest.raises(ValueError, match="unknown rung"):
        make_rung("gpt5")


def test_an_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="unknown finetune backend"):
        FinetunedRung(backend="magic")


def test_the_rung_survives_a_single_class_target():
    """A degenerate fold must not crash a cross-validated ladder."""
    rung = FinetunedRung(backend=StubBackend())
    rung.fit(["a", "b", "c"], np.ones(3, dtype=int))
    assert list(rung.predict(["a", "z"])) == [1, 1]


def test_the_default_ladder_does_not_include_it():
    """It needs an extra and trains per fold, so it must be opt-in."""
    assert "finetune" not in RUNG_ORDER
    assert RUNG_ORDER_WITH_FINETUNE[-1] == "finetune"


# -- the rung inside the ladder ---------------------------------------------


@pytest.fixture(scope="module")
def traces():
    return le.synthetic.deep(n=240, seed=1)


def test_finetune_true_appends_the_rung(traces):
    result = le.scorer.ladder(
        traces, le.judge.oracle(), n_boot=50, n_splits=3,
        finetune={"backend": StubBackend()},
    )
    assert [r.name for r in result.rungs] == list(RUNG_ORDER_WITH_FINETUNE)
    assert result.has_nondiagnostic_rung


def test_the_note_appears_wherever_the_number_does(traces):
    result = le.scorer.ladder(
        traces, le.judge.oracle(), n_boot=50, n_splits=3,
        finetune={"backend": StubBackend()},
    )
    assert FINETUNE_NOTE in result.table()
    payload = next(r for r in result.as_dict()["rungs"] if r["rung"] == "finetune")
    assert payload["is_diagnostic"] is False


def test_a_memoriser_does_not_beat_group_aware_cross_validation(traces):
    """The stub memorises its training set exactly. Under a group-aware split it sees
    none of the held-out text, so it must fall back to the majority class and score
    kappa ~0 — which is the split doing its job. If this ever passes with a high kappa,
    the CV has stopped being honest and every rung above it is inflated."""
    result = le.scorer.ladder(
        traces, le.judge.oracle(), n_boot=50, n_splits=3,
        finetune={"backend": StubBackend()},
    )
    stub = next(r for r in result.rungs if r.name == "finetune")
    assert abs(stub.kappa.point) < 0.2


def test_a_cheap_rung_still_wins_when_one_clears_the_bar():
    """`judge_depth` is the *cheapest* rung over the bar, so adding a strong rung 5
    must not change the recommendation on a shallow judge."""
    shortcut = le.synthetic.shortcut(n=240, seed=1)
    result = le.scorer.ladder(
        shortcut, le.judge.oracle(), n_boot=50, n_splits=3,
        finetune={"backend": StubBackend()},
    )
    assert result.depth == "keyword"


def test_the_reading_changes_when_only_the_finetuned_rung_clears(traces):
    """Constructed rather than trained: what matters is the wording the user sees."""
    result = le.scorer.ladder(traces, le.judge.oracle(), n_boot=50, n_splits=3,
                              finetune={"backend": StubBackend()})
    ft = next(r for r in result.rungs if r.name == "finetune")
    object.__setattr__(ft.kappa, "point", 0.93)
    assert result.depth == "finetune"
    reading = result.reading()
    assert "deployable" in reading
    assert "statement about your options" in reading


def test_the_reading_suggests_finetuning_when_nothing_cheap_works(traces):
    """The pointer only appears when the fine-tuned rung has not already been run."""
    result = le.scorer.ladder(traces, le.judge.oracle(), n_boot=50, n_splits=3)
    assert result.depth is None
    assert "finetune=True" in result.reading()


# -- the real thing ----------------------------------------------------------


@pytest.mark.download
@pytest.mark.slow
def test_a_real_finetune_learns_a_shallow_judge():
    """Trains a small encoder for real. Marked `download` because it fetches a
    checkpoint; run with `pytest -m download`."""
    traces = le.synthetic.shortcut(n=200, seed=1)
    result = le.scorer.ladder(
        traces, le.judge.oracle(), n_boot=50, n_splits=2,
        finetune={"backend": "local", "epochs": 3, "batch_size": 16},
    )
    ft = next(r for r in result.rungs if r.name == "finetune")
    assert ft.kappa.point > 0.8, f"the rung failed to learn a one-phrase rule:\n{result.table()}"
    assert ft.seconds_per_turn > 0
    assert "epochs" in ft.detail


@pytest.mark.download
@pytest.mark.slow
def test_a_real_finetune_is_far_slower_per_turn_than_a_bag_of_words():
    """The honest cost of the deployment option, and the reason the cheap rungs matter:
    a transformer is orders of magnitude slower per turn than tf-idf."""
    traces = le.synthetic.shortcut(n=200, seed=1)
    result = le.scorer.ladder(
        traces, le.judge.oracle(), n_boot=50, n_splits=2,
        finetune={"backend": "local", "epochs": 1, "batch_size": 16},
    )
    bow = next(r for r in result.rungs if r.name == "bow")
    ft = next(r for r in result.rungs if r.name == "finetune")
    assert ft.seconds_per_turn > 10 * bow.seconds_per_turn
