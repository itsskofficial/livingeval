"""The five rungs, cheapest first.

Each rung is a complete scorer: `fit(texts, labels)` then `predict(texts)`. They are
ordered by how much structure they can express, and the ordering is the measurement -
the cheapest rung that reproduces a judge tells you what that judge's decision was a
function of.

    0 majority    no free parameters       constant prediction
    1 length      one threshold            a decision stump on character count
    2 keyword     one token + polarity     "does this phrase appear"
    3 bow         tf-idf word 1-2 grams    linear in lexical content
    4 charngram   tf-idf char 3-5 grams    linear in morphology and formatting

Two deliberate constraints:

- **Every rung is linear or simpler.** A gradient-boosted model would climb higher and
  tell you nothing, because "a strong model can predict the judge" is true of almost
  any judge and is not a statement about the judge's depth. The ladder's usefulness
  comes from its rungs being weak in *interpretable* ways.
- **Rung 4 is the only case-sensitive one.** Word-level tf-idf lowercases, which
  erases exactly the camelCase/snake_case distinction that separates rung 4 from
  rung 3. Keeping one rung case-sensitive is what makes "your judge is reading
  formatting" a hypothesis the ladder can actually test.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

__all__ = ["RUNGS", "RUNG_ORDER", "Rung", "make_rung"]


class Rung:
    """Base class: a fitted scorer plus honest timing."""

    name = "rung"
    n_params = 0
    #: Rungs 0-4 are diagnostics: weak in nameable ways, so a match says what the
    #: judge's decision was a function of. `finetune` sets this False, and the ladder
    #: attaches a note wherever its number appears - a strong fine-tune says a
    #: transformer can fit the judge, which is true of most judges and diagnoses
    #: nothing. See scorer/finetune.py.
    is_diagnostic = True

    def fit(self, texts: Sequence[str], labels: np.ndarray) -> Rung:
        raise NotImplementedError

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    # -- timing -----------------------------------------------------------------

    def time_single(self, texts: Sequence[str], n: int = 200) -> float:
        """Seconds per prediction when scoring **one item at a time**.

        This is the number that matters for per-turn online scoring and it is not the
        same as batch throughput - vectorising 2,000 documents at once hides the
        per-call overhead that dominates a single live turn. Both are recorded; this
        one is what the ladder reports.
        """
        sample = list(texts[: min(n, len(texts))]) or [""]
        t0 = time.perf_counter()
        for t in sample:
            self.predict([t])
        return (time.perf_counter() - t0) / len(sample)

    def time_batch(self, texts: Sequence[str]) -> float:
        """Seconds per prediction when scoring the whole set at once."""
        texts = list(texts) or [""]
        t0 = time.perf_counter()
        self.predict(texts)
        return (time.perf_counter() - t0) / len(texts)


class MajorityRung(Rung):
    """Rung 0. Predicts the training majority class.

    A judge that a constant reproduces is a judge whose labels barely vary, which
    means the eval is measuring nothing regardless of how good the number looks.
    """

    name = "majority"
    n_params = 0

    def fit(self, texts, labels):
        labels = np.asarray(labels, dtype=int)
        values, counts = np.unique(labels, return_counts=True)
        self.value_ = int(values[int(np.argmax(counts))])
        return self

    def predict(self, texts):
        return np.full(len(texts), self.value_, dtype=int)


class LengthRung(Rung):
    """Rung 1. One threshold on character count.

    Verbose-equals-good and terse-equals-refusal are both real judge failure modes,
    and both are one number.
    """

    name = "length"
    n_params = 1

    def fit(self, texts, labels):
        x = np.asarray([[len(t)] for t in texts], dtype=float)
        self.tree_ = DecisionTreeClassifier(max_depth=1, random_state=0).fit(x, np.asarray(labels, dtype=int))
        return self

    def predict(self, texts):
        x = np.asarray([[len(t)] for t in texts], dtype=float)
        return self.tree_.predict(x).astype(int)


class KeywordRung(Rung):
    """Rung 2. The single most predictive token, plus a polarity.

    Fit exhaustively rather than heuristically: for every token in the training
    vocabulary, score both rules ("present implies pass" and "present implies fail")
    by accuracy, and keep the best. Two sparse matrix-vector products cover the whole
    vocabulary, so exhaustive is also fast.
    """

    name = "keyword"
    n_params = 2

    def __init__(self, min_df: int = 2, ngram_range: tuple[int, int] = (1, 3)):
        self.min_df = min_df
        self.ngram_range = ngram_range

    def fit(self, texts, labels):
        y = np.asarray(labels, dtype=int)
        n = len(y)
        self.fallback_ = int(np.bincount(y, minlength=2).argmax())
        try:
            vec = CountVectorizer(binary=True, min_df=self.min_df, ngram_range=self.ngram_range)
            x = vec.fit_transform(list(texts))
        except ValueError:  # empty vocabulary
            self.token_ = None
            return self
        if x.shape[1] == 0:
            self.token_ = None
            return self

        n1 = int(y.sum())
        n0 = n - n1
        present_1 = np.asarray(x.T @ y).ravel().astype(float)
        present_total = np.asarray(x.sum(axis=0)).ravel().astype(float)
        present_0 = present_total - present_1
        absent_1 = n1 - present_1
        absent_0 = n0 - present_0

        acc_pos = (present_1 + absent_0) / n  # present -> 1
        acc_neg = (present_0 + absent_1) / n  # present -> 0
        best_pos, best_neg = int(np.argmax(acc_pos)), int(np.argmax(acc_neg))

        if acc_pos[best_pos] >= acc_neg[best_neg]:
            idx, self.polarity_ = best_pos, 1
        else:
            idx, self.polarity_ = best_neg, 0

        self.token_ = vec.get_feature_names_out()[idx]
        self.analyzer_ = vec.build_analyzer()
        return self

    def predict(self, texts):
        if self.token_ is None:
            return np.full(len(texts), self.fallback_, dtype=int)
        out = np.empty(len(texts), dtype=int)
        for i, t in enumerate(texts):
            present = self.token_ in self.analyzer_(t)
            out[i] = self.polarity_ if present else 1 - self.polarity_
        return out

    def describe(self) -> str:
        if self.token_ is None:
            return "no usable token"
        verdict = "pass" if self.polarity_ == 1 else "fail"
        return f'"{self.token_}" present -> {verdict}'


class BowRung(Rung):
    """Rung 3. tf-idf over word 1-2 grams into logistic regression.

    Reproducing a judge here means its decision is linearly separable in lexical
    content on this traffic. That is the most common place for an LLM judge to land,
    and it is the finding worth publishing.
    """

    name = "bow"
    n_params = -1  # vocabulary-sized; reported as "many"

    def __init__(self, max_features: int = 20_000, C: float = 1.0):
        self.max_features = max_features
        self.C = C

    def _pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=self.max_features,
                                        sublinear_tf=True)),
                ("clf", LogisticRegression(C=self.C, max_iter=2000, class_weight="balanced")),
            ]
        )

    def fit(self, texts, labels):
        y = np.asarray(labels, dtype=int)
        self.fallback_ = int(np.bincount(y, minlength=2).argmax())
        self.pipe_ = self._pipeline()
        try:
            self.pipe_.fit(list(texts), y)
        except ValueError:
            self.pipe_ = None
        return self

    def predict(self, texts):
        if self.pipe_ is None:
            return np.full(len(texts), self.fallback_, dtype=int)
        return self.pipe_.predict(list(texts)).astype(int)

    def top_features(self, k: int = 8) -> list[tuple[str, float]]:
        if self.pipe_ is None:
            return []
        names = self.pipe_.named_steps["vec"].get_feature_names_out()
        coef = self.pipe_.named_steps["clf"].coef_.ravel()
        order = np.argsort(np.abs(coef))[::-1][:k]
        return [(str(names[i]), float(coef[i])) for i in order]


class CharNgramRung(BowRung):
    """Rung 4. tf-idf over **case-sensitive** character 3-5 grams.

    The only case-sensitive rung, and the only one that sees inside a token. It picks
    up casing conventions, punctuation habits, markdown scaffolding and identifier
    morphology - the things a judge reads without anyone intending it to.
    """

    name = "charngram"

    def _pipeline(self) -> Pipeline:
        return Pipeline(
            [
                # (2, 5) rather than (3, 5): the lowercase-to-uppercase transition
                # that marks a casing convention is a *bigram*, and at 3-grams it is
                # diluted across every pair of surrounding characters until min_df
                # discards it.
                ("vec", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                        lowercase=False, max_features=self.max_features,
                                        sublinear_tf=True)),
                ("clf", LogisticRegression(C=self.C, max_iter=2000, class_weight="balanced")),
            ]
        )


RUNGS = {
    "majority": MajorityRung,
    "length": LengthRung,
    "keyword": KeywordRung,
    "bow": BowRung,
    "charngram": CharNgramRung,
}

#: Cheapest first. The ladder walks this order and stops reporting `judge_depth` at
#: the first rung that clears the bar.
RUNG_ORDER = ("majority", "length", "keyword", "bow", "charngram")

#: The full ladder including the fine-tuned rung. Opt in with
#: `ladder(..., finetune=True)` or by passing this as `rungs=`. Kept out of the
#: default because it needs `livingeval[finetune]` and a training run per CV fold,
#: and because it answers the deployment question rather than the diagnostic one.
RUNG_ORDER_WITH_FINETUNE = (*RUNG_ORDER, "finetune")


def make_rung(name: str, **kw) -> Rung:
    """Build a rung by name. `finetune` is resolved lazily so torch stays optional."""
    if name == "finetune":
        from livingeval.scorer.finetune import FinetunedRung

        return FinetunedRung(**kw)
    try:
        cls = RUNGS[name]
    except KeyError:
        raise ValueError(
            f"unknown rung {name!r}; choose from {list(RUNG_ORDER_WITH_FINETUNE)}"
        ) from None
    return cls(**kw)
