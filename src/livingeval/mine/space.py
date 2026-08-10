"""The representation everything geometric is computed in.

Coverage, clustering and case proposal all need a notion of "these two traces are
similar". That notion is a choice, it changes every number downstream, and it is
therefore recorded in every result record rather than assumed.

**The default is tf-idf reduced by truncated SVD, not neural embeddings.** Three
reasons, in order of importance:

1. It is deterministic and needs no download, no GPU and no API key, so the coverage
   numbers in the test suite are checked on every commit rather than skipped in CI.
2. Coverage computed in two different embedding spaces is not comparable, and a default
   that silently changes when someone upgrades a model is a reproducibility trap. A
   vectoriser pinned to the corpus does not drift.
3. On agent traffic, topic separation is mostly lexical. The expensive representation
   buys less here than it does on the sentence-similarity benchmarks it was tuned for.

**Neural embeddings are a first-class option, not an afterthought.** Pass
`embed=livingeval.embed.hf("...")` (or `"openai:..."`, or any callable) and the geometry
comes from there instead. Whether it *changes* your conclusions is an empirical question
about your traffic, and one worth answering rather than assuming — `scripts/compare_spaces.py`
runs both and diffs the coverage and the blind-spot ranking.

**Term labels survive either way.** Even when an embedder supplies the geometry, a
tf-idf vectoriser is fitted alongside for the sole purpose of naming clusters. Without
it a blind-spot report reads "cluster 7", which is useless to a human; with it the same
row reads "wallet address, wrong chain, gas fee". The geometry and the label come from
different places on purpose, and the labels never feed a number.

**The space is fitted on the production traces, not on the suite.** Coverage asks
whether the suite represents the traffic, so the traffic defines the geometry. Fitting
on the union would let a suite improve its own coverage by adding cases in a corner of
the space nothing else occupies.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from livingeval.trace.render import render_view
from livingeval.trace.types import Trace

__all__ = ["Space"]


class Space:
    """A fitted text representation plus cosine geometry."""

    def __init__(
        self,
        vectorizer: TfidfVectorizer | None,
        svd: TruncatedSVD | None,
        view: str,
        embed: Callable[[list[str]], np.ndarray] | None = None,
        name: str = "tfidf+svd",
        dim: int = 0,
    ):
        self.vectorizer = vectorizer
        self.svd = svd
        self.view = view
        self.embed = embed
        self.name = name
        self.dim = dim

    # -- construction -----------------------------------------------------------

    @classmethod
    def fit(
        cls,
        traces: Iterable[Trace],
        view: str = "request",
        n_components: int = 64,
        min_df: int = 2,
        max_features: int = 30_000,
        seed: int = 0,
        embed: Callable[[list[str]], np.ndarray] | str | None = None,
        label_terms: bool = True,
    ) -> Space:
        """Fit a space on `traces`.

        `embed` takes an `Embedder`, any callable, or a spec string understood by
        `livingeval.embed.resolve` (`"hashing"`, `"hf:<model>"`, `"openai:<model>"`).
        """
        texts = [render_view(t, view) for t in traces]
        if not texts:
            raise ValueError("cannot fit a Space on zero traces")

        if embed is not None:
            if isinstance(embed, str):
                from livingeval.embed import resolve

                embed = resolve(embed)
            name = getattr(embed, "name", getattr(embed, "__name__", "custom-embed"))
            dim = int(getattr(embed, "dim", 0))
            # A tf-idf vectoriser for cluster labels only. It never enters the
            # geometry; see the module docstring.
            labeller = None
            if label_terms:
                try:
                    labeller = TfidfVectorizer(
                        ngram_range=(1, 2), min_df=min_df, max_features=max_features,
                        sublinear_tf=True,
                    ).fit(texts)
                except ValueError:
                    labeller = None
            return cls(labeller, None, view, embed, name=f"embed:{name}", dim=dim)

        vec = TfidfVectorizer(
            ngram_range=(1, 2), min_df=min_df, max_features=max_features, sublinear_tf=True
        )
        x = vec.fit_transform(texts)
        k = int(min(n_components, max(2, min(x.shape) - 1)))
        svd = TruncatedSVD(n_components=k, random_state=seed).fit(x)
        return cls(vec, svd, view, None, name=f"tfidf+svd({k})", dim=k)

    # -- use --------------------------------------------------------------------

    @property
    def uses_embeddings(self) -> bool:
        return self.embed is not None

    def transform(self, traces: Iterable[Trace]) -> np.ndarray:
        """L2-normalised coordinates, so a dot product is a cosine similarity."""
        texts = [render_view(t, self.view) for t in traces]
        if not texts:
            return np.zeros((0, max(1, self.dim or 1)), dtype=float)
        if self.embed is not None:
            x = np.asarray(self.embed(texts), dtype=float)
        else:
            x = self.svd.transform(self.vectorizer.transform(texts))  # type: ignore[union-attr]
        return normalize(np.atleast_2d(x))

    def term_matrix(self, traces: Iterable[Trace]):
        """The tf-idf matrix used for cluster *labels*. `None` when unavailable."""
        if self.vectorizer is None:
            return None
        return self.vectorizer.transform([render_view(t, self.view) for t in traces])

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Pairwise cosine distance between two sets of normalised rows.

        Clipped into [0, 2]: SVD coordinates can be negative, so the similarity is
        genuinely signed, and floating point occasionally pushes it a hair past 1.
        """
        if a.size == 0 or b.size == 0:
            return np.zeros((a.shape[0], b.shape[0]), dtype=float)
        return np.clip(1.0 - a @ b.T, 0.0, 2.0)

    def as_dict(self) -> dict:
        return {
            "representation": self.name,
            "view": self.view,
            "n_components": int(self.svd.n_components) if self.svd is not None else None,
            "embedding_dim": self.dim or None,
            "uses_embeddings": self.uses_embeddings,
        }
