"""Cluster production traffic into the units a blind spot is reported in.

`k` is chosen by silhouette over a candidate range rather than by the elbow, because
the elbow is read by eye and this has to run unattended in CI. The chosen `k`, the
silhouette at every candidate, and the seed all go into the record, so "why did the
cluster count change" has an answer.

Clusters are a reporting convenience, not a claim about the data's true structure.
The number that matters - coverage - is computed per trace against the suite, not per
cluster; clustering only decides how that number is summarised for a human.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from livingeval.mine.space import Space
from livingeval.trace.types import TraceSet

__all__ = ["Clustering", "cluster"]


@dataclass
class Clustering:
    """Cluster assignments over a trace set."""

    labels: np.ndarray
    k: int
    silhouette: float
    candidates: dict[int, float]
    space: Space
    seed: int
    #: Cluster centroids in the space's coordinates, row `i` for `sorted(sizes)[i]`.
    #: Kept on the clustering so that suite cases can be assigned to the *traffic's*
    #: partition rather than to one computed separately.
    centroids: np.ndarray | None = None

    @property
    def sizes(self) -> dict[int, int]:
        values, counts = np.unique(self.labels, return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts, strict=False)}

    def shares(self) -> dict[int, float]:
        n = len(self.labels)
        return {c: s / n for c, s in self.sizes.items()}

    def label_traces(self, traces: TraceSet, key: str = "cluster") -> TraceSet:
        """Write the cluster id into each trace's `meta`, so that
        `scorer.ladder(..., by="cluster")` can produce a per-cluster ladder."""
        for t, lab in zip(traces, self.labels, strict=False):
            t.meta[key] = int(lab)
        return traces

    def describe(self, traces: TraceSet, top_terms: int = 5) -> dict[int, str]:
        """A few high-weight terms per cluster, so a blind-spot report names
        something a human recognises rather than "cluster 7".

        Labels come from a tf-idf matrix even when the *geometry* came from an
        embedder — the space keeps one for exactly this purpose. Labels never feed a
        number; they only make the output readable.
        """
        x = self.space.term_matrix(traces)
        if x is None or self.space.vectorizer is None:
            return dict.fromkeys(self.sizes, "")
        names = self.space.vectorizer.get_feature_names_out()
        out: dict[int, str] = {}
        for c in sorted(self.sizes):
            rows = np.flatnonzero(self.labels == c)
            if rows.size == 0:
                out[c] = ""
                continue
            centroid = np.asarray(x[rows].mean(axis=0)).ravel()
            # Contrast against the corpus mean so that terms shared by every cluster
            # (greetings, role tags) do not describe all of them identically.
            overall = np.asarray(x.mean(axis=0)).ravel()
            order = np.argsort(centroid - overall)[::-1][:top_terms]
            out[c] = ", ".join(str(names[i]) for i in order)
        return out

    def as_dict(self) -> dict:
        return {
            "k": self.k,
            "silhouette": self.silhouette,
            "candidates": {str(k): v for k, v in self.candidates.items()},
            "sizes": {str(k): v for k, v in self.sizes.items()},
            "seed": self.seed,
            **self.space.as_dict(),
        }


def cluster(
    traces: TraceSet,
    space: Space | None = None,
    k: int | str = "auto",
    k_range: tuple[int, int] = (3, 12),
    seed: int = 0,
    view: str = "request",
) -> Clustering:
    """Cluster traces. `k="auto"` selects by silhouette over `k_range`."""
    space = space or Space.fit(traces, view=view, seed=seed)
    x = space.transform(traces)
    n = x.shape[0]
    if n < 4:
        centroid = x.mean(axis=0, keepdims=True) if n else np.zeros((1, x.shape[1]))
        return Clustering(np.zeros(n, dtype=int), 1, float("nan"), {}, space, seed, centroid)

    # Asking for more clusters than there are *distinct* points is not merely
    # noisy -- KMeans silently returns fewer, so the silhouette being compared
    # belongs to a different k than the one on the axis. Traffic with heavy
    # repetition (templated prompts, a handful of intents) hits this routinely.
    distinct = int(np.unique(np.round(x, 6), axis=0).shape[0])

    candidates: dict[int, float] = {}
    if k == "auto":
        lo, hi = k_range
        hi = int(min(hi, max(2, n // 10), distinct))
        lo = int(min(lo, hi))
        best_k, best_s = max(2, lo), -1.0
        for candidate in range(max(2, lo), hi + 1):
            labels = KMeans(n_clusters=candidate, n_init=10, random_state=seed).fit_predict(x)
            if len(np.unique(labels)) < 2:
                continue
            score = float(silhouette_score(x, labels, metric="cosine"))
            candidates[candidate] = score
            if score > best_s:
                best_k, best_s = candidate, score
        chosen = best_k
    else:
        chosen = int(k)

    km = KMeans(n_clusters=max(1, min(chosen, distinct)), n_init=10,
                random_state=seed).fit(x)
    labels = km.labels_.astype(int)
    sil = (
        float(silhouette_score(x, labels, metric="cosine"))
        if len(np.unique(labels)) > 1
        else float("nan")
    )
    centroids = np.stack([x[labels == c].mean(axis=0) for c in sorted(set(labels.tolist()))])
    return Clustering(labels, int(chosen), sil, candidates, space, seed, centroids)
