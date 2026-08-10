"""Local neural embeddings over Ollama, e.g. `nomic-embed-text`.

Slots into `mine.Space.fit(embed=...)` like every other backend, and its identity goes
into the result record - coverage computed in two embedding spaces is not comparable, so
the space has to be part of the number.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np

from livingeval import ollama as client

__all__ = ["OllamaEmbedder"]


class OllamaEmbedder:
    """A local embedding model served by Ollama."""

    def __init__(self, model: str | None = None, host: str | None = None, dim: int = 0):
        self.model = model or os.environ.get("LIVINGEVAL_EMBED_MODEL", "nomic-embed-text")
        self.host = host or client.default_host()
        self.dim = dim
        self.name = f"ollama:{self.model}"

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        vectors = client.embed(self.model, list(texts), host=self.host)
        self.dim = int(vectors.shape[1])
        return vectors
