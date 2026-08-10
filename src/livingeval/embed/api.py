"""Hosted embeddings.

One backend, OpenAI, because it is the one your credits cover and because a hosted
embedding API is a commodity — a second provider would be the same twenty lines.

Keys come from the environment only. Every call is disk-cached by the wrapper, so a
corpus costs money once. That is not a nicety: an uncached embedder turns "re-plot the
coverage figure" into a billing event, and a tool with that property gets run once and
abandoned.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np

__all__ = ["PRICES", "OpenAIEmbedder"]

#: USD per 1M tokens. Override with `prices=`; anything unlisted reports 0.0 rather
#: than a guess, same rule as the judges. See DECISIONS.md #14.
PRICES = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}


class OpenAIEmbedder:
    """OpenAI's embeddings endpoint."""

    def __init__(self, model: str = "text-embedding-3-small", dimensions: int | None = None,
                 prices: dict | None = None):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover - optional extra
            raise ImportError("pip install 'livingeval[openai]'") from e
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("set OPENAI_API_KEY in the environment")
        self._client = OpenAI()
        self.model = model
        self.dimensions = dimensions
        self.prices = {**PRICES, **(prices or {})}
        # `text-embedding-3-*` support truncation to a requested dimension, which
        # changes the space, so it belongs in the identity.
        self.dim = dimensions or (1536 if "small" in model or "ada" in model else 3072)
        self.name = f"openai:{model}(dim={self.dim})"
        self.total_tokens = 0
        self.total_cost_usd = 0.0

    def __call__(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - network
        kwargs: dict = {"model": self.model, "input": list(texts)}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        response = self._client.embeddings.create(**kwargs)
        vectors = np.asarray([d.embedding for d in response.data], dtype=np.float32)
        self.dim = int(vectors.shape[1])
        used = getattr(response.usage, "total_tokens", 0) or 0
        self.total_tokens += used
        self.total_cost_usd += used * self.prices.get(self.model, 0.0) / 1_000_000.0
        # OpenAI returns unit-norm vectors, but truncated `dimensions` output is not
        # renormalised, so do it here rather than assume.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)
