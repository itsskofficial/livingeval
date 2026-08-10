"""A dense embedding that needs no download, no GPU and no network.

Signed random projection of hashed word and character n-grams — the hashing trick with
a fixed seed, then L2 normalisation. Deterministic across machines and Python versions
because the hash is `blake2b`, not Python's salted `hash()`.

**What it is for.** It is the fixture that keeps the embedding abstraction honest: the
whole `Embedder` code path — resolution, batching, disk caching, `Space` integration,
coverage computed in a dense space — is exercised by the test suite on every commit,
with no model to download and no key to configure. Without it, every test touching
embeddings would be skipped in CI, which is the same as not having them.

**What it is not.** It is not competitive with a trained sentence encoder and makes no
claim to be. It has no notion of synonymy: "refund" and "reimbursement" are orthogonal
here and close together in a real encoder. For actual work use `hf:` or `openai:`.

Being explicit about that matters more than it might seem. A cheap default that quietly
underperforms is how a coverage number ends up wrong in a direction nobody checks — so
`name` says `hashing`, it lands in every record, and the docs say what it costs you.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

__all__ = ["HashingEmbedder"]


def _blake(token: str, seed: int) -> int:
    """A stable hash. Python's built-in `hash()` is salted per process, so using it
    would make embeddings — and therefore coverage — differ between runs."""
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=8, salt=seed.to_bytes(8, "little"))
    return int.from_bytes(h.digest(), "little")


class HashingEmbedder:
    """Signed random projection of hashed n-grams."""

    def __init__(
        self,
        dim: int = 512,
        word_ngrams: tuple[int, int] = (1, 2),
        char_ngrams: tuple[int, int] = (3, 4),
        seed: int = 0,
        lowercase: bool = True,
    ):
        self.dim = int(dim)
        self.word_ngrams = word_ngrams
        self.char_ngrams = char_ngrams
        self.seed = seed
        self.lowercase = lowercase
        self.name = (
            f"hashing(dim={dim},w={word_ngrams[0]}-{word_ngrams[1]},"
            f"c={char_ngrams[0]}-{char_ngrams[1]},seed={seed})"
        )

    def _tokens(self, text: str) -> list[str]:
        if self.lowercase:
            text = text.lower()
        words = [w for w in "".join(c if c.isalnum() else " " for c in text).split() if w]
        out: list[str] = []
        lo, hi = self.word_ngrams
        for n in range(lo, hi + 1):
            out.extend("_".join(words[i : i + n]) for i in range(max(0, len(words) - n + 1)))
        lo, hi = self.char_ngrams
        padded = f" {text} "
        for n in range(lo, hi + 1):
            out.extend(f"#{padded[i : i + n]}" for i in range(max(0, len(padded) - n + 1)))
        return out

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._tokens(text):
                h = _blake(token, self.seed)
                # Low bits pick the bucket, one further bit picks the sign - the
                # signed variant so that unrelated collisions cancel in expectation
                # rather than accumulating.
                idx = h % self.dim
                sign = 1.0 if (h >> 40) & 1 else -1.0
                out[row, idx] += sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)
