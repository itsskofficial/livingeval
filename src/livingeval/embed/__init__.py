"""Embedding backends for coverage, clustering and case proposal.

    from livingeval import embed, mine

    space = mine.Space.fit(traces, embed=embed.resolve("hf:sentence-transformers/all-MiniLM-L6-v2"))
    print(mine.coverage(suite, traces, space=space).summary())

Four backends behind one protocol:

| spec | what it is | needs |
|---|---|---|
| `hashing` | signed random projection of hashed n-grams | nothing |
| `hf:<model>` | mean-pooled Hugging Face encoder | `transformers`, `torch` |
| `st:<model>` | sentence-transformers, model's own pooling | `sentence-transformers` |
| `openai:<model>` | hosted embeddings | `openai`, a key |

All of them are disk-cached, so a corpus is encoded once. All of them put their
identity into the result record, because **coverage computed in two embedding spaces is
not comparable** and a number that moves when someone upgrades a model, with nothing in
the output to say so, is worse than no number.

`hashing` needs no download and is therefore the one the test suite uses. It is a real
dense embedding and a deliberately weak one — see `embed/hashing.py` for what it costs
you and why it exists anyway.
"""

from livingeval.embed.base import (
    CachedEmbedder,
    Embedder,
    cache_dir,
    describe,
    fingerprint,
    resolve,
)
from livingeval.embed.hashing import HashingEmbedder

__all__ = [
    "CachedEmbedder",
    "Embedder",
    "HashingEmbedder",
    "cache_dir",
    "describe",
    "fingerprint",
    "hashing",
    "hf",
    "openai",
    "resolve",
    "sentence_transformers",
]


def hashing(dim: int = 512, cache: bool = True, **kw) -> CachedEmbedder:
    """Offline dense embedding. No download, no GPU, deterministic."""
    return CachedEmbedder(HashingEmbedder(dim=dim, **kw), enabled=cache)


def hf(model: str = "sentence-transformers/all-MiniLM-L6-v2", cache: bool = True, **kw):
    """Mean-pooled Hugging Face encoder. Needs `livingeval[embed]`."""
    from livingeval.embed.hf import HFEmbedder

    return CachedEmbedder(HFEmbedder(model, **kw), enabled=cache)


def sentence_transformers(model: str = "all-MiniLM-L6-v2", cache: bool = True, **kw):
    """sentence-transformers, which applies the model's own pooling config."""
    from livingeval.embed.hf import SentenceTransformerEmbedder

    return CachedEmbedder(SentenceTransformerEmbedder(model, **kw), enabled=cache)


def openai(model: str = "text-embedding-3-small", cache: bool = True, **kw):
    """OpenAI embeddings. Needs `livingeval[openai]` and `OPENAI_API_KEY`."""
    from livingeval.embed.api import OpenAIEmbedder

    return CachedEmbedder(OpenAIEmbedder(model, **kw), enabled=cache)
