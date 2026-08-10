"""Embedding backends, and the cache that makes them affordable.

`mine.Space.fit(embed=...)` has always accepted any callable from a list of strings to
a matrix. What was missing were the backends and, more importantly, the discipline that
makes them safe to use:

**Every embedder is identified, and the identity goes into the record.** Coverage
computed in two embedding spaces is not comparable. A number that changes when someone
upgrades a model, with nothing in the output to say so, is worse than no number. So an
`Embedder` has a `name` that includes the model and the dimension, and
`Space.as_dict()` carries it.

**Every embedder is cached on disk, keyed by (embedder identity, text hash).** A
coverage run over 20k traces is 20k embedding calls. Uncached, that is a bill or a
GPU-minute every time you re-plot a figure, and a tool with that property gets run once.

**One backend needs no network, no download and no GPU**, so the code path is exercised
by the test suite on every commit rather than skipped. `embed.hashing()` is a genuine
dense embedding - signed random projection of character and word n-gram hashes - and it
is deterministic. It is not competitive with a trained sentence encoder and it is not
meant to be; it is the fixture that keeps the abstraction honest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["CachedEmbedder", "Embedder", "cache_dir", "resolve"]


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into vectors."""

    name: str
    dim: int

    def __call__(self, texts: Sequence[str]) -> np.ndarray: ...


def cache_dir() -> Path:
    """`$LIVINGEVAL_CACHE/embeddings`, else `~/.cache/livingeval/embeddings`."""
    import os

    base = os.environ.get("LIVINGEVAL_CACHE")
    root = Path(base) if base else Path.home() / ".cache" / "livingeval"
    return root / "embeddings"


def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


class CachedEmbedder:
    """Wraps an embedder with a content-addressed disk cache.

    Storage is one `.npz` per embedder identity: a key array and a matrix. Loading is
    a single read, so a warm cache costs about as much as `np.load`. Cold entries are
    computed in one batch call to the wrapped embedder, because every hosted embedding
    API charges per request as well as per token.
    """

    def __init__(self, inner: Embedder, directory: Path | str | None = None,
                 enabled: bool = True, batch_size: int = 256):
        self.inner = inner
        self.name = getattr(inner, "name", inner.__class__.__name__)
        self.dim = getattr(inner, "dim", 0)
        self.enabled = enabled
        self.batch_size = batch_size
        self.dir = Path(directory) if directory is not None else cache_dir()
        fingerprint = hashlib.sha256(self.name.encode("utf-8")).hexdigest()[:16]
        self.path = self.dir / f"embed-{fingerprint}.npz"
        self._keys: dict[str, int] = {}
        self._matrix: np.ndarray | None = None
        self._dirty = False
        if self.enabled:
            self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with np.load(self.path, allow_pickle=False) as data:
                keys = [str(k) for k in data["keys"]]
                self._matrix = data["matrix"]
            self._keys = {k: i for i, k in enumerate(keys)}
        except (OSError, ValueError, KeyError):
            # A corrupt cache is a performance problem, never a correctness one.
            self._keys, self._matrix = {}, None

    def flush(self) -> None:
        if not (self.enabled and self._dirty and self._matrix is not None):
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        keys = np.asarray(sorted(self._keys, key=lambda k: self._keys[k]))
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, keys=keys, matrix=self._matrix)
        tmp.replace(self.path)
        self._dirty = False

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, max(1, self.dim)), dtype=np.float32)

        keys = [_text_key(t) for t in texts]
        missing = [t for t, k in zip(texts, keys, strict=True) if k not in self._keys]
        missing_keys = [k for k in keys if k not in self._keys]
        # De-duplicate: production traffic repeats, and a batch of 5k traces routinely
        # contains only 3k distinct renderings.
        unique: dict[str, str] = {}
        for t, k in zip(missing, missing_keys, strict=True):
            unique.setdefault(k, t)

        if unique:
            todo_keys = list(unique)
            todo_texts = [unique[k] for k in todo_keys]
            chunks = [
                self.inner(todo_texts[i : i + self.batch_size])
                for i in range(0, len(todo_texts), self.batch_size)
            ]
            fresh = np.vstack([np.atleast_2d(np.asarray(c, dtype=np.float32)) for c in chunks])
            if self._matrix is None:
                self._matrix = fresh
                self._keys = {k: i for i, k in enumerate(todo_keys)}
            else:
                offset = self._matrix.shape[0]
                self._matrix = np.vstack([self._matrix, fresh])
                for i, k in enumerate(todo_keys):
                    self._keys[k] = offset + i
            self.dim = int(self._matrix.shape[1])
            self._dirty = True
            self.flush()

        assert self._matrix is not None
        return self._matrix[[self._keys[k] for k in keys]]

    def __enter__(self) -> CachedEmbedder:
        return self

    def __exit__(self, *exc) -> None:
        self.flush()


def resolve(spec: str | Embedder | Callable, cache: bool = True, **kw) -> CachedEmbedder:
    """Turn a spec string into a cached embedder.

    | spec | backend |
    |---|---|
    | `hashing` / `hashing:512` | offline signed random projection, no download |
    | `hf:<model>` | mean-pooled Hugging Face encoder, e.g. `hf:sentence-transformers/all-MiniLM-L6-v2` |
    | `st:<model>` | sentence-transformers, if installed |
    | `ollama:<model>` | a local embedder via Ollama, e.g. `ollama:nomic-embed-text` |
    | `openai:<model>` | OpenAI embeddings API, e.g. `openai:text-embedding-3-small` |
    """
    if callable(spec) and not isinstance(spec, str):
        return CachedEmbedder(spec, enabled=cache)  # type: ignore[arg-type]

    kind, _, rest = str(spec).partition(":")
    if kind == "hashing":
        from livingeval.embed.hashing import HashingEmbedder

        inner: Embedder = HashingEmbedder(dim=int(rest) if rest else 512, **kw)
    elif kind == "hf":
        from livingeval.embed.hf import HFEmbedder

        inner = HFEmbedder(rest or "sentence-transformers/all-MiniLM-L6-v2", **kw)
    elif kind == "st":
        from livingeval.embed.hf import SentenceTransformerEmbedder

        inner = SentenceTransformerEmbedder(rest or "all-MiniLM-L6-v2", **kw)
    elif kind == "ollama":
        from livingeval.embed.ollama import OllamaEmbedder

        inner = OllamaEmbedder(rest or "nomic-embed-text", **kw)
    elif kind == "openai":
        from livingeval.embed.api import OpenAIEmbedder

        inner = OpenAIEmbedder(rest or "text-embedding-3-small", **kw)
    else:
        raise ValueError(
            f"unknown embedder {spec!r}; use hashing | hf:<model> | st:<model> | "
            "ollama:<model> | openai:<model>"
        )
    return CachedEmbedder(inner, enabled=cache)


def describe(embedder) -> dict:
    """What goes into a result record so two coverage numbers can be compared."""
    return {
        "embedder": getattr(embedder, "name", str(embedder)),
        "dim": int(getattr(embedder, "dim", 0)),
    }


def fingerprint(embedder) -> str:
    return hashlib.sha256(json.dumps(describe(embedder), sort_keys=True).encode()).hexdigest()[:12]
