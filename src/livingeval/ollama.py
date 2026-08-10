"""Ollama: local models, over HTTP, with no Python dependency.

[Ollama](https://ollama.com) runs open-weight models on your own machine and exposes a
small HTTP API on `localhost:11434`. Everything here is `urllib` against that API — no
SDK, because the surface we need is two endpoints and an SDK would be a dependency
carrying a pin for no benefit.

Why this module matters to the project rather than being a convenience:

**It makes the whole loop runnable with no keys and no bills.** The agent, the judge and
the embeddings can all be local, so someone who clones this repository can reproduce
every number without an account. That is the difference between a project people can
check and a project people have to believe.

**It gives you two judges of genuinely different capability.** A local 8B judge and a
frontier judge disagree in interesting ways, and their agreement with each other is the
*ceiling* any cheap scorer should be measured against — you cannot expect a bag of words
to reproduce a judge better than another frontier model does.

This module is the **client only** — `chat`, `generate`, `embed` and a few helpers. It
imports nothing from livingeval, which is what lets it sit at the very bottom of the
layer stack. The pieces that plug into the library live where their siblings do:

- `livingeval.judge.ollama.OllamaJudge`     — beside the OpenAI judge
- `livingeval.embed.ollama.OllamaEmbedder`  — beside the OpenAI embedder

That split was not cosmetic. The first version put all three here, and the layers
contract failed immediately: `judge.spec` imports the judge, the judge imported
`judge.cache`, and `livingeval.judge -> livingeval.ollama -> livingeval.judge` is a
cycle. The checker found it before it could become a confusing import error.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = [
    "chat",
    "default_host",
    "embed",
    "generate",
    "is_running",
    "list_models",
    "require_model",
]

DEFAULT_TIMEOUT = 300.0


def default_host() -> str:
    return os.environ.get("LIVINGEVAL_OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def _post(path: str, payload: dict, host: str | None = None,
          timeout: float = DEFAULT_TIMEOUT) -> dict:
    url = f"{host or default_host()}{path}"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:400]
        raise RuntimeError(f"ollama {path} -> HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"cannot reach Ollama at {host or default_host()}: {e.reason}\n"
            "  Is it running?  `ollama serve`  (it usually starts with the desktop app)"
        ) from e


def is_running(host: str | None = None, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{host or default_host()}/api/tags", timeout=timeout):
            return True
    except OSError:
        return False


def list_models(host: str | None = None) -> list[str]:
    with urllib.request.urlopen(f"{host or default_host()}/api/tags", timeout=10) as resp:
        return sorted(m["name"] for m in json.loads(resp.read())["models"])


def require_model(model: str, host: str | None = None) -> None:
    """Fail early with the command to fix it, rather than mid-run with a 404.

    Ollama tags are `name:tag`; `qwen3:8b` and `qwen3` are different strings but the
    same model family, so the check is prefix-tolerant.
    """
    available = list_models(host)
    if model in available or any(m.split(":")[0] == model.split(":")[0] for m in available):
        return
    raise RuntimeError(
        f"Ollama does not have {model!r}.\n"
        f"  available: {available}\n"
        f"  fix:       ollama pull {model}"
    )


# ---------------------------------------------------------------------------
# raw calls
# ---------------------------------------------------------------------------


def chat(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.0,
    host: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    think: bool = False,
    **options,
) -> dict:
    """One chat completion, optionally with tool definitions.

    `think=False` matters for the reasoning models (Qwen3 among them): left on, they
    emit a long `<think>` block before answering. That is fine for a human and wrong
    here — it triples latency and puts reasoning text into the trace that the judge then
    grades as if it were the answer.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": {"temperature": temperature, **options},
    }
    if tools:
        payload["tools"] = tools
    return _post("/api/chat", payload, host, timeout)


def generate(model: str, prompt: str, temperature: float = 0.0,
             host: str | None = None, timeout: float = DEFAULT_TIMEOUT, **options) -> str:
    result = _post(
        "/api/generate",
        {"model": model, "prompt": prompt, "stream": False, "think": False,
         "options": {"temperature": temperature, **options}},
        host, timeout,
    )
    return result.get("response", "")


def embed(model: str, texts: Sequence[str], host: str | None = None,
          timeout: float = DEFAULT_TIMEOUT) -> np.ndarray:
    result = _post("/api/embed", {"model": model, "input": list(texts)}, host, timeout)
    vectors = np.asarray(result["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)
