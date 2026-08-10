"""Resolve a judge from a string.

    judge.from_spec("oracle:0.05")
    judge.from_spec("rule:my_module:my_function")
    judge.from_spec("openai:gpt-4o-mini")
    judge.from_spec("anthropic:claude-sonnet-4-5")

This exists so that the CLI, the platform and a config file can all name a judge the
same way without any of them owning the parser. It used to live in `cli.py`, which
meant the web layer imported the CLI to reach it - an upward dependency that the
`import-linter` layers contract caught. Moving it here was the fix, and the contract
finding it is the reason the layers contract is worth having.

API backends are imported lazily, so `from_spec` itself never pulls a client library
into the process.
"""

from __future__ import annotations

import importlib

__all__ = ["from_spec"]


def from_spec(spec: str):
    """Build a judge from a spec string.

    | spec | judge |
    |---|---|
    | `oracle` / `oracle:0.05` | synthetic ground truth, optionally noisy (synthetic corpora only) |
    | `rule:<module>:<function>` | any callable taking a `Trace` |
    | `keyword:<phrase>` | fails a trace when the phrase appears - a straw man of known depth |
    | `ollama:<model>` | a local model via Ollama - no key, no bill |
    | `openai:<model>` | needs `livingeval[openai]` and `OPENAI_API_KEY` |
    | `anthropic:<model>` | needs `livingeval[anthropic]` and `ANTHROPIC_API_KEY` |
    """
    from livingeval.judge import anthropic, keyword_judge, openai, oracle, rule

    parts = spec.split(":")
    kind = parts[0]

    if kind == "rule":
        if len(parts) != 3:
            raise ValueError("rule judges look like rule:module.path:function_name")
        module = importlib.import_module(parts[1])
        try:
            fn = getattr(module, parts[2])
        except AttributeError:
            raise ValueError(f"{parts[1]} has no attribute {parts[2]!r}") from None
        return rule(fn)
    if kind == "oracle":
        return oracle(noise=float(parts[1]) if len(parts) > 1 else 0.0)
    if kind == "keyword":
        if len(parts) < 2:
            raise ValueError("keyword judges look like keyword:<phrase>")
        return keyword_judge(":".join(parts[1:]))
    if kind == "ollama":
        from livingeval.judge.ollama import OllamaJudge

        # Rejoin: Ollama tags contain a colon, so `ollama:qwen3:8b` splits into three.
        return OllamaJudge(":".join(parts[1:]) if len(parts) > 1 else None)
    if kind == "openai":
        return openai(parts[1] if len(parts) > 1 else "gpt-4o-mini")
    if kind == "anthropic":
        return anthropic(parts[1] if len(parts) > 1 else "claude-sonnet-4-5")

    raise ValueError(
        f"unrecognised judge spec {spec!r}; use oracle[:noise] | rule:mod:fn | "
        "keyword:<phrase> | ollama:<model> | openai:<model> | anthropic:<model>"
    )
