"""LLM-as-judge over the OpenAI and Anthropic APIs.

Both clients are imported lazily, so neither is a dependency of the default install
and neither is touched by the test suite. Keys come from the environment only.

Three things this does that a fifteen-line judge usually does not:

- **Caches on the rubric, not just the trace.** Editing one word of the prompt
  changes the judge; serving the previous verdicts would hide that.
- **Reports cost from the API's own token counts**, so the ladder's cost column is a
  measurement. Prices come from a table you can override, because published prices
  move faster than any library's release cycle - and a stale price is reported as
  `cost_usd=0.0` with `price_unknown` in the meta rather than as a wrong number.
- **Fails loudly on an unparseable response.** `strict=True` (the default) raises
  rather than defaulting to "pass", because a judge that silently returns the
  majority class when the API hiccups produces a suite score that drifts upward for
  reasons nobody can trace.
"""

from __future__ import annotations

import json
import os
import re
import time

from livingeval.judge.base import Verdict
from livingeval.judge.cache import VerdictCache, judge_fingerprint
from livingeval.trace.render import render_view
from livingeval.trace.types import Trace

__all__ = ["DEFAULT_RUBRIC", "PRICES", "APIJudge", "anthropic", "openai"]

DEFAULT_RUBRIC = """You are grading a single interaction between a user and an AI agent.

Decide whether the agent did the right thing.

Answer with a JSON object and nothing else:
{"pass": true|false, "reason": "<one short sentence>"}

Interaction:
---
{trace}
---"""

#: USD per 1M tokens, (input, output). Override per call with `prices=`; anything
#: not listed reports cost 0.0 and flags `price_unknown` rather than guessing.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "claude-opus-4-1": (15.00, 75.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

_JSON = re.compile(r"\{.*\}", re.S)


def _parse(text: str, strict: bool) -> tuple[int, str | None]:
    match = _JSON.search(text or "")
    if match:
        try:
            payload = json.loads(match.group(0))
            value = payload.get("pass", payload.get("passed", payload.get("label")))
            if value is not None:
                if isinstance(value, str):
                    value = value.strip().lower() in ("true", "yes", "pass", "1")
                return int(bool(value)), payload.get("reason")
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    lowered = (text or "").strip().lower()
    if lowered.startswith(("true", "yes", "pass")):
        return 1, None
    if lowered.startswith(("false", "no", "fail")):
        return 0, None
    if strict:
        raise ValueError(f"could not parse a verdict from judge output: {text[:200]!r}")
    return 0, "unparseable"


def _price(model: str, prices: dict, usage_in: int, usage_out: int) -> tuple[float, bool]:
    table = {**PRICES, **(prices or {})}
    if model not in table:
        return 0.0, False
    pin, pout = table[model]
    return (usage_in * pin + usage_out * pout) / 1_000_000.0, True


class APIJudge:
    """An LLM judge over a hosted API."""

    def __init__(
        self,
        model: str,
        provider: str,
        rubric: str = DEFAULT_RUBRIC,
        view: str = "full",
        temperature: float = 0.0,
        max_tokens: int = 200,
        cache: bool = True,
        cache_dir=None,
        prices: dict | None = None,
        strict: bool = True,
        name: str | None = None,
    ):
        if "{trace}" not in rubric:
            raise ValueError("rubric must contain a {trace} placeholder")
        self.model = model
        self.provider = provider
        self.rubric = rubric
        self.view = view
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prices = prices or {}
        self.strict = strict
        self.name = name or f"{provider}:{model}"
        self.fingerprint = judge_fingerprint(
            provider=provider, model=model, rubric=rubric, view=view, temperature=temperature
        )
        self.cache = VerdictCache(self.fingerprint, cache_dir, enabled=cache)
        self._client = None

    # -- providers --------------------------------------------------------------

    def _get_client(self):  # pragma: no cover - network
        if self._client is not None:
            return self._client
        if self.provider == "openai":
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError("pip install 'livingeval[openai]'") from e
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("set OPENAI_API_KEY in the environment")
            self._client = OpenAI()
        elif self.provider == "anthropic":
            try:
                from anthropic import Anthropic
            except ImportError as e:
                raise ImportError("pip install 'livingeval[anthropic]'") from e
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("set ANTHROPIC_API_KEY in the environment")
            self._client = Anthropic()
        else:
            raise ValueError(f"unknown provider {self.provider!r}")
        return self._client

    def _call(self, prompt: str) -> tuple[str, int, int]:  # pragma: no cover - network
        client = self._get_client()
        if self.provider == "openai":
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            usage = resp.usage
            return resp.choices[0].message.content or "", usage.prompt_tokens, usage.completion_tokens
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        return text, resp.usage.input_tokens, resp.usage.output_tokens

    # -- judge protocol ---------------------------------------------------------

    def __call__(self, trace: Trace) -> Verdict:
        key = trace.content_hash()
        hit = self.cache.get(key)
        if hit is not None:
            return Verdict(
                label=int(hit["label"]),
                score=hit.get("score"),
                rationale=hit.get("rationale"),
                cost_usd=0.0,
                latency_s=0.0,
                cached=True,
            )

        prompt = self.rubric.replace("{trace}", render_view(trace, self.view))
        t0 = time.perf_counter()
        text, tin, tout = self._call(prompt)
        elapsed = time.perf_counter() - t0
        label, reason = _parse(text, self.strict)
        cost, known = _price(self.model, self.prices, tin, tout)

        verdict = Verdict(
            label=label,
            rationale=reason,
            cost_usd=cost,
            latency_s=elapsed,
            cached=False,
            meta={"tokens_in": tin, "tokens_out": tout, "price_unknown": not known},
        )
        self.cache.put(key, {"label": label, "rationale": reason})
        self.cache.flush()
        return verdict


def openai(model: str = "gpt-4o-mini", **kw) -> APIJudge:
    """An OpenAI judge. Needs `OPENAI_API_KEY` and `pip install 'livingeval[openai]'`."""
    return APIJudge(model=model, provider="openai", **kw)


def anthropic(model: str = "claude-sonnet-4-5", **kw) -> APIJudge:
    """An Anthropic judge. Needs `ANTHROPIC_API_KEY` and `pip install 'livingeval[anthropic]'`.

    Running two judges from different providers over the same traces and reporting
    their kappa is the cheapest honest check on either of them.
    """
    return APIJudge(model=model, provider="anthropic", **kw)
