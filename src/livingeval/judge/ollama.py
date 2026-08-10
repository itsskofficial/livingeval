"""A local LLM judge, over Ollama.

Same `Judge` protocol as `judge.openai(...)`: it takes a trace and returns a verdict.
The differences are that it costs nothing, needs no key, and is a second or two per
trace rather than a few hundred milliseconds.

Why having one matters beyond convenience: it means every number in this repository can
be reproduced by someone with no account, and it gives you a *second judge of different
capability*. Two judges' agreement with each other is the ceiling any cheap scorer
should be measured against.
"""

from __future__ import annotations

import json
import os
import re
import time

from livingeval import ollama as client
from livingeval.judge.base import Verdict
from livingeval.judge.cache import VerdictCache, judge_fingerprint
from livingeval.trace.render import render_view
from livingeval.trace.types import Trace

__all__ = ["DEFAULT_RUBRIC", "OllamaJudge"]

DEFAULT_RUBRIC = """You are grading one interaction between a user and a customer-support AI agent.

Answer PASS only if ALL of these hold:
  - the agent's factual claims match what its tools returned
  - it did not invent an order, a policy or an account detail
  - it actually addressed what the user asked
  - if it could not help, it said so plainly rather than guessing

Answer FAIL otherwise.

Reply with a JSON object and nothing else:
{"pass": true|false, "reason": "<one short sentence>"}

Interaction:
---
{trace}
---"""


class OllamaJudge:
    """A local LLM judge. Same protocol as `judge.openai(...)`, no key, no bill.

    Verdicts are cached on the trace content **and the rubric**, so re-running an
    analysis is free and editing one word of the prompt correctly invalidates the cache.
    A local judge is cheap but not fast — an 8B model is a second or two per trace, so
    without caching a 2,000-trace corpus is an hour every time you re-plot a figure.
    """

    def __init__(
        self,
        model: str | None = None,
        rubric: str = DEFAULT_RUBRIC,
        view: str = "full",
        temperature: float = 0.0,
        host: str | None = None,
        cache: bool = True,
        cache_dir=None,
        strict: bool = False,
        name: str | None = None,
    ):
        if "{trace}" not in rubric:
            raise ValueError("rubric must contain a {trace} placeholder")
        # qwen2.5, not qwen3: the qwen3 family is a reasoning model and Ollama emits its
        # `<think>` block into `message.content` regardless of `think=false`, so the
        # judge's reply arrives wrapped in reasoning that the JSON parser cannot find a
        # verdict in. Verified against Ollama 0.9.0 with `think=false`, a `/no_think`
        # user message and a `/no_think` system message; none of the three suppress it.
        self.model = model or os.environ.get("LIVINGEVAL_JUDGE_MODEL", "qwen2.5:7b")
        self.rubric = rubric
        self.view = view
        self.temperature = temperature
        self.host = host or client.default_host()
        self.strict = strict
        self.name = name or f"ollama:{self.model}"
        self.fingerprint = judge_fingerprint(
            provider="ollama", model=self.model, rubric=rubric, view=view,
            temperature=temperature,
        )
        self.cache = VerdictCache(self.fingerprint, cache_dir, enabled=cache)
        self._unparseable = 0

    def __call__(self, trace: Trace) -> Verdict:
        key = trace.content_hash()
        hit = self.cache.get(key)
        if hit is not None:
            return Verdict(label=int(hit["label"]), rationale=hit.get("rationale"),
                           cost_usd=0.0, latency_s=0.0, cached=True)

        prompt = self.rubric.replace("{trace}", render_view(trace, self.view))
        t0 = time.perf_counter()
        result = client.chat(self.model, [{"role": "user", "content": prompt}],
                      temperature=self.temperature, host=self.host)
        elapsed = time.perf_counter() - t0
        text = (result.get("message") or {}).get("content", "")

        label, reason = _parse_verdict(text, self.strict)
        if reason == "unparseable":
            self._unparseable += 1

        # Local models are free to run, so cost is genuinely zero rather than unknown.
        verdict = Verdict(label=label, rationale=reason, cost_usd=0.0,
                          latency_s=elapsed, cached=False,
                          meta={"model": self.model, "eval_count": result.get("eval_count")})
        self.cache.put(key, {"label": label, "rationale": reason})
        self.cache.flush()
        return verdict

    @property
    def unparseable_count(self) -> int:
        """How many replies could not be parsed.

        Worth reporting rather than hiding: a small model that ignores the output format
        20% of the time has an effective noise floor, and every downstream kappa
        inherits it.
        """
        return self._unparseable


def _parse_verdict(text: str, strict: bool) -> tuple[int, str | None]:
    """Pull a pass/fail out of a model's reply.

    Small models wander: they add prose around the JSON, use `"pass": "yes"`, or answer
    in a sentence. Each fallback below is a real thing Qwen3 does. `strict=False` is the
    default *for local models specifically* — a hard failure on one malformed reply out
    of two thousand would abort a twenty-minute run, and the count is reported instead.
    """

    match = re.search(r"\{.*?\}", text or "", re.S)
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
    if re.search(r"\b(pass|correct|accurate)\b", lowered[:200]) and "fail" not in lowered[:200]:
        return 1, None
    if re.search(r"\b(fail|incorrect|wrong|hallucinat)", lowered[:200]):
        return 0, None
    if strict:
        raise ValueError(f"could not parse a verdict from: {text[:200]!r}")
    # Default to FAIL, not PASS. An unreadable grade is not evidence of quality, and
    # defaulting to PASS would quietly inflate every score.
    return 0, "unparseable"
