"""Judges, their cost, and whether they are measurements."""

from livingeval.judge.base import Judge, JudgeRun, Verdict, run_judge
from livingeval.judge.cache import VerdictCache, default_cache_dir, judge_fingerprint
from livingeval.judge.rule import OracleJudge, RuleJudge, keyword_judge, oracle, rule
from livingeval.judge.spec import from_spec
from livingeval.judge.validate import JudgeValidation, validate


def openai(*args, **kw):
    """An OpenAI judge. Imported lazily so the default install stays offline."""
    from livingeval.judge.api import openai as _openai

    return _openai(*args, **kw)


def anthropic(*args, **kw):
    """An Anthropic judge. Imported lazily so the default install stays offline."""
    from livingeval.judge.api import anthropic as _anthropic

    return _anthropic(*args, **kw)


__all__ = [
    "Judge",
    "JudgeRun",
    "JudgeValidation",
    "OracleJudge",
    "RuleJudge",
    "Verdict",
    "VerdictCache",
    "anthropic",
    "default_cache_dir",
    "from_spec",
    "judge_fingerprint",
    "keyword_judge",
    "openai",
    "oracle",
    "rule",
    "run_judge",
    "validate",
]
