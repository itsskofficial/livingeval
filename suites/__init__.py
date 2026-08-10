"""Bundled scenarios.

Everything scenario-specific lives here and nothing in `src/livingeval/` may import
it - an `import-linter` contract in CI enforces that. The engine knows about traces,
judges, coverage and power; it does not know what a support queue is, and the moment
it does the library has become one person's project with an API on top.

Four scenarios, chosen so that their failure mechanisms sit at four different rungs
of the judge-complexity ladder. That is what makes them useful: the ladder can be
checked against known answers before anyone points it at a real judge.

| scenario         | quality question                            | designed rung |
|------------------|---------------------------------------------|---------------|
| `support_triage` | did the agent refuse instead of answering?  | keyword       |
| `rag_grounding`  | is the answer grounded or hedged?           | bow           |
| `code_patch`     | does the patch follow repo convention?      | charngram     |
| `tool_calling`   | does the claim match the tool result?       | none          |

They are **simulated**, and the README says so where the numbers are reported. The
audit's first two findings are about suite composition and gate statistics, which are
properties of the measuring instrument and for which simulation is the right tool.
The third uses them to validate the ladder against known depth, which is the
precondition for pointing it at a real judge and believing the answer.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SCENARIOS", "Scenario", "load"]


@dataclass
class Scenario:
    """A named traffic generator plus the judge that grades it."""

    name: str
    mechanism: str
    designed_depth: str | None
    question: str
    failure: str
    judge_noise: float = 0.0

    def traces(self, n: int = 1200, seed: int = 0, drift: bool = True, **kw):
        from livingeval.synthetic import DriftSchedule, TrafficSpec, generate

        return generate(
            TrafficSpec(
                n=n,
                mechanism=self.mechanism,
                seed=seed,
                judge_noise=self.judge_noise,
                drift=DriftSchedule() if drift else DriftSchedule.none(),
                **kw,
            )
        )

    def judge(self, noise: float | None = None, seed: int = 0):
        """The scenario's judge.

        Every bundled judge is an oracle over the generator's ground truth, optionally
        noisy. That is deliberate: to validate the ladder you need a judge whose
        depth you already know, and an LLM judge is precisely the thing whose depth
        you do not. Swap in `livingeval.judge.openai(...)` to run the same scenario
        against a real one.
        """
        from livingeval.judge import oracle

        return oracle(noise=self.judge_noise if noise is None else noise, seed=seed,
                      name=f"{self.name}-judge")

    def as_dict(self) -> dict:
        return {
            "scenario": self.name,
            "mechanism": self.mechanism,
            "designed_depth": self.designed_depth or "none",
            "question": self.question,
            "failure": self.failure,
        }


SCENARIOS: dict[str, Scenario] = {
    "support_triage": Scenario(
        name="support_triage",
        mechanism="shortcut",
        designed_depth="keyword",
        question="did the agent resolve the request, or refuse it?",
        failure="the agent emits one fixed inability phrase",
        judge_noise=0.03,
    ),
    "rag_grounding": Scenario(
        name="rag_grounding",
        mechanism="lexical",
        designed_depth="bow",
        question="is the answer grounded in the record, or hedged from memory?",
        failure="any of twelve unrelated hedges appears",
        judge_noise=0.03,
    ),
    "code_patch": Scenario(
        name="code_patch",
        mechanism="morphology",
        designed_depth="charngram",
        question="does the patch follow the repository's naming convention?",
        failure="identifiers are rewritten in camelCase in a snake_case repo",
        judge_noise=0.03,
    ),
    "tool_calling": Scenario(
        name="tool_calling",
        mechanism="deep",
        designed_depth=None,
        question="does the agent's claim match what the tool returned?",
        failure="the agent asserts the negation of the tool result",
        judge_noise=0.03,
    ),
}


def load(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError:
        raise ValueError(f"unknown scenario {name!r}; choose from {sorted(SCENARIOS)}") from None


def make_suite(scenario: Scenario, traces, n: int = 150, seed: int = 0, name: str | None = None):
    """A curated suite sampled from a slice of traffic."""
    from livingeval.suite import EvalSuite

    return EvalSuite.from_traces(traces.sample(n, seed=seed), name=name or f"{scenario.name}-suite")
