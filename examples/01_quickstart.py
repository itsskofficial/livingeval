"""livingeval quickstart - about 30 seconds, no network, no API key, no GPU.

    python examples/01_quickstart.py

Walks the four questions the library answers about an eval setup:

    1. does this suite still represent my traffic?
    2. which parts of my traffic can it not see at all?
    3. would it catch a regression there?
    4. is the judge behind all of it a measurement?
"""

from __future__ import annotations

import livingeval as le

print(__doc__)

# ---------------------------------------------------------------------------
# Setup. In your project these two lines are `le.ingest.jsonl("traces/*.jsonl")`
# and `le.judge.openai("gpt-4o-mini", prompt=MY_RUBRIC)`.
# ---------------------------------------------------------------------------

traces = le.synthetic.drifting(n=1600, seed=0)  # a new intent appears mid-stream
judge = le.judge.oracle(noise=0.05, seed=7)  # stands in for an LLM judge
windows = traces.sorted_by_time().windows(6)

# The suite was curated from the first window, six windows ago. Like yours.
suite = le.EvalSuite.from_traces(windows[0].sample(150, seed=0), name="golden-set")
recent = windows[-1]

print(f"\ntraffic  {traces}")
print(f"suite    {suite}")

# ---------------------------------------------------------------------------
# 1. The score, which is the only number most tools report.
# ---------------------------------------------------------------------------

print("\n" + "=" * 74)
run = le.run_suite(suite, judge, "today")
print(run.summary())
print(f"\n  {run.score:.2f}. Ship it? Read on.")

# ---------------------------------------------------------------------------
# 2. Coverage: is that score about the traffic you serve?
# ---------------------------------------------------------------------------

print("\n" + "=" * 74)
space = le.mine.Space.fit(recent, seed=0)
clustering = le.mine.cluster(recent, space=space, seed=0)
coverage = le.mine.coverage(suite, recent, space=space, clustering=clustering)
print(coverage.summary())

# ---------------------------------------------------------------------------
# 3. Blind spots: which traffic, specifically.
# ---------------------------------------------------------------------------

print("\n" + "=" * 74)
report = le.mine.blindspots(suite, recent, space=space, clustering=clustering, top=5)
print(report.summary())

# ---------------------------------------------------------------------------
# 4. Power: would the gate fire if that traffic broke?
# ---------------------------------------------------------------------------

print("\n" + "=" * 74)
regression = le.power.on_meta("is_new_intent", True, effect=0.5)
power = le.power.estimate(suite, run, regression, n_sim=300)
print(power.summary())

# ---------------------------------------------------------------------------
# 5. The gate now has three answers, not two.
# ---------------------------------------------------------------------------

print("\n" + "=" * 74)
decision = le.gate.evaluate(
    run.with_outcomes(run.outcomes, "current"),
    run,
    coverage=coverage.coverage,
    power=power.power.point,
)
print(decision.summary())
print(f"\n  CI would exit {decision.exit_code}. A plain threshold gate would have exited 0.")

# ---------------------------------------------------------------------------
# 6. Close the loop: mine the blind spots, confirm, re-measure.
# ---------------------------------------------------------------------------

print("\n" + "=" * 74)
proposals = le.mine.propose(recent, suite, n=50, judge=judge, seed=0)
print(proposals.summary())

grown = suite.extend(proposals.confirm(reviewer="quickstart"), name="golden-set+mined")
new_power = le.power.estimate(grown, le.run_suite(grown, judge), regression, n_sim=300)
print(f"\n  power before mining  {power.power}   ({power.n_affected_cases} affected cases)")
print(f"  power after mining   {new_power.power}   ({new_power.n_affected_cases} affected cases)")
print(
    "\n  Power is a function of how many *confirmed* cases land in the affected region,\n"
    "  so one round of review buys a bounded amount of it. Getting from 0.37 to 0.9 means\n"
    "  more review, and now you can say how much before spending it."
)

# ---------------------------------------------------------------------------
# 7. And the question underneath all of it: is the judge a measurement?
# ---------------------------------------------------------------------------

print("\n" + "=" * 74)
print(le.judge.validate(judge, traces, n_boot=400).summary())

print("\n" + "=" * 74)
ladder = le.scorer.ladder(traces, judge, n_boot=400)
print(ladder.summary())

le.report.save([coverage, report, power, ladder], "results/quickstart.json")
print("\nwrote results/quickstart.json - every figure redraws from that file alone.")
