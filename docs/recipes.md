# Recipes

---

## Point it at traces you already have

```python
import livingeval as le

traces = le.ingest.jsonl("traces/2026-08/*.jsonl")   # native or loose JSONL
traces = le.ingest.otel("otel-export.json")          # OTel spans
traces = le.ingest.read_langfuse_json("export.json") # a Langfuse export
```

The loose JSONL reader auto-detects the usual key names (`id`/`trace_id`,
`messages`/`turns`, `timestamp`/`created_at`, `sessionId`/`thread_id`), because the
first thing anyone does with a new eval tool is point it at the log file they already
have, and failing at that step is where adoption is lost.

Two fields decide how much the library can tell you:

| field | if present | if absent |
|---|---|---|
| `ts` | staleness is a windowed quantity | drift analysis is unavailable |
| `session_id` | ladder CV is group-aware | every trace is its own group; the split degrades to random and the ladder over-reports depth |
| `label` | judges can be validated | every judge is `UNVALIDATED` |

## Build a suite from what you have

```python
suite = le.EvalSuite.from_traces(labelled_traces, name="support")   # uses trace.label
suite = le.EvalSuite.from_traces(traces, expected=[1, 0, 1, ...])   # or pass labels
suite.save("evals/support.json")
```

`from_traces` raises rather than guessing when a label is missing. A case with an
invented expectation is worse than no case.

## Write a judge

Any callable from a trace to a verdict:

```python
def routed_correctly(trace):
    answer = trace.last("assistant").content
    return "escalated to billing" in answer

judge = le.judge.rule(routed_correctly)
```

Or a hosted one:

```python
judge = le.judge.openai("gpt-4o-mini", prompt="""
Did the agent resolve the user's request?
Answer with JSON and nothing else: {"pass": true|false, "reason": "..."}

{trace}
""")
```

Verdicts are cached on `(model, temperature, rubric, view)` + trace content, so
iterating on the analysis costs nothing after the first pass. Editing the rubric
correctly invalidates the cache.

## Validate the judge before trusting anything downstream

```python
print(le.judge.validate(judge, traces.labelled()).summary())
```

If you have no human labels yet, label 200 traces. It is a day of work and it converts
every number downstream from an opinion into a measurement. Until then the result reads
`UNVALIDATED` and says so in every table.

## Run the ladder against a real judge

The first experiment worth running, and one line:

```python
ladder = le.scorer.ladder(traces, le.judge.openai("gpt-4o-mini"), cv="session")
print(ladder.summary())
print(le.scorer.ladder(traces, judge, by="cluster"))   # per-cluster, which is the honest version
```

Cost: one judge call per trace, cached. On 2,000 traces with `gpt-4o-mini` that is
single-digit dollars, once.

Get a ceiling on it by running two judges from different providers and taking their κ
against each other. No cheap scorer should be expected to reproduce a judge better than
a second frontier model does.

## Wire the gate into CI

```yaml
- name: Eval gate
  run: |
    livingeval gate \
      --suite evals/support.json \
      --judge openai:gpt-4o-mini \
      --baseline results/main.json \
      --traces 'traces/last-7-days/*.jsonl' \
      --out results/pr.json
    case $? in
      0) echo "PASS" ;;
      1) echo "FAIL - regression detected"; exit 1 ;;
      2) echo "BLIND - the suite can no longer justify a pass"; exit 0 ;;
    esac
```

Treating `BLIND` as non-blocking-but-visible is the usual first configuration: it opens
a ticket rather than stopping a deploy. Escalating it to blocking is a decision for
when the mining loop is running.

## Close the loop

```python
report    = le.mine.blindspots(suite, traces, top=5)
proposals = le.mine.propose(traces, suite, n=20, judge=judge)

print(proposals.summary())            # review these
confirmed = proposals.confirm(reviewer="sk", expected={"t01847": 0})  # correct as you go
suite     = suite.extend(confirmed)
suite.save("evals/support.json")
```

Unconfirmed proposals can be added directly (`proposals.cases()`) — they improve
coverage and stay out of every gate decision until someone signs off. That guard-rail
is the reason the loop cannot converge on "the model is correct because the cases came
from the model".

## Budget the review

Before spending a day labelling, ask what it buys:

```python
base = le.run_suite(suite, judge)
reg  = le.power.on_meta("intent", "the_new_thing", effect=0.4)
print(le.power.estimate(suite, base, reg).summary())     # power now
# ... add n confirmed cases ...
print(le.power.estimate(bigger, le.run_suite(bigger, judge), reg).summary())
```

## Calibrate `flake` to your own system

Run the same suite twice against an unchanged system:

```python
a = le.run_suite(suite, judge, "run-a")
b = le.run_suite(suite, judge, "run-b")
f = le.power.measure_flake(a, b)
print(f"per-case instability: {f:.3f}")

le.power.false_alarm_rate(suite, a, flake=f, mode="threshold", threshold=0.90)
```

This is what turns the false-alarm table from an illustration into a fact about your
pipeline.

## Regenerate figures without re-running anything

```bash
livingeval figures results/audit.json --out figs/
```

Analyses serialise to versioned JSON; plotting reads records. Redrawing the full figure
set takes a second and cannot change a number.
