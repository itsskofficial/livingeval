# The livingeval guide

The long-form version of the README: why the three measurements exist, what the audit
found, and every part of the library in detail.

New here? [`concepts.md`](concepts.md) defines the vocabulary first.

---

## The problem

Every team evaluating an LLM agent has converged on the same recipe:

```python
suite = load_golden_set()                     # curated once, six months ago
score = mean(judge(case) for case in suite)   # GPT-4 as judge
if score < 0.90: fail_ci()                    # the gate
```

Three defects. The field knows about all three in the abstract and runs none of the
corresponding controls.

**The suite no longer describes the traffic.** Voker's *State of YC AI Agents 2026*
survey found "a super-majority of respondents said evals often under-deliver because
keeping them up to date becomes an impossible task." Everyone experiences this as a
maintenance chore. Nobody measures it as a quantity. **Coverage is computable and it
is not reported.**

**A score is not a decision.** `0.87` does not tell you whether to ship. The question
the gate is really asking is *would this suite fire if the regression I'm worried
about were present?* — and the answer depends on how many of the suite's cases live in
the affected subpopulation. A frozen suite has zero cases from last month's new user
intent, so a regression confined to it is not harder to see. It is **invisible**.

**The judge is unvalidated, and so is everything downstream of it.** Two questions
almost never get asked: does the judge agree with a human, chance-corrected? And —
the sharper one — could something far cheaper have produced the same labels?

`livingeval` makes those three the default return values, the way
[`probeit`](https://github.com/itsskofficial/probeit) makes selectivity the default
return value of a probe.

## Quickstart

No model, no download, no API key. This is the real output:

```python
import livingeval as le

traces = le.synthetic.drifting(n=1600)                  # a new intent appears mid-stream
judge  = le.judge.oracle(noise=0.05)                    # stands in for your LLM judge
w      = traces.sorted_by_time().windows(6)
suite  = le.EvalSuite.from_traces(w[0].sample(150))     # curated six windows ago. Like yours.

print(le.run_suite(suite, judge).summary())
```

```
suite run  golden-set[today]  judge=oracle(noise=0.05)
  score 0.9333 over 150 gateable case(s)
```

0.93. Ship it?

```python
print(le.mine.blindspots(suite, w[-1], top=5).summary())
```

```
blind spots  golden-set   (overall coverage 37.8%, threshold 30%)
cluster    traffic  covered  cases  severity  terms
4            9.7%     0.0%      0     0.097  chain for, the wrong, went to, wrong, wrong chain
1            9.4%     0.0%      0     0.094  reverted, reverted on, was reverted, on the, on
10          12.0%    37.5%     22     0.075  stopped, working after, working, stopped working, me
2            6.7%     0.0%      0     0.067  confirmation, pending, pending confirmation, is pending
0           12.4%    54.5%     25     0.056  second unit, the second, second, refused, was refused

  cluster 4 carries 9.7% of traffic with 0 suite case(s). A regression confined to it
  cannot move the suite score.
```

```python
regression = le.power.on_meta("is_new_intent", True, effect=0.5)
print(le.power.estimate(suite, run, regression).summary())
```

```
power (simulated)  golden-set  vs  is_new_intent=True(effect=0.5)
  0/150 gateable case(s) affected (0.0%)   gate mode=paired   300 simulations
  power  0.0100 [0.0034, 0.0290]
  NO case in this suite is affected by this regression. The number above is the gate's
  false-alarm rate, not its detection rate.
```

So the gate has three answers, not two:

```
gate  BLIND  (exit 2)   mode=paired
  score 0.9333  baseline 0.9333   gateable cases 150
  ! coverage 37.8% is below 70%: 62% of traffic is unrepresented by this suite
  ! simulated detection power 1.0% is below 80% for the regression tested
```

**CI exits 2. A threshold gate would have exited 0.** Full walkthrough:
[`examples/01_quickstart.py`](examples/01_quickstart.py).

---

## The staleness audit

Three findings, each reproducible with one command and 71 seconds of laptop CPU:

```bash
python scripts/run_audit.py        # writes results/audit.json + figs/
```

**The traffic here is simulated, and that is stated first rather than buried.**
Findings 1 and 2 are claims about *suite composition and gate statistics* — properties
of the measuring instrument, for which simulation is the correct tool, exactly as a
synthetic leaky dataset is the correct tool for demonstrating cross-validation leakage.
Finding 3 uses simulation to validate the ladder against judges of *known* depth,
which is the precondition for pointing it at a real judge and believing the answer.

### Finding 1 — a frozen suite's blind spot grows while its score does not move

A suite curated in week 0, held fixed, against traffic where a new intent appears at
week 2 and grows to a third of volume.

| suite | metric | w0 | w1 | w2 | w3 | w4 | w5 |
|---|---|---|---|---|---|---|---|
| `frozen` | coverage | 0.78 | 0.55 | 0.54 | 0.47 | 0.45 | **0.43** |
| `frozen` | power (averaged over clusters) | 0.64 | 0.69 | 0.60 | 0.74 | 0.74 | 0.76 |
| `frozen` | power in its **worst** cluster | 0.125 | 0.435 | 0.105 | 0.085 | 0.020 | **0.005** |
| `mined` | coverage | 0.78 | 0.58 | 0.58 | 0.55 | 0.52 | **0.61** |
| `mined` | power (averaged over clusters) | 0.65 | 0.77 | 0.57 | 0.67 | 0.77 | 0.72 |
| `mined` | power in its **worst** cluster | 0.150 | 0.335 | 0.050 | 0.075 | 0.285 | **0.335** |

Three things in that table are worth more than the headline.

**The average power never drops.** It sits between 0.60 and 0.76 the whole way while
the worst cluster falls to 0.005. Averaging detection power over traffic is exactly
the aggregation that hides a blind spot — the same failure, one level up, that the
library exists to point at. `livingeval` reports both and the gate reads the worst.

**Coverage moves first and costs nothing to compute.** It is falling by week 1, three
windows before the power collapse is unambiguous. If you instrument one number, that
is the one.

**The loop helps and does not solve it.** The mined arm runs the real thing — blind
spots, proposals from the least-covered clusters, a named reviewer confirming and
correcting the judge, a bounded rolling suite. At a budget of 40 confirmed cases per
window it recovers coverage from 0.43 to 0.61, not to 1.0. That is the honest number:
staying current has an ongoing review cost, and this measures it rather than promising
it away.

Said in one number, against a regression confined to the intent that appeared:

| suite | affected cases | detection power |
|---|---|---|
| `frozen` | 0 / 200 | **0.013** [0.005, 0.029] |
| `mined` | 49 / 200 | **0.963** [0.939, 0.977] |

Throughout all of that the frozen suite's own score went from 0.9550 to 0.9500. It is
the same cases, so of course it did. That is the problem: the only number anyone looks
at is the one number that cannot move when the traffic moves.

### Finding 2 — the gate everyone ships fires at random

Both arms are reruns of an **unchanged** system with 5% per-case instability. Every
fire in this table is a false alarm. The threshold is set two points under the score a
rerun is *expected* to produce.

| suite size | threshold | threshold gate fires | paired gate fires |
|---|---|---|---|
| 25 | 0.894 | **0.343** [0.306, 0.382] | 0.000 [0.000, 0.006] |
| 50 | 0.822 | **0.320** [0.284, 0.358] | 0.002 [0.000, 0.009] |
| 100 | 0.885 | **0.178** [0.150, 0.211] | 0.012 [0.006, 0.024] |
| 200 | 0.863 | **0.088** [0.068, 0.114] | 0.012 [0.006, 0.024] |
| 400 | 0.894 | **0.047** [0.032, 0.067] | 0.022 [0.013, 0.037] |
| 800 | 0.887 | **0.003** [0.001, 0.012] | 0.017 [0.009, 0.030] |

**A 50-case suite behind a fixed threshold fails CI once in three runs with nothing
wrong.** The paired test on the same data, the same noise and the same suite holds its
Type-I error at or under 2.2% throughout. The reason your team stopped trusting the
eval gate is arithmetic, not vibes.

A fixed threshold does eventually work — at 800 cases. Almost nobody maintains 800
hand-labelled cases, which is the point: the naive gate is only safe at a scale that
makes it unnecessary.

There is a subtler trap underneath this one, and getting it wrong inflates every
number in the table. See [DECISIONS.md #7](DECISIONS.md).

### Finding 3 — the judge-complexity ladder, validated against known depth

Five scorers of increasing capability are fit under **group-aware** cross-validation to
reproduce the judge's labels. The cheapest rung that clears the κ bar is
simultaneously the deployment recommendation and the judge's effective complexity.

| scenario | designed depth | measured depth | majority | length | keyword | bow | charngram |
|---|---|---|---|---|---|---|---|
| `support_triage` | keyword | **keyword** | 0.000 | 0.332 | 0.849 | 0.849 | 0.849 |
| `rag_grounding` | bow | **bow** | 0.000 | 0.000 | 0.307 | 0.836 | 0.830 |
| `code_patch` | charngram | **charngram** | 0.000 | 0.298 | −0.001 | −0.022 | 0.838 |
| `tool_calling` | none | **none** | 0.000 | −0.010 | 0.009 | 0.023 | 0.017 |

**Four for four.** The instrument recovers a depth that was known by construction, at
each of four different rungs — including the one where the honest answer is "nothing
cheaper works". That last row is the one that matters: `tool_calling`'s label is the
agreement between a tool result and a claim, both marginally balanced, so no linear
model over any bag of features can beat chance at any vocabulary size. Without it,
"your judge is shallow" would be a conclusion the ladder reaches about everything.

Pointing it at a real GPT-4 judge is one line. That experiment is next, and it is the
first thing this repository should be judged on.

---

## The judge-complexity ladder

This is the idea the library is really about, so it gets its own section.

The obvious thing to do with an expensive LLM judge is distil it into a cheap
classifier so every production turn can be scored, then validate the distillation by
measuring agreement. High agreement, success.

**High agreement is not success. It is ambiguous**, and the ambiguity is the
interesting part:

| κ(cheap scorer, judge) | reading A | reading B |
|---|---|---|
| **high** | you can now score every turn for ~$0 | your judge's decision was a function of surface features all along |
| **low** | online scoring needs a real model | your judge was doing something a bag of words cannot |

Both readings are true at once. **The distilled scorer and the shortcut baseline are
the same object.** This is Hewitt & Liang's control task transplanted from probing
classifiers to LLM judges: fit the cheapest possible model to the thing you claim is
sophisticated, and see how much survives.

| rung | scorer | free parameters | what a match means |
|---|---|---|---|
| 0 | `majority` | 0 | the judge's labels barely vary; the eval measures nothing |
| 1 | `length` | 1 threshold | the judge is reading response length |
| 2 | `keyword` | 1 token + polarity | it is pattern-matching a single phrase |
| 3 | `bow` | tf-idf word 1–2-grams | the decision is linear in lexical content |
| 4 | `charngram` | tf-idf char 2–5-grams, case-sensitive | it is linear in formatting and morphology |
| 5 | the judge | — | nothing cheaper reproduces it |

```
rung       kappa vs judge                acc    us/turn      $/1k
-----------------------------------------------------------------
majority   0.0000 [0.0000, 0.0000]    0.7050        1.3    0.0000
length     0.3032 [0.2505, 0.3503]    0.7163       47.9    0.0000
keyword    0.8776 [0.8508, 0.9004]    0.9506       25.3    0.0000  <-
bow        0.8776 [0.8508, 0.9004]    0.9506      368.8    0.0000
charngram  0.8776 [0.8508, 0.9004]    0.9506      723.5    0.0000
judge      1.0000 (by definition)                   1.7    0.0000

  judge_depth = keyword
```

Two caveats the output states rather than hides:

- A match at rung 3 means the decision is separable in lexical features **on this
  traffic**. Strong evidence of shallowness, not proof — an easy population makes a
  deep judge look shallow. `ladder(..., by="cluster")` reports one ladder per traffic
  cluster for exactly that reason, and a judge that is rung 2 on your busiest cluster
  and unreachable on the tail is a better finding than one global number.
- Cross-validation is group-aware over `session_id`. Traces from one session are
  near-duplicates; a random split makes every rung look one level deeper than it is.

And the deployment recommendation falls out for free. The reason nobody runs per-turn
online evaluation is that everyone assumed the judge had to come along.

---

## The local platform

The measurements above are a library. This is the thing you can put on a screen.

```bash
livingeval serve --demo          # seeds a drifting corpus, deploys a scorer, opens :8000
```

Three pages, one SQLite file, no authentication, bound to localhost. It is a control
surface for your own machine — see [DECISIONS.md #25](DECISIONS.md) for why it is
deliberately not more than that.

| | |
|---|---|
| **`/`** | the three numbers live: coverage, power, gate. Buttons to run each measurement, deploy a scorer, and mine new cases. A latency sparkline for the online scorer. |
| **`/review`** | the human-in-the-loop queue. Two buttons and a keyboard shortcut, because approving mined proposals from a REPL is a chore nobody repeats. |
| **`/docs`** | the OpenAPI surface, generated. |

### Per-turn online scoring

The endpoint the whole judge-complexity ladder exists to justify:

```bash
curl -X POST localhost:8000/v1/score -H 'Content-Type: application/json' \
     -d '{"text": "I am unable to verify that right now."}'
```

```json
{"label": 0, "pass": false, "scorer": "keyword", "latency_us": 40.9, "kappa_vs_judge": 1.0}
```

**41 microseconds.** The original spec asked for under 50 ms; measured across 200 turns
the deployed rung answers in ~18 µs of model time, about **2,700× under** it. And it
ships its own agreement number with every score, because a label with no κ attached is a
random number generator with good latency.

### The API

```
POST /v1/score                       one turn, microseconds, no tokens
POST /v1/scorer/refit                fit the rung the ladder cleared
GET  /v1/coverage  /v1/blindspots    is the suite still a measurement?
GET  /v1/power     /v1/gate          would it fire? what does CI do?
GET  /v1/ladder    /v1/judge/validate
POST /v1/traces                      ingest (native or loose shape)
POST /v1/mine                        queue proposals from blind clusters
POST /v1/proposals/{id}/confirm      requires a named reviewer — 422 without one
POST /v1/promote  /v1/export/finetune
GET  /v1/status    /v1/records
```

Every measurement is persisted as a versioned record, so the dashboard and the JSON you
commit cannot disagree.

### Storage

```python
from livingeval.store import open_store

store = open_store("sqlite:///livingeval.db")     # default: stdlib, one file
store = open_store("postgresql://...")            # Supabase works unchanged
```

SQLite is the default because it is in the standard library, works everywhere, and
leaves you with one file you can copy or delete. Same schema on Postgres for when an
ingester, a scorer and a dashboard become three processes.

### Live ingestion

```bash
livingeval ingest --traces 'traces/*.jsonl' --db sqlite:///live.db --suite-from-first-window
livingeval ingest --follow --interval 30       # poll a live Langfuse project
```

`put_traces` is idempotent on `trace_id`, so a re-poll over an overlapping window is
harmless and the poller needs no cursor bookkeeping.

---

## Embeddings

The default geometry is tf-idf + SVD. Neural embeddings are a first-class option:

```python
from livingeval import embed, mine

space = mine.Space.fit(traces, embed=embed.hf("sentence-transformers/all-MiniLM-L6-v2"))
space = mine.Space.fit(traces, embed=embed.openai("text-embedding-3-small"))
space = mine.Space.fit(traces, embed=embed.hashing())    # offline, no download
```

All four backends are disk-cached — a corpus is encoded once — and all four put their
identity into the result record, because **coverage computed in two embedding spaces is
not comparable**.

Whether it changes your conclusions is an empirical question, so there is a command that
answers it rather than an opinion:

```bash
livingeval compare-spaces --traces 'traces/*.jsonl' \
    --embed hashing hf:sentence-transformers/all-MiniLM-L6-v2
```

On the bundled drifting corpus, tf-idf and a hashed dense embedding put coverage at
**0.432 vs 0.390** — a spread of 0.042, and both rank the same cluster as the worst blind
spot. On that traffic the cheap default is not costing a conclusion. On yours it might;
that is the point of measuring.

One detail worth knowing: even when an embedder supplies the geometry, a tf-idf
vectoriser is fitted alongside purely to *name* clusters. Without it a blind-spot report
reads "cluster 7", which nobody can act on. Labels never feed a number.

---

## The fine-tuned rung

Rungs 0–4 are diagnostics: weak in nameable ways, so a match tells you what the judge was
reading. A fine-tune has no such reading — a high κ means "a transformer can fit a
self-consistent labelling function", which is true of almost any judge.

So it ships as **rung 5, opt-in, and answering a different question**: not *what is the
judge reading* but *can I deploy anything at all*. Reach for it when `judge_depth` came
back `None` and you still want per-turn scoring.

```python
le.scorer.ladder(traces, judge, finetune=True)                          # local, CPU
le.scorer.ladder(traces, judge, finetune={"backend": "unsloth"})        # LoRA on a GPU
le.scorer.ladder(traces, judge, finetune={"model": "distilbert-base-uncased"})
```

```
rung       kappa vs judge                acc    us/turn      $/1k
-----------------------------------------------------------------
majority   0.0000 [0.0000, 0.0000]    0.7633        0.9    0.0000
length     0.0000 [0.0000, 0.0000]    0.7633       35.1    0.0000
keyword   -0.0176 [-0.0746, 0.0308]   0.7467       30.4    0.0000
bow        0.0762 [-0.0385, 0.1751]   0.5900      267.3    0.0000
charngram  0.0698 [-0.0513, 0.1798]   0.6100      659.8    0.0000
finetune  -0.0457 [-0.1491, 0.0356]   0.6167    12171.7    0.0000
           NOTE: a fine-tuned rung is a deployment number, not a diagnostic. It clearing
           the bar says a transformer can fit this judge, which is true of most judges.
           Read rungs 0-4 for what the judge was reading.
           local:sentence-transformers/all-MiniLM-L6-v2@main(epochs=3), 32.8s to train
```

That note prints wherever the number appears. Three findings from running it on the
bundled scenarios, all reproducible with `pytest -m download`:

- On `shortcut` it reaches **κ = 1.00** — the rung works.
- On `tool_calling` (`deep`) it reaches **κ ≈ 0**. Even a fine-tuned transformer cannot
  learn a rule that is the XOR of two balanced features from 200 examples. **A fine-tune
  is not a guaranteed ceiling**, and pretending otherwise is how you spend a GPU week on
  a judge that was never reproducible.
- At **12 ms per turn** it is ~500× slower than `keyword`. That is the honest cost of the
  deployment option, and the reason the cheap rungs are worth measuring first.

Three backends: `local` (torch + transformers, CPU-capable), `unsloth` (LoRA, needs CUDA),
`fireworks` (hosted). Human-confirmed labels export as training data in three formats:

```bash
livingeval export --suite support --format chat --split 0.2
```

`confirmed_only` is the default, so the exported dataset is human-labelled by
construction. Training a scorer on the judge's own labels and then measuring it against
the judge is a closed loop that reports success no matter how wrong the judge is.


## What it does

| | |
|---|---|
| **Coverage** | calibrated-radius coverage with a Wilson interval, per traffic cluster, with published sensitivity to the one tunable |
| **Blind spots** | clusters ranked by `traffic_share × (1 − coverage)`, named by their distinctive terms, with the suite-case count |
| **Power** | Monte-Carlo detection power against a named regression, false-alarm rate from the same code path, staleness curves over time |
| **Judges** | pluggable (rule / OpenAI / Anthropic), content-addressed disk cache, cost and latency measured not quoted, κ against human labels or `UNVALIDATED` |
| **Ladder** | five rungs, group-aware CV, chance-corrected agreement, `judge_depth`, per-cluster breakdown |
| **Gate** | `PASS` / `FAIL` / `BLIND` with distinct exit codes, exact McNemar and paired bootstrap, `legacy` threshold mode for comparison |
| **Mining** | tf-idf → SVD → KMeans clustering, greedy k-centre proposals from blind clusters, mandatory named confirmation |
| **Ingest** | JSONL (native or loose), OpenTelemetry spans, Langfuse exports and live polling |
| **Store** | SQLite (stdlib, default) or Supabase/Postgres — traces, verdicts, suites, the review queue, records; versioned migrations with an advisory lock |
| **Deploy** | Docker, ECS Fargate + ALB via Terraform, S3 or Supabase Storage for artifacts, bearer auth, `/healthz` + `/readyz` + `/metrics`, a background worker |
| **Embeddings** | tf-idf+SVD (default), offline hashed dense, Hugging Face, sentence-transformers, OpenAI — all cached, all recorded |
| **Fine-tuning** | rung 5 of the ladder: local torch, Unsloth LoRA, or Fireworks; plus training-data export in three formats |
| **Platform** | local FastAPI app: dashboard, review queue, `/v1/score` at ~40 µs, full REST API |

## Using it on your own traces

```python
import livingeval as le

traces = le.ingest.jsonl("traces/2026-08/*.jsonl")   # or .otel(...) / .langfuse(...)
suite  = le.EvalSuite.load("evals/support.json")
judge  = le.judge.openai("gpt-4o-mini", prompt=MY_RUBRIC)

cov    = le.mine.coverage(suite, traces)
blind  = le.mine.blindspots(suite, traces, top=5)
val    = le.judge.validate(judge, traces.labelled())
ladder = le.scorer.ladder(traces, judge, cv="session")

props  = le.mine.propose(traces, suite, n=20, judge=judge)
suite2 = suite.extend(props.confirm(reviewer="sk"))
le.report.save([cov, blind, val, ladder], "results/support.json")
```

```bash
livingeval coverage   --suite evals/support.json --traces 'traces/*.jsonl'
livingeval blindspots --suite evals/support.json --traces 'traces/*.jsonl' --top 5
livingeval ladder     --traces 'traces/*.jsonl'  --judge openai:gpt-4o-mini
livingeval power      --suite evals/support.json --traces 'traces/*.jsonl' --effect 0.3
livingeval gate       --suite evals/support.json --baseline results/main.json \
                      --traces 'traces/*.jsonl'          # exits 0 / 1 / 2
livingeval audit                                          # the three findings
```

Two fields on a trace do all the work. `ts` is what makes staleness a measurable
quantity rather than a feeling. `session_id` is what makes cross-validation honest.
Without them the library still runs and says so.

## Testing

Three layers, and the scenario-shaped one is not the important one:

- **Layer A — analytical invariants and library promises.** No model, no network, no
  reference numbers. κ of independent labellings ≈ 0, exact McNemar reduces to the
  binomial, bootstrap CIs cover truth ~95% of the time, coverage of a suite against
  itself is 1.0 — plus the five that encode what the README claims: *a keyword
  reproduces a one-phrase judge*, *nothing below the judge reproduces a cross-span
  one*, *a frozen suite loses coverage under drift*, *its power against an unseen
  regression is the false-alarm rate*, *the threshold gate false-alarms more than the
  paired one*. This is the real test suite.
- **Layer B — fixed-fixture goldens.** A pinned deterministic config, recorded once by
  an explicit script. Catches numerical drift from a refactor or a scikit-learn
  release. Golden numbers need determinism, not a publication — and there is no
  `--update` flag, because a suite that rewrites its own expectations verifies nothing.
- **Layer C — scenario reproductions.** `pytest -m reproduction`, off by default. Each
  bundled scenario asserts the ladder recovers its designed depth. Off because slow,
  not because optional.

Plus the one that guards everything else: a subprocess test asserting that
`import livingeval` and a full analysis run never pull torch, transformers, fastapi or
matplotlib into the process. The `import-linter` contracts approximate that statically;
this checks what the import machinery actually did.

**143 tests, ~38 seconds, no downloads.** A further 11 are marked `reproduction` or
`download` and run on demand.

## Architecture

```
src/livingeval/   the engine -- zero scenario-specific content, enforced by import-linter in CI
suites/           scenario generators, judges and failure taxonomies
```

If a symbol names a specific agent task, a specific judge prompt or a specific failure
taxonomy, it does not live in `src/livingeval/`. **Four `import-linter` contracts** are
enforced in CI:

1. the engine may not import `suites`;
2. the analysis layers may only import downward;
3. nothing outside `serve/` may import fastapi — the platform may depend on the library,
   never the reverse;
4. modules that never touch an optional dependency must keep it that way.

Contract 3 caught a real defect while this was being built: `serve/app.py` was importing
a judge-spec parser from `cli.py`, an upward dependency. The fix was to move the parser
into `judge/spec.py` and `sources.py`, where the CLI, the platform and any script can
share it. That is the contract earning its place. See [DECISIONS.md](DECISIONS.md).

## Running it for real

```bash
python scripts/dev.py up        # local Supabase: Postgres + Storage + Studio, migrated and seeded
python scripts/dev.py serve     # the dashboard against it
python scripts/dev.py test      # the tests that need a real Supabase
```

The local stack is the **real** Supabase, not SQLite and not a bare Postgres — same
pooler semantics, same Storage API, same row-level security, same migrations. Going to
Supabase Cloud and AWS is then a diff of five environment variables and no code.

[`DEPLOYMENT.md`](DEPLOYMENT.md) has the whole path: local → Supabase Cloud → ECS
Fargate, the exact credentials needed at each step, and what it costs (~$50/month).

## Related

[`probeit`](https://github.com/itsskofficial/probeit) applies the same argument one
layer down the stack, to probing classifiers in interpretability: control tasks and
selectivity as the default return value, group-aware CV, permutation tests, figures
from records. Different domain, same claim — *a measurement you haven't controlled is
not a measurement, and the control has to be the default or nobody runs it.*

