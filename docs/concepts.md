# Concepts

Four quantities, each with a precise definition, a computation and a failure mode.

---

## Coverage

**The question.** Does this suite still represent the traffic I serve?

**The definition.** A production trace is *covered* when its distance to the nearest
suite case is at most `tau`, where

> `tau` = the *q*-th percentile (default 50) of nearest-neighbour distance **within the
> suite itself**

Read it as: covered means *no further from the suite than the suite's own members are
from each other*.

**The computation.** Fit a `Space` on the traffic (tf-idf → truncated SVD, L2
normalised, so a dot product is a cosine similarity). Transform the suite into it.
Compute `tau` from the suite's internal nearest-neighbour distances, then the fraction
of traffic within `tau` of any case. The interval is Wilson.

**Why the radius is calibrated rather than chosen.** A fixed cosine threshold is
unitless and corpus-dependent — there is no argument you can have with it. See
[DECISIONS.md #1](../DECISIONS.md).

**Failure modes, all reported rather than hidden.**

| situation | what you get |
|---|---|
| suite has fewer than 2 cases | `UNCALIBRATED`, `coverage: null` |
| suite is empty | `EMPTY_SUITE` |
| the answer moves a lot with `q` | the `sensitivity` field says so — check q25 and q75 |
| you changed representation | the `representation` field differs; the two numbers are not comparable |

**Read it as a leading indicator.** In the audit it is the first of the three signals
to move and by far the cheapest to compute. If you instrument one number, this one.

---

## Blind spots

**The question.** *Which* traffic can it not see?

**The computation.** Cluster the traffic (tf-idf → SVD → KMeans, `k` by silhouette,
seeded). Compute coverage per cluster. Assign each suite case to the *traffic's*
nearest centroid — the same partition the coverage table uses — and count. Rank by

> `severity = traffic_share × (1 − coverage)`

**Why that product.** Neither factor alone orders correctly. A completely uncovered
cluster carrying 0.4% of traffic is not the problem. A cluster carrying 30% that is
already well covered is not either.

**What to do with it.** A cluster with `n_suite_cases == 0` and non-trivial traffic
share is one where a regression cannot move the suite score at all. That is the
actionable line, and `mine.propose` draws from exactly those clusters.

Clusters are a **summarisation device**, not a claim about real structure. The
decision-relevant quantity, coverage, is computed per trace.

---

## Detection power

**The question.** If the regression I am worried about were present, how often would
the gate fire?

**The definition.** A `Regression` is a predicate over cases plus an effect size: among
the cases the predicate selects, a currently-passing case fails with probability
`effect`. Power is the fraction of Monte-Carlo replicates in which the gate returns
`FAIL`.

**The simulation, precisely.** Each replicate draws a fresh baseline arm *and* a fresh
current arm from the same per-case pass probabilities (`1-flake` where the case passed,
`flake` where it failed), then applies the regression to the current arm only, then
runs the gate you actually configured.

**`flake` is the parameter to think about.** At `flake=0` every gate has a false-alarm
rate of zero and every comparison between gate designs is vacuous. The 5% default is a
placeholder. `power.measure_flake(run_a, run_b)` estimates it from two real runs of an
unchanged suite; use that.

**Companion number.** `power.false_alarm_rate` is the same code path at `effect=0` — the
gate's realised Type-I error. They come from one implementation because otherwise they
are not comparable.

**`is_blind`.** When no case in the suite is affected by the regression, the reported
"power" *is* the false-alarm rate, and the summary says so in those words rather than
letting a 0.01 read as a detection rate.

**It is a statement about the instrument.** Given this suite's composition and this
gate's statistics. Not a prediction about your model.

---

## Judge depth

**The question.** Could something far cheaper have produced the judge's labels?

**The computation.** Five scorers of increasing capability are fit under group-aware
cross-validation over `session_id` to reproduce the judge's out-of-fold labels.
`judge_depth` is the cheapest rung whose Cohen's κ against the judge clears the bar
(default 0.80).

| rung | scorer | what a match means |
|---|---|---|
| 0 | `majority` | the labels barely vary; the eval measures nothing |
| 1 | `length` | the judge is reading response length |
| 2 | `keyword` | it is pattern-matching a single phrase |
| 3 | `bow` | the decision is linear in lexical content |
| 4 | `charngram` | it is linear in formatting and morphology |
| 5 | the judge | nothing cheaper reproduces it |

**Two readings, always together.** The matching rung is what you deploy for per-turn
online scoring, *and* it is a statement about how shallow the judge's decision was. See
[DECISIONS.md #5](../DECISIONS.md).

**What it does not prove.** A match at rung 3 means the decision is separable in
lexical features *on this traffic*. An easy population makes a deep judge look shallow.
Use `ladder(..., by="cluster")`: a judge that is rung 2 on your busiest cluster and
unreachable on the tail is a better and more accurate finding than one global number.

**Judge noise caps the ladder.** `judge_depth` measures reproducibility of the judge's
*labels*, noise included. A 10%-noisy judge cannot be reproduced above κ ≈ 0.78 by
anything, which is why the pinned Layer-B fixture records `ladder_depth: null` for a
judge whose underlying rule is a single phrase.

---

## The gate

| outcome | exit | meaning |
|---|---|---|
| `PASS` | 0 | no regression detected, and the suite could have detected one |
| `FAIL` | 1 | a regression was detected |
| `BLIND` | 2 | no regression detected, but the suite lacks the coverage or power to justify saying so |

**Precedence is `FAIL` first.** A suite that caught a regression has done its job;
downgrading that to `BLIND` because coverage slipped would suppress a real signal.

**The test.** `mode="paired"` runs exact McNemar on the discordant cases and reports the
effect size with a paired-bootstrap interval. Both arms are the same cases, so the
between-case variance cancels — that is why a 40-case suite can detect anything.

**`mode="threshold"`** reproduces `score < threshold`. It ships so the gap between the
two can be measured; the audit measures it at up to 34% false alarms on a 25-case
suite against 0% for the paired test.
