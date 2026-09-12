# Decisions

Why `livingeval` is built the way it is. Each entry records the alternative that was
rejected, because that is usually the more useful half.

---

## 1. Coverage uses a radius calibrated from the suite, not a chosen threshold

**Decision.** A production trace is covered when its distance to the nearest suite case
is at most `tau`, where `tau` is the *q*-th percentile (default 50) of
nearest-neighbour distance **within the suite itself**.

**Why.** "Within 0.3 cosine" is unitless, corpus-dependent and unfalsifiable — it
cannot be argued with, only accepted. Calibrating from the suite's own geometry gives
the definition a reading a person can dispute: *a trace is covered when it is no
further from the suite than the suite's own members are from each other*. It is
scale-free, so it survives a change of representation, and it degrades correctly: a
tightly clustered suite earns a small radius and correctly reports that it covers very
little of diverse traffic.

**Rejected.** A fixed cosine threshold. A k-nearest-neighbours rule (same arbitrariness,
moved into `k`). A density-ratio estimator (better statistics, but it needs far more
data than a 150-case suite has, and the failure is silent).

**Consequence.** Every coverage number ships with its sensitivity to *q*, because one
tunable that nobody varies is a constant in disguise. A suite of one case has no
internal distances, so `tau` is undefined and the result is `UNCALIBRATED` with
`coverage: null` rather than a fabricated number.

---

## 2. Clustering is tf-idf → SVD → KMeans, not neural embeddings

**Decision.** The default `Space` is tf-idf over word 1–2-grams reduced by truncated
SVD. `Space.fit(embed=...)` swaps in any embedding function, and the representation
name is recorded in every result.

**Why.** Three reasons in order of importance. It is deterministic and needs no
download, no GPU and no API key, so the coverage assertions in the test suite run on
every commit rather than being skipped in CI. Coverage computed in two embedding
spaces is not comparable, and a default that silently changes when someone upgrades a
model is a reproducibility trap. And on agent traffic, topic separation is largely
lexical — the expensive representation buys less here than on the sentence-similarity
benchmarks it was tuned for.

**Consequence.** Cross-project coverage comparisons require the same representation,
so the representation is in the record rather than in a footnote.

---

## 3. The space is fitted on the traffic, never on the suite or the union

**Decision.** `Space.fit(traces)`, then the suite is transformed into it.

**Why.** Coverage asks whether the suite represents the traffic, so the traffic defines
the geometry. Fitting on the union lets a suite improve its own coverage by adding
cases in a corner of the space nothing else occupies — the metric would reward exactly
the behaviour it exists to discourage.

---

## 4. Clustering reads the request, judging reads the whole trace

**Decision.** `mine.*` defaults to `view="request"` (user turns only). `judge.*` and
`scorer.ladder` default to `view="full"`.

**Why.** A blind spot is a region of *demand*: something users are asking for that the
suite has no case for. Cluster the full trace and a large share of the partition ends
up describing your own response templates — "the cluster where the agent says it has
logged the reference" — which is a fact about your prompt, not your traffic, and which
moves every time you edit that prompt. This was visible in development: on the same
corpus, full-trace clustering produced five of eleven clusters keyed on the agent's
closing sentences.

The ladder is the opposite case. It must see what the judge saw, or it answers a
different question than the one asked.

**Consequence.** The view is part of every record. Two coverage numbers computed under
different views are not comparable.

---

## 5. The distilled scorer *is* the shortcut baseline

**Decision.** One object, two readings, reported together. `judge_depth` is both the
deployment recommendation and the shallowness diagnostic.

**Why.** The industry default is to distil a judge into a cheap model, measure
agreement, and report high agreement as success. High agreement is ambiguous: it means
you can score every turn for nothing *and* that the judge's decision was a function of
surface features. Reporting only the first half is half a result. This is Hewitt &
Liang's control task moved from probing classifiers to LLM judges.

**Rejected.** A single distilled model with a "distillation quality" score. A
gradient-boosted or fine-tuned distillation — it would climb higher and tell you
nothing, because "a strong model can predict the judge" is true of almost any judge.
The ladder is useful precisely because its rungs are weak in interpretable ways.

**Consequence.** Every rung is linear or simpler, and rung 4 is the only case-sensitive
one, because word-level tf-idf lowercases and erases exactly the camelCase/snake_case
distinction that separates "reading content" from "reading formatting".

---

## 6. The ladder's cross-validation is group-aware over `session_id`

**Decision.** `StratifiedGroupKFold` by default. When the data cannot support it the
split degrades to stratified-only and **the fallback is recorded in the result**.

**Why.** Traces from one session share a user, a topic and most of their tokens. A
random split scores the model on text it has effectively already seen, which makes
every rung look one level deeper than it is — the ladder would systematically report
judges as shallower than they are, which is the specific error it exists to avoid
making. This is the same defect `probeit` measures on probing datasets, in a new place.

**Rejected.** Silently falling back to a random split. A group-blind number reported as
group-aware is worse than no number.

---

## 7. The false-alarm simulation redraws **both** arms, and the threshold is set against the expected rerun score

**Decision.** Each replicate draws a fresh baseline and a fresh current arm from the
same per-case probabilities. `power.expected_rerun_score` gives the mean a rerun should
produce, and fixed thresholds in the audit are set against that.

**Why.** Two traps, and the second cost a full audit run to find.

Holding the baseline at its observed value and perturbing only the current arm gives
the gate a noise-free reference it never has in practice. Every gate design then scores
a false-alarm rate of zero and the comparison between them is vacuous.

The subtler one: under per-case instability `f`, the *expected* score of a rerun is
`(1-f)·s + f·(1-s)`, pulled toward one half — not the `s` you observed once. Setting a
threshold two points under the **observed** score therefore puts the line *above* the
rerun mean, and the gate fires almost every time for a reason unrelated to the model.
The first version of the audit did this and reported threshold-gate false-alarm rates
rising from 0.71 to 0.995 with suite size, which is backwards: the correct numbers fall
from 0.34 to 0.003 as the standard error shrinks. A table measuring your own mistake
looks exactly like a table measuring theirs.

**Consequence.** `flake` is the parameter to think hardest about, and at `flake=0`
every gate has a false-alarm rate of zero and the whole comparison is empty. The 5%
default is a placeholder; `power.measure_flake(run_a, run_b)` estimates it from two
real runs of an unchanged suite, and that is the number to use.

---

## 8. Gates are paired tests; `legacy` threshold mode ships anyway

**Decision.** `mode="paired"` uses exact McNemar on the discordant cases and reports the
effect with a paired-bootstrap interval. `mode="threshold"` reproduces the naive gate,
is labelled, and is used for the audit comparison.

**Why.** The same cases run before and after, so the comparison is paired and the
between-case variance — most of it — cancels. That width is why a 40-case suite can
detect anything at all. McNemar conditions on the discordant count and reduces to a
binomial, so there is no reason to use a chi-square approximation on the small counts
an eval suite actually produces.

The naive gate stays because the gap between the two is worth **measuring** rather than
asserting, and the measurement is Finding 2.

---

## 9. `BLIND` is a third outcome with its own exit code

**Decision.** `PASS` → 0, `FAIL` → 1, `BLIND` → 2. `FAIL` takes precedence over
`BLIND`.

**Why.** "The suite passed" and "the suite is no longer capable of failing" are
different states, and folding the second into the first is the specific failure this
library exists to prevent. A distinct exit code means a CI config can branch on it —
warn, open a ticket, page nobody — instead of a human noticing a dashboard.

Precedence goes to `FAIL` because a suite that caught a regression has done its job,
and downgrading that to `BLIND` because coverage slipped would suppress a real signal.

---

## 10. Mined cases require an explicit, named confirmation

**Decision.** `Proposal.confirm(reviewer=...)` stamps who confirmed it and what they
said the answer was. Unconfirmed mined cases may enter a suite — they improve coverage
and the reviewer needs to see them — but they are excluded from every gate decision and
rendered with a marker.

**Why.** An eval suite that grows by absorbing production traces will, unwatched,
converge on asserting that current behaviour is correct by definition: the model passes
because the cases were harvested from what the model already does. That failure is
silent, slow and total, and the only defence is procedural. So confirmation is a field
with a name in it, not a boolean.

**Consequence.** `confirm()` takes per-case label overrides, because the common outcome
of a review is that most suggestions stand and a few were wrong — and the wrong ones
are exactly the cases worth having.

---

## 11. Nothing scenario-specific may live in the engine

**Decision.** `src/livingeval/` contains no agent task, judge rubric or failure
taxonomy. Those live in `suites/`. Enforced by an `import-linter` contract in CI,
alongside a layers contract that forbids upward imports.

**Why.** A library whose core namespace contains `support_queue_correct` is one
person's project with an API on top. The rule has to be mechanical, because "we'll keep
it clean" never survives the second scenario.

---

## 12. The default install is pure numpy/scipy/scikit-learn

**Decision.** No torch, no API client, no network in `pip install livingeval`. OpenAI,
Anthropic, matplotlib and PyYAML are extras, all lazily imported.

**Why.** The entire invariant suite runs on every commit in about 25 seconds with no
model, no GPU and no download. That is the difference between tests that run and tests
that are skipped in CI. A dedicated CI job asserts the default install stays clean,
because this erodes one convenience import at a time.

**Consequence.** The distilled scorers are tf-idf and logistic regression rather than a
fine-tuned transformer. This is not a compromise — see #5. It is the design.

---

## 13. Judge verdicts are content-addressed and cached to disk

**Decision.** The key is `(provider, model, temperature, rubric hash, view)` paired with
the trace's content hash. One JSON file per judge identity.

**Why.** A tool that burns API budget on every iteration gets run once. Caching on the
rubric as well as the trace matters: editing one word of the prompt changes the judge,
and serving the previous verdicts would make a judge revision invisible.

**Consequence.** The trace content hash deliberately excludes `ts`, `label` and `meta`.
The verdict is a function of the conversation, so the same conversation seen twice must
hit the same entry — otherwise every rerun with a fresh timestamp is a cache miss and a
bill.

---

## 14. Cost and latency are measured, never quoted

**Decision.** Every `Verdict` carries wall-clock, and API judges derive cost from the
provider's own token counts. A model missing from the price table reports
`cost_usd=0.0` with `price_unknown` in the meta rather than a wrong number.

**Why.** Published prices move faster than any library's release cycle, and a stale
price presented as a measurement is worse than an absent one.

**Consequence.** The ladder reports **single-item** latency, timed one call at a time.
Vectorising 2,000 documents at once hides the per-call overhead that dominates a live
turn, and per-turn online scoring is the entire use case. Batch throughput is recorded
too, and it is the smaller number.

---

## 15. Power is a simulation, and every surface says so

**Decision.** The word "simulated" appears in the function docstring, the printed
summary, the record schema and the README.

**Why.** The estimate answers *given this suite's composition and this gate's
statistics, if the named regression were present, how often would the gate fire?* It is
a statement about the measuring instrument, not a prediction about your model.
Overclaiming here is the fastest way to lose the only audience that can tell the
difference.

**Consequence.** A `Regression` is an object you construct and that goes into the
record. "Your power is 0.4" without naming the alternative is not a number.

---

## 16. Both the averaged and the worst-cluster power are reported

**Decision.** `staleness_curve` returns expected power over clusters *and* the worst
cluster's power, and the summary points at the gap.

**Why.** In the audit the averaged power never falls below 0.60 across six windows
while the worst cluster falls to 0.005. Reporting only the average would hide the blind
spot — which is the same aggregation failure, one level up, that the library exists to
point at. Reporting only the worst would be alarmist on a suite with one small ragged
cluster. Both, with the gap named.

---

## 17. Golden numbers come from a pinned config, not from a publication

**Decision.** Three layers. **A**: analytical invariants and library promises — no
model, no network, no reference numbers. **B**: goldens from a pinned deterministic
generator config, recorded once by an explicit script. **C**: scenario reproductions,
marked `reproduction`, off by default.

**Why.** Correctness regression and reproduction of a published result are different
jobs that got conflated under one name. For CI, a synthetic fixture is strictly better
than a real corpus: faster, no download, no licence question, no GPU.

**Consequence.** Recording expectations is a separate script with no `--update` flag. A
suite that rewrites its own expectations on failure verifies nothing.

---

## 18. The bundled scenarios are simulated, and the README says so first

**Decision.** All four ship as generators with a known failure mechanism at a known
rung, and the audit section opens by stating it.

**Why.** Findings 1 and 2 are claims about suite composition and gate statistics —
properties of the measuring instrument — and simulation is the right instrument for
those, exactly as a synthetic leaky dataset is the right instrument for demonstrating
CV leakage. Finding 3 is the load-bearing one: it validates the ladder against judges
whose depth is known by construction, which is the precondition for pointing it at a
real judge and believing the answer. Burying that would trade a small amount of
credibility now for all of it later.

---

## 19. `argparse` for the CLI

**Decision.** No CLI framework dependency.

**Why.** `pip install livingeval` should give a working `livingeval` command with
nothing else pulled in. Nine subcommands is well inside what argparse handles.

---

## 20. Secrets come from the environment only

**Decision.** No key, token or credential is ever read from a source file.
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `LANGFUSE_*` come from the environment.

**Why.** Research and eval code routinely ships with a hard-coded key in a notebook. It
is a one-line habit that costs a credential rotation.

---

## 21. Adapters read files, not APIs

**Decision.** `ingest.langfuse` reads an export file. `ingest.otel` reads a JSON dump.
`langfuse.fetch()` exists, is the only function that touches the network, and it writes
an export file then calls the offline reader.

**Why.** Every number this library produces should be reproducible from an artifact you
can commit, diff and attach to a bug report. A function that re-fetches from a live API
produces numbers nobody else can reproduce, including you next week.

---

## 22. Embeddings are a first-class option, and the default is still tf-idf

**Decision.** Four embedding backends ship behind one protocol — offline hashed dense,
Hugging Face, sentence-transformers, OpenAI — all disk-cached, all recording their
identity in the result. `Space.fit(embed=...)` swaps the geometry. The **default stays
tf-idf + SVD**.

**Why not make a neural encoder the default.** Three reasons in order: the default has
to run in CI with no download, so coverage assertions are checked on every commit rather
than skipped; coverage computed in two embedding spaces is not comparable, and a default
that silently changes when someone upgrades a model is a reproducibility trap; and on
agent traffic, topic separation is mostly lexical.

**But that last one is a claim, so it is measured rather than asserted.**
`livingeval compare-spaces` runs both and diffs the coverage and the blind-spot ranking.
On the bundled drifting corpus the answer is 0.432 against 0.390 — a spread of 0.042,
with the same cluster ranked worst by both. On that traffic the cheap default costs
nothing. On other traffic it might, which is exactly why the command exists instead of a
paragraph of reassurance.

**Consequence.** `embed.hashing()` is a genuine dense embedding — signed random
projection of hashed n-grams, `blake2b` rather than Python's salted `hash()` so it is
stable across processes — and a deliberately weak one. It has no notion of synonymy. It
exists so the abstraction is exercised on every commit, and its docstring says what it
costs you.

---

## 23. Even with an embedder, cluster labels come from tf-idf

**Decision.** When an embedder supplies the geometry, a tf-idf vectoriser is fitted
alongside for the sole purpose of naming clusters.

**Why.** A blind-spot report that reads "cluster 7" is useless to the person who has to
act on it; one that reads "wallet address, wrong chain, gas fee" is actionable. Embedding
coordinates have no readable labels, so the label has to come from somewhere else.

**Consequence.** Geometry and labels come from different places on purpose, and the
labels never feed a number. Stated here because a reader who notices two representations
in one object should be told which one the arithmetic uses.

---

## 24. The fine-tuned rung ships, opt-in, and answers a different question

**Decision.** `ladder(..., finetune=True)` appends rung 5. It is not in the default
ladder. Its `RungResult` carries `is_diagnostic=False`, and a note is printed wherever
its number appears.

**Why it was almost left out.** Rungs 0–4 are weak in *nameable* ways, so a match says
what the judge's decision was a function of. A fine-tune has no such reading: a high κ
means "a transformer can fit a self-consistent labelling function", true of almost any
judge. Reporting it as though it diagnosed something is the mistake the ladder exists to
avoid.

**Why it ships anyway.** There is one situation where it is exactly right: `judge_depth`
came back `None`, nothing cheap reproduces the judge, and you still want per-turn online
scoring. That is a real engineering problem, the ladder has already given you the κ floor
to beat, and answering "can I deploy anything at all" is worth a rung. Leaving it out
would have been answering the user's question with a lecture.

**What running it taught us**, all reproducible with `pytest -m download`:

- On `shortcut` it reaches κ = 1.00. The rung works.
- On `deep` it reaches κ ≈ 0. Even a fine-tuned transformer cannot learn the XOR of two
  balanced features from 200 examples. **A fine-tune is not a guaranteed ceiling**, and
  assuming it is would cost a GPU week on a judge that was never reproducible.
- At ~12 ms per turn it is roughly 500× slower than `keyword`. That is the honest price
  of the deployment option and the reason to measure the cheap rungs first.

**Rejected.** Making it the default (slow, needs an extra, trains per fold). Reporting it
without the note (the industry default, and half a result). Using gradient boosting
instead — same objection, and no training cost to make you think twice.

**Implementation note.** The default checkpoint is
`sentence-transformers/all-MiniLM-L6-v2` rather than a smaller BERT. Several obvious tiny
checkpoints publish only a slow tokenizer, which `transformers` 5.x refuses to convert
without `sentencepiece`; the rung would fail on a fresh install with an error about
backend tokenizers. It is also the checkpoint `embed.hf()` defaults to, so the two
subsystems share one download.

---

## 25. The platform is local, unauthenticated, and stays that way

**Decision.** `livingeval serve` binds `127.0.0.1`, has no authentication, and keeps
everything in one SQLite file.

**Why.** Two things a library genuinely cannot do: serve per-turn scores from a warm
model, and give a human a review queue they will actually use. Both need a process. What
neither needs is multi-tenancy, and adding it is the point at which this becomes a
platform competing with four funded companies — which is not the pitch, and would put the
measurement behind a product.

**Consequence.** Analyses run on request and are cached in the store rather than on a
schedule. Coverage over 20k traces takes a couple of seconds; a scheduler would add a
configuration surface and a class of "why is this number stale" bug in exchange for
nothing at this size.

**Enforced.** An `import-linter` contract forbids anything outside `serve/` from importing
fastapi. The platform may depend on the library; the library may never depend on the
platform, or `import livingeval` starts needing a web stack.

---

## 26. The review queue refuses an unnamed reviewer

**Decision.** `POST /v1/proposals/{id}/confirm` returns 422 without a `reviewer`. So does
`reject`. `store.confirm_proposal` raises. The UI blocks the buttons.

**Why.** It is the one guard-rail that makes the mining loop safe to run unattended. An
eval suite that grows by absorbing production traces converges, unwatched, on asserting
that the model's current behaviour is correct by definition. The defence is procedural,
so it has to be a required field rather than a convention — and it is checked at three
layers because a guard-rail with one bypass is not one.

**Consequence.** `promote()` reports `skipped_unreviewed` rather than silently ignoring
pending proposals, so "the suite grew by 4" and "6 are still waiting for you" are both
visible.

---

## 27. SQLite is the default store, Postgres is the same schema

**Decision.** `sqlite3` from the standard library, one file, WAL mode. Postgres/Supabase
is the identical schema over `psycopg2`, behind an extra.

**Why.** For a local platform there is no second-best: no dependency, works everywhere,
and the whole database is a file you can copy, inspect or delete. Postgres exists for
when an ingester, a scorer and a dashboard become three processes and SQLite's
single-writer model becomes the bottleneck — and keeping one schema means moving is a URL
change, not a migration.

**Consequence.** No ORM, no migrations, no pooling. Six tables that have not changed
shape; `CREATE TABLE IF NOT EXISTS` on connect is the right amount of machinery, and more
would be infrastructure this library has no business owning.

**Detail worth knowing.** `put_traces` is idempotent on `trace_id`, so replaying a file
or re-polling an overlapping Langfuse window is harmless and the poller needs no cursor
bookkeeping of its own.

---

## 28. Training data is exported from human-confirmed labels only

**Decision.** `export_finetune_data(confirmed_only=True)` is the default. The holdout
split is by session.

**Why.** Once you have human-confirmed labels they are the correct supervision for a
distilled scorer — better than the judge's own labels, because they are the thing the
judge is approximating. Training on judge labels and then measuring agreement against the
judge is a closed loop that reports success no matter how wrong the judge is. Turning it
off is possible and it is not the default.

**Why the split is session-aware.** Same reason the ladder's CV is: sessions are
near-duplicates, and a random row split reports a validation number inflated by
memorisation.

---

## 29. Spec strings are parsed once, below both callers

**Decision.** `judge.from_spec("openai:gpt-4o-mini")` lives in `judge/spec.py`;
`sources.load_traces("synthetic:drifting,n=1200")` lives in `sources.py`. The CLI and the
platform both call them.

**Why.** They started in `cli.py`, which meant the web layer imported the CLI to reach
them — an upward dependency. The layers contract caught it on the first run after `serve/`
was added, which is the contract earning its place: nobody would have noticed by reading,
and the cost would have been paid later as a circular import.

**Consequence.** Format detection sniffs rather than requiring a flag. The first thing
anyone does with a new eval tool is point it at the log file they already have, and making
them declare its shape first is where adoption is lost.

---

## 30. The local database is Supabase, not SQLite and not a bare Postgres

**Decision.** `python scripts/dev.py up` runs the **real Supabase stack** locally — the
same Postgres, pgBouncer, Storage API, RLS and migrations that the cloud project has.
SQLite remains the default for `pip install livingeval`; Supabase is the default for
anything that will be deployed.

**Why not SQLite for development.** It is a different database. Every Supabase-specific
failure — pooler semantics, TLS enforcement, RLS on `public`, Storage permissions,
migration locking — is invisible until deploy day, which is the worst possible moment to
discover any of them. Two of the bugs in this build were found *only* because the local
stack was real: the session pooler being misclassified as a transaction pooler, and TLS
being required against a host that does not speak it.

**Why not a plain `postgres:16` container.** Closer, and still not it. No Storage API, no
`anon` role, no RLS enforcement to verify, no PostgREST to check is denied. A test that
proves the anon key cannot read production traces cannot be written against bare
Postgres, and that is a security property rather than a nicety.

**Consequence.** `deploy/docker-compose.yml` has no database service. It points at
Supabase — local by default, cloud by changing one variable. Moving from local to
production is a diff of five environment variables and nothing else.

---

## 31. Local Supabase runs on 544xx, not Supabase's 543xx defaults

**Decision.** `supabase/config.toml` shifts every port into the 544xx range.

**Why.** The default ports collide the moment you have a second Supabase project, and
the failure — `port is already allocated` — does not say which project took it. This was
not hypothetical: the first `supabase start` here failed against an unrelated project
already running on 54322.

**Consequence.** `store/supabase.py` carries the local endpoints as constants, so
detection and the docs cannot drift from the config.

---

## 32. Row-level security is on, with no policies, and it is tested

**Decision.** Migration `0002_rls` enables RLS on all six tables, defines **no policies**,
and revokes the `anon` and `authenticated` grants.

**Why.** Supabase exposes the `public` schema through PostgREST to the anon key that
ships in browser code. These tables hold production traces — real user text. livingeval
never uses the Data API; it connects over SQL as the database owner, and the service role
bypasses RLS. So denying PostgREST completely costs nothing and closes the hole.

**Verified, not assumed.** `test_the_anon_key_cannot_read_traces` makes a real HTTP
request with a real anon key against the real stack and asserts a denial. A security
property nobody executed is a comment.

---

## 33. Migrations are versioned, locked, and refused over the transaction pooler

**Decision.** An ordered `MIGRATIONS` list with a `schema_migrations` table, applied under
`pg_advisory_lock`. `livingeval db migrate` refuses a port-6543 URL.

**Why.** `CREATE TABLE IF NOT EXISTS` on connect is fine for one process against one
file, and races silently once two containers start together. The advisory lock fixes
that — but only on a session connection. pgBouncer in transaction mode hands the
connection back after every statement, so the lock would be acquired and released
immediately and the protection would be imaginary. Refusing is the only honest option.

**Consequence.** The same list generates `deploy/supabase/001_init.sql` and
`supabase/migrations/`, and a test asserts the committed file matches the code. Three
copies of a schema that can drift is two too many.

---

## 34. A planned metric produces a number or names what it is missing

**Decision.** Every metric in the plan appears in the result file: as a score, or in
`unmeasured` with the reason and the one line that would fix it. `livingeval gate`
reports the second group BLIND rather than passing them.

**Why.** A metric with no data does not fail. It disappears — no error, no zero, no
row — and the report that remains is a report on whatever happened to have data. A
live run of a twenty-two-metric suite came back green with sixteen metrics in it and
nothing anywhere saying so. That is worse than a failing suite, because a failing suite
gets looked at.

**Consequence.** Reference-free metrics needed inputs they did not have, so they get
generated ones. Telemetry metrics needed the run rather than their own pipeline, so
they are computed once at the end. What genuinely cannot be measured — time to first
token without a streamed call — is named every run.

---

## 35. Generated inputs are marked synthetic and excluded from coverage

**Decision.** Inputs this tool writes — safety probes and the input sets that
reference-free metrics run on — carry `kind: "synthetic"` and do not count towards
`drift` coverage unless asked for.

**Why.** Coverage answers "does this suite still describe your traffic". Cases the tool
invented describe the tool. Counting them inflates the number in exactly the situation
it exists to detect: a fresh suite, no real cases yet, and a coverage figure that says
everything is fine.

**Consequence.** A generated suite reports low coverage on day one, which is correct
and is the prompt to replace the inputs with real questions. `--include-probes`
overrides it and the report says what that costs.

---

## 36. Promotion carries the question, not the answer

**Decision.** A confirmed review-queue case is written into the golden sets as a real
input. Its `expected` is filled in only if the reviewer typed one; a plain confirmation
leaves the answer owed and the set incomplete.

**Why.** A reviewer marking an output acceptable has judged what the system said. That
is not the answer a domain expert would give, and the two come apart on exactly the
cases worth having: a plausible, agreeable, subtly wrong answer is the one that passes
review. Promoting it as the reference would encode the failure as the standard — the
same circularity as [#10](#10-mined-cases-require-an-explicit-named-confirmation),
one step further along.

**Consequence.** Promotion improves the inputs immediately and the answers only when
somebody writes them. A confirmed *failure* is the most valuable row it produces: a
real question known to break the system, waiting on the answer it should have given.
