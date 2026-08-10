# Deployment

Local Supabase → Supabase Cloud → AWS. **The only thing that changes is credentials.**

Same Postgres, same pgBouncer semantics, same Storage API, same row-level security,
same migrations, same container. That is the whole point of running the real Supabase
locally instead of SQLite: "it worked locally" carries weight.

---

## What you actually need

Nothing on this list is required to run the whole thing locally.

| | needed for | where you get it | when |
|---|---|---|---|
| **Docker Desktop** | the local Supabase stack | already installed | now |
| **Node** (for `npx supabase`) | the Supabase CLI | already installed | now |
| **Supabase Cloud project** | the deployed database | supabase.com — a YC credit | before AWS |
| **`SUPABASE_DB_PASSWORD`** | connecting to it | Dashboard → Project Settings → Database | before AWS |
| **`SUPABASE_SERVICE_ROLE_KEY`** | Supabase Storage | Dashboard → Project Settings → API | only if artifacts go to Supabase |
| **AWS account + credentials** | the infrastructure | AWS Activate — a YC credit | at deploy |
| **`OPENAI_API_KEY`** | `judge.openai(...)`, OpenAI embeddings | platform.openai.com | only for a real-judge run |
| **`ANTHROPIC_API_KEY`** | a second-judge ceiling | console.anthropic.com | optional |
| **`LANGFUSE_*`** | `ingest --follow` | cloud.langfuse.com | optional |

Everything else — coverage, blind spots, power, the gate, the ladder, mining, review,
per-turn scoring, the worker — runs with none of them.

---

## 1. Local: the real Supabase, on your machine

```bash
pip install -e ".[dev,serve,postgres]"
python scripts/dev.py up
```

That starts Supabase, applies the migrations, turns on RLS, creates the private Storage
bucket, and seeds the drifting corpus. Then:

```bash
python scripts/dev.py serve      # dashboard on http://127.0.0.1:8000
python scripts/dev.py worker     # the measurement loop, in another terminal
python scripts/dev.py status     # what is running and what is in it
python scripts/dev.py env        # the exports, for your own shell
python scripts/dev.py test       # the Supabase-backed tests, against this stack
python scripts/dev.py down
```

| service | url |
|---|---|
| livingeval | http://127.0.0.1:8000 |
| Supabase Studio | http://127.0.0.1:54423 |
| Supabase API / Storage | http://127.0.0.1:54421 |
| Postgres | `postgresql://postgres:postgres@127.0.0.1:54422/postgres` |

**The ports are livingeval's own 544xx range, not Supabase's 543xx defaults**, so this
runs alongside any other Supabase project you have going. You already had one
(`capmatch-dev`) on the defaults; the collision error does not tell you which project
took the port, so avoiding it is worth the config line.

No key is needed. The local stack's service-role key is a published constant that only
signs against a local secret, so `scripts/dev.py` resolves it automatically.

### Containers against local Supabase

```bash
docker compose -f deploy/docker-compose.yml up --build
```

There is **no Postgres service in that compose file** on purpose. The database is
Supabase, and a plain `postgres:16` would be a different thing wearing the same name —
every Supabase-specific problem (pooler semantics, RLS, Storage) would go undetected
until deploy day.

---

## 2. Supabase Cloud

1. Create a project at [supabase.com](https://supabase.com). **Pick the region you will
   deploy into** — `ap-south-1` (Mumbai) from Pune. Cross-region adds latency to every
   query the analysis makes.
2. Copy the database password shown at creation. It is not shown again.
3. Apply the schema — either way works, they are generated from the same source:

   ```bash
   # From your machine, using the session pooler (port 5432):
   export LIVINGEVAL_DATABASE_URL=$(python -m livingeval.cli supabase url \
     --project-ref <ref> --password '<password>' --purpose session)
   livingeval db migrate
   livingeval db check
   ```

   Or paste `deploy/supabase/001_init.sql` into the Dashboard's SQL editor.

4. Create the Storage bucket and verify everything:

   ```bash
   export SUPABASE_PROJECT_REF=<ref>
   export SUPABASE_DB_PASSWORD='<password>'
   export SUPABASE_URL=https://<ref>.supabase.co
   export SUPABASE_SERVICE_ROLE_KEY=<service role key>
   livingeval supabase init
   ```

### Which of the three connection strings

Supabase gives you three and they are not interchangeable. `livingeval supabase url`
picks for you; this is what it is picking between.

| purpose | port | use it for | why |
|---|---|---|---|
| **transaction pooler** | 6543 | the running service | IPv4, survives connection churn. **This is what Fargate needs.** |
| **session pooler** | 5432 | migrations, `psql` | keeps session state, so advisory locks work |
| **direct** | 5432 | migrations, if you have IPv6 | IPv6-only on the free tier |

`livingeval db migrate` **refuses** the transaction pooler and says why: the advisory
lock that makes concurrent migrations safe needs a session, and pgBouncer in transaction
mode hands your connection back after every statement, so the lock would be taken and
released immediately and the protection would be imaginary.

### Security

Migration `0002_rls` enables row-level security on all six tables with **no policies**,
and revokes the `anon` and `authenticated` grants. These tables hold production traces —
real user text — and Supabase exposes `public` through PostgREST to the anon key that
ships in browsers. Verified, not assumed:

```
anon reading traces          -> 401 permission denied
anon downloading an artifact -> 404 (RLS hides it)
unauthenticated public read  -> 400 bucket not found (the bucket is private)
service role                 -> works
```

livingeval never uses the Data API. It connects over SQL as the database owner, so
denying PostgREST entirely costs nothing.

Optionally add the Fargate security group to **Database → Network Restrictions**;
`terraform output task_security_group` gives you the id.

---

## 3. AWS

```bash
cd deploy/terraform
terraform init
terraform apply \
  -var="supabase_project_ref=<ref>" \
  -var="supabase_db_password=<password>" \
  -var="supabase_url=https://<ref>.supabase.co" \
  -var="api_key=$(python -c 'import secrets;print(secrets.token_urlsafe(32))')" \
  -var="budget_alert_email=itsskofficial03@gmail.com" \
  -var="openai_api_key=$OPENAI_API_KEY"
```

Terraform prints the exact next commands. In short:

```bash
# build and push
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin $(terraform output -raw ecr_repository_url)
docker build -f deploy/Dockerfile -t $(terraform output -raw ecr_repository_url):latest ../..
docker push $(terraform output -raw ecr_repository_url):latest

# migrate, then roll
aws ecs run-task --cluster $(terraform output -raw ecs_cluster) --task-definition $(terraform output -raw migrate_task_definition) --launch-type FARGATE --network-configuration '...'
aws ecs update-service --cluster $(terraform output -raw ecs_cluster) --service $(terraform output -raw ecs_api_service) --force-new-deployment
```

Or push a `v*` tag and let `.github/workflows/deploy.yml` do it. That workflow
authenticates with **OIDC**, not a stored access key — GitHub gets a short-lived token
from STS, so there is no long-lived AWS secret in the repository.

### What gets created

```
VPC, 2 public subnets, IGW, 2 security groups
ECR repository            (keeps the 10 most recent images)
ECS Fargate cluster
  api service    x1       0.5 vCPU / 1 GB, behind an ALB, 2 uvicorn workers
  worker service x1       0.5 vCPU / 1 GB, no ingress
  migrate task            run once, before each roll
Application Load Balancer
S3 bucket                 artifacts: private, encrypted, versioned, 30-day expiry
Secrets Manager           one secret holding every credential
CloudWatch logs           14-day retention
Budget + 2 alarms         5xx rate, and no-healthy-targets
```

### Cost, on demand pricing in ap-south-1

| | per month |
|---|---|
| ALB | ~$18 — the only unavoidable fixed cost |
| Fargate api (0.5 vCPU) | ~$15 |
| Fargate worker (0.5 vCPU) | ~$15 |
| ECR + S3 + CloudWatch | ~$2 |
| **total** | **~$50**, about 0.5% of the AWS credit per month |

Deliberately **no NAT gateway** (~$32/month). Tasks run in public subnets with public
IPs and a security group that only accepts traffic from the load balancer. That is the
standard trade for a small service; it is worth knowing you are making it.

**Set the budget email.** AWS Activate credits do not shut services down when they
expire — they start billing the card on file. `budget_alert_email` creates alerts at
50% actual and 100% forecast. The forecast one is the useful one; it arrives in time.

### Two things to check on the first deploy

**Region match.** `region` (AWS) and `supabase_region` (the pooler) should be the same,
or every query pays a cross-region round trip and the analysis endpoints get slow.

**Migrations first.** The deploy workflow runs the migration task and **fails the
deployment if it exits non-zero**. Do not remove that: a service rolled onto a schema
that was not migrated fails in ways that look like data corruption.

---

## Configuration reference

Everything comes from the environment. `livingeval config --check` prints the resolved
settings with secrets redacted, and warns about deployment mistakes.

| variable | default | |
|---|---|---|
| `LIVINGEVAL_DATABASE_URL` | `sqlite:///livingeval.db` | any Postgres/Supabase URL |
| `LIVINGEVAL_ARTIFACT_URL` | `file://./results` | `s3://…`, `supabase://bucket/prefix` |
| `LIVINGEVAL_API_KEY` | none | **required** to bind non-locally |
| `LIVINGEVAL_HOST` / `_PORT` | `127.0.0.1` / `8000` | |
| `LIVINGEVAL_SUITE` | `default` | which suite this instance measures |
| `LIVINGEVAL_JUDGE` | `oracle` | `openai:gpt-4o-mini`, `rule:mod:fn`, … |
| `LIVINGEVAL_EMBED` | tf-idf+SVD | `hashing`, `hf:<model>`, `openai:<model>` |
| `LIVINGEVAL_LOG_JSON` | `0` | on in the container |
| `LIVINGEVAL_WORKER_INTERVAL_S` | `900` | |
| `LIVINGEVAL_DB_SSLMODE` | `require` off-localhost | override only for a plaintext private Postgres |
| `LIVINGEVAL_ALLOW_INSECURE` | `0` | binds publicly with no key. Deliberate. |

Full template in [`.env.example`](.env.example).

### The refusals

Three things the code will not do, each because the failure is otherwise silent:

1. **Bind to anything but localhost without an API key.** The platform is
   unauthenticated by design when local; on `0.0.0.0` that is a hole.
2. **Connect to a remote Postgres without TLS.** psycopg2 defaults to `sslmode=prefer`,
   which quietly accepts an unencrypted connection.
3. **Migrate over a transaction pooler.** The advisory lock would not be held.

Each raises with the fix in the message.

---

## Operating it

```bash
livingeval config --check                  # resolved settings + deployment warnings
livingeval db check                        # connection, migrations, row counts
livingeval supabase status                 # which Supabase, is it reachable
livingeval artifacts --prefix default      # what the worker has written
aws logs tail /ecs/livingeval-prod --follow
curl $URL/healthz                          # liveness, touches nothing external
curl $URL/readyz                           # readiness, pings the database
curl $URL/metrics                          # Prometheus text
```

`/healthz` deliberately does not touch the database: a liveness probe that fails on a
database blip gets a healthy process killed. `/readyz` is the one that checks it.

---

## Moving from local to cloud, concretely

The whole diff is environment variables:

```diff
- LIVINGEVAL_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:54422/postgres
+ LIVINGEVAL_DATABASE_URL=postgresql://postgres.<ref>:<pw>@aws-0-ap-south-1.pooler.supabase.com:6543/postgres
- SUPABASE_URL=http://127.0.0.1:54421
+ SUPABASE_URL=https://<ref>.supabase.co
- SUPABASE_SERVICE_ROLE_KEY=<the published local demo key>
+ SUPABASE_SERVICE_ROLE_KEY=<your project's service role key>
+ LIVINGEVAL_API_KEY=<a long random string>
+ LIVINGEVAL_HOST=0.0.0.0
```

No code changes, no schema changes, no different backend. That is the point.
