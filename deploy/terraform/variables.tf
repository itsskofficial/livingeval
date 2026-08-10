###############################################################################
# Everything you can set. Only two have no default: `database_url` and `api_key`.
###############################################################################

variable "project" {
  description = "Prefix for every resource name."
  type        = string
  default     = "livingeval"
}

variable "environment" {
  description = "Environment name; part of every resource name and of LIVINGEVAL_ENV."
  type        = string
  default     = "prod"
}

variable "region" {
  description = "AWS region. ap-south-1 (Mumbai) is the low-latency choice from Pune."
  type        = string
  default     = "ap-south-1"
}

###############################################################################
# the two you must supply
###############################################################################

variable "database_url" {
  description = <<-EOT
    Supabase connection string for the **running service**: the transaction pooler,
    port 6543. It is IPv4 (the direct host is IPv6-only on the free tier, which a
    Fargate task in this stack cannot reach) and it survives connection churn.

    Dashboard -> Project Settings -> Database -> Connection string -> Transaction pooler.

    Leave empty and set supabase_project_ref + supabase_db_password instead, and the
    URL is built for you. Ignored when create_rds = true.
  EOT
  type        = string
  sensitive   = true
  default     = ""
}

variable "supabase_project_ref" {
  description = "Your Supabase project ref. With supabase_db_password, builds database_url."
  type        = string
  default     = ""
}

variable "supabase_db_password" {
  description = "Supabase database password. Dashboard -> Project Settings -> Database."
  type        = string
  sensitive   = true
  default     = ""
}

variable "supabase_region" {
  description = "Supabase pooler region, e.g. ap-south-1. Must match your project."
  type        = string
  default     = "ap-south-1"
}

variable "supabase_url" {
  description = "https://<ref>.supabase.co. Needed only if artifacts go to Supabase Storage."
  type        = string
  default     = ""
}

variable "supabase_service_role_key" {
  description = <<-EOT
    Supabase service-role key, for Storage. Bypasses RLS, so it is backend-only and
    never reaches a browser. Omit to keep artifacts in S3, which is the default here.
  EOT
  type        = string
  sensitive   = true
  default     = ""
}

variable "artifacts_to_supabase" {
  description = <<-EOT
    Write records and figures to Supabase Storage instead of S3. S3 is the default
    because the bucket is created here and the task role already reaches it with no
    extra credential.
  EOT
  type        = bool
  default     = false
}

variable "api_key" {
  description = <<-EOT
    Bearer token the API requires. Generate one with:
      python -c "import secrets; print(secrets.token_urlsafe(32))"
    The app refuses to bind non-locally without it.
  EOT
  type        = string
  sensitive   = true
}

###############################################################################
# optional credentials, all pulled from your YC credit pool
###############################################################################

variable "openai_api_key" {
  description = "For judge.openai(...) and OpenAI embeddings."
  type        = string
  sensitive   = true
  default     = ""
}

variable "anthropic_api_key" {
  description = "For judge.anthropic(...), useful as a second-judge ceiling."
  type        = string
  sensitive   = true
  default     = ""
}

variable "langfuse_public_key" {
  description = "For `livingeval ingest --follow` against a Langfuse project."
  type        = string
  sensitive   = true
  default     = ""
}

variable "langfuse_secret_key" {
  type      = string
  sensitive = true
  default   = ""
}

variable "langfuse_host" {
  type    = string
  default = "https://cloud.langfuse.com"
}

###############################################################################
# application configuration
###############################################################################

variable "suite_name" {
  type    = string
  default = "default"
}

variable "judge_spec" {
  description = "oracle | rule:mod:fn | openai:gpt-4o-mini | anthropic:claude-sonnet-4-5"
  type        = string
  default     = "oracle"
}

variable "embed_spec" {
  description = "Empty for tf-idf+SVD. Otherwise hashing | hf:<model> | openai:<model>."
  type        = string
  default     = ""
}

variable "trace_limit" {
  description = "Most recent traces each analysis reads. Bounds memory per task."
  type        = number
  default     = 20000
}

variable "log_level" {
  type    = string
  default = "info"
}

###############################################################################
# sizing - the defaults are deliberately small
###############################################################################

variable "image_tag" {
  type    = string
  default = "latest"
}

variable "api_cpu" {
  description = "Fargate CPU units. 512 = 0.5 vCPU."
  type        = string
  default     = "512"
}

variable "api_memory" {
  description = "MiB. scikit-learn plus a tf-idf matrix over 20k traces fits in 1 GB."
  type        = string
  default     = "1024"
}

variable "api_count" {
  type    = number
  default = 1
}

variable "worker_cpu" {
  type    = string
  default = "512"
}

variable "worker_memory" {
  type    = string
  default = "1024"
}

variable "worker_count" {
  description = "Keep at 1. Two workers would mine the same suite twice."
  type        = number
  default     = 1

  validation {
    condition     = var.worker_count <= 1
    error_message = "Run at most one worker: concurrent workers duplicate mined proposals."
  }
}

variable "worker_interval_s" {
  type    = number
  default = 900
}

variable "worker_mine" {
  description = "Proposals queued per tick. They wait for a human either way."
  type        = number
  default     = 20
}

###############################################################################
# database
###############################################################################

variable "create_rds" {
  description = "Create an RDS Postgres instead of using Supabase. Adds ~$15/month."
  type        = bool
  default     = false
}

variable "rds_instance_class" {
  type    = string
  default = "db.t4g.micro"
}

###############################################################################
# access, observability and cost
###############################################################################

variable "allowed_cidrs" {
  description = <<-EOT
    Who may reach the load balancer. Defaults to the whole internet because the API
    key is the real control, but narrowing this to your own IP is strictly better
    while you are developing.
  EOT
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "certificate_arn" {
  description = "ACM certificate ARN. Empty means HTTP only - fine for a demo, not for real traffic."
  type        = string
  default     = ""
}

variable "log_retention_days" {
  type    = number
  default = 14
}

variable "container_insights" {
  description = "Container Insights costs extra per metric. Off by default."
  type        = bool
  default     = false
}

variable "monthly_budget_usd" {
  description = <<-EOT
    A budget alarm, because AWS Activate credits do not stop the service when they
    expire - they start billing the card on file. Set the alert address and pay
    attention to it.
  EOT
  type        = number
  default     = 100
}

variable "budget_alert_email" {
  description = "Where budget alerts go. Empty disables the budget."
  type        = string
  default     = ""
}
