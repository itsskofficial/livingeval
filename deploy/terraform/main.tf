###############################################################################
# livingeval on AWS: ECR + ECS Fargate + ALB + S3 + Secrets Manager + budget
#
#   cd deploy/terraform
#   terraform init
#   terraform apply -var="database_url=postgresql://...supabase..." \
#                   -var="api_key=$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"
#
# What this creates and roughly what it costs on demand pricing in ap-south-1:
#
#   ALB                       ~$18/month   the only unavoidable fixed cost
#   Fargate api  (0.5 vCPU)   ~$15/month
#   Fargate worker (0.25)     ~$8/month
#   ECR + S3 + logs           ~$2/month
#                             ----------
#                             ~$43/month, i.e. about 0.4% of the AWS credit per month
#
# Two things this deliberately does NOT create:
#
#   * A database. The default is Supabase, which is a credit you already have and one
#     less thing to pay for. `var.create_rds = true` adds an RDS instance if you would
#     rather keep it all in AWS — it adds roughly $15/month.
#   * A NAT gateway. It is ~$32/month and this workload does not need one: tasks run in
#     public subnets with public IPs, locked down by a security group that only accepts
#     traffic from the load balancer. That is the standard trade for a small service and
#     it is worth knowing you are making it.
###############################################################################

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = {
      Project     = "livingeval"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

data "aws_caller_identity" "current" {}
data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  name = "${var.project}-${var.environment}"

  # Supabase's transaction pooler, assembled from the project ref when a full URL was
  # not supplied. `urlencode` on the password matters: Supabase-generated passwords
  # routinely contain characters that break a naively concatenated URL.
  supabase_url = var.supabase_project_ref == "" ? "" : format(
    "postgresql://postgres.%s:%s@aws-0-%s.pooler.supabase.com:6543/postgres",
    var.supabase_project_ref,
    urlencode(var.supabase_db_password),
    var.supabase_region,
  )

  # Deliberately not `coalesce`: it raises when every argument is empty, and that is
  # exactly the case where you forgot to supply a database. Its error names `coalesce`
  # rather than the mistake, so the check below does it instead.
  resolved_database_url = (
    var.create_rds ? local.rds_url :
    var.database_url != "" ? var.database_url :
    local.supabase_url
  )

  artifact_url = var.artifacts_to_supabase ? "supabase://${var.artifacts_bucket}/livingeval" : "s3://${aws_s3_bucket.artifacts.bucket}/artifacts"

  # Configuration that is not secret goes in the task definition as plain env vars;
  # anything secret goes through Secrets Manager and arrives as a `secrets` entry, so it
  # never appears in `describe-task-definition` output or in CloudTrail.
  common_env = {
    LIVINGEVAL_ENV          = var.environment
    LIVINGEVAL_HOST         = "0.0.0.0"
    LIVINGEVAL_PORT         = "8000"
    LIVINGEVAL_LOG_JSON     = "1"
    LIVINGEVAL_LOG_LEVEL    = var.log_level
    LIVINGEVAL_SUITE        = var.suite_name
    LIVINGEVAL_JUDGE        = var.judge_spec
    LIVINGEVAL_EMBED        = var.embed_spec
    LIVINGEVAL_ARTIFACT_URL = local.artifact_url
    SUPABASE_URL            = var.supabase_url
    LIVINGEVAL_TRACE_LIMIT  = tostring(var.trace_limit)
    LIVINGEVAL_WORKER_MINE  = tostring(var.worker_mine)
    AWS_REGION              = var.region
  }
}

###############################################################################
# networking - a small VPC with public subnets only
###############################################################################

resource "aws_vpc" "main" {
  cidr_block           = "10.20.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = { Name = local.name }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = local.name }
}

# Two AZs because an ALB requires subnets in at least two.
resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  tags                    = { Name = "${local.name}-public-${count.index}" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  tags = { Name = "${local.name}-public" }
}

resource "aws_route_table_association" "public" {
  count          = length(aws_subnet.public)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_security_group" "alb" {
  name        = "${local.name}-alb"
  description = "Public ingress to the load balancer"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidrs
  }

  dynamic "ingress" {
    for_each = var.certificate_arn == "" ? [] : [1]
    content {
      description = "HTTPS"
      from_port   = 443
      to_port     = 443
      protocol    = "tcp"
      cidr_blocks = var.allowed_cidrs
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "tasks" {
  name        = "${local.name}-tasks"
  description = "Fargate tasks: only the ALB may reach the app port"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "App port, from the load balancer only"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  # Outbound is open because the tasks pull images from ECR, write to S3, and reach
  # Supabase and any judge API. Tightening this means VPC endpoints, which cost more
  # than the exposure is worth here.
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

###############################################################################
# ECR
###############################################################################

resource "aws_ecr_repository" "app" {
  name                 = local.name
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Untagged images accumulate on every push and are billed per GB.
resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the 10 most recent images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 10 }
      action       = { type = "expire" }
    }]
  })
}

###############################################################################
# S3 for artifacts (records, figures, exported datasets)
###############################################################################

resource "random_id" "suffix" {
  byte_length = 4
}

resource "aws_s3_bucket" "artifacts" {
  bucket        = "${local.name}-artifacts-${random_id.suffix.hex}"
  force_destroy = var.environment != "prod"
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

# The worker writes a timestamped record every tick. Without expiry that grows forever
# and the bill grows with it.
resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    id     = "expire-old-versions"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

###############################################################################
# secrets
###############################################################################

# Fail at plan time with a sentence that says what to do, rather than at apply time
# with a type error, or at runtime with a task that crash-loops on an empty DSN.
resource "terraform_data" "database_url_required" {
  lifecycle {
    precondition {
      condition     = local.resolved_database_url != ""
      error_message = <<-EOT
        No database configured. Pick one:

          Supabase (recommended - it is a credit you already have):
            -var="supabase_project_ref=<ref>" -var="supabase_db_password=<password>"
          or paste the transaction-pooler URL directly:
            -var="database_url=postgresql://postgres.<ref>:<pw>@aws-0-<region>.pooler.supabase.com:6543/postgres"
          or let Terraform create an RDS instance (~$15/month):
            -var="create_rds=true"
      EOT
    }
  }
}

resource "aws_secretsmanager_secret" "app" {
  name                    = "${local.name}-config"
  description             = "livingeval database URL, API key and judge credentials"
  recovery_window_in_days = 0 # so `terraform destroy` really removes it
}

resource "aws_secretsmanager_secret_version" "app" {
  secret_id = aws_secretsmanager_secret.app.id
  secret_string = jsonencode(merge(
    {
      LIVINGEVAL_DATABASE_URL = local.resolved_database_url
      LIVINGEVAL_API_KEY      = var.api_key
    },
    var.supabase_service_role_key == "" ? {} : {
      SUPABASE_SERVICE_ROLE_KEY = var.supabase_service_role_key
    },
    var.openai_api_key == "" ? {} : { OPENAI_API_KEY = var.openai_api_key },
    var.anthropic_api_key == "" ? {} : { ANTHROPIC_API_KEY = var.anthropic_api_key },
    var.langfuse_public_key == "" ? {} : {
      LANGFUSE_PUBLIC_KEY = var.langfuse_public_key
      LANGFUSE_SECRET_KEY = var.langfuse_secret_key
      LANGFUSE_HOST       = var.langfuse_host
    },
  ))
}

###############################################################################
# optional RDS, when you would rather not use Supabase
###############################################################################

resource "random_password" "rds" {
  count   = var.create_rds ? 1 : 0
  length  = 32
  special = false
}

resource "aws_db_subnet_group" "main" {
  count      = var.create_rds ? 1 : 0
  name       = local.name
  subnet_ids = aws_subnet.public[*].id
}

resource "aws_security_group" "rds" {
  count       = var.create_rds ? 1 : 0
  name        = "${local.name}-rds"
  description = "Postgres, reachable from the tasks only"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.tasks.id]
  }
}

resource "aws_db_instance" "main" {
  count                        = var.create_rds ? 1 : 0
  identifier                   = local.name
  engine                       = "postgres"
  engine_version               = "16"
  instance_class               = var.rds_instance_class
  allocated_storage            = 20
  storage_encrypted            = true
  db_name                      = "livingeval"
  username                     = "livingeval"
  password                     = random_password.rds[0].result
  db_subnet_group_name         = aws_db_subnet_group.main[0].name
  vpc_security_group_ids       = [aws_security_group.rds[0].id]
  skip_final_snapshot          = true
  publicly_accessible          = false
  backup_retention_period      = 7
  auto_minor_version_upgrade   = true
  performance_insights_enabled = false
}

variable "artifacts_bucket" {
  description = "Supabase Storage bucket, when artifacts_to_supabase = true."
  type        = string
  default     = "livingeval-artifacts"
}

locals {
  rds_url = var.create_rds ? format(
    "postgresql://livingeval:%s@%s/livingeval?sslmode=require",
    random_password.rds[0].result,
    aws_db_instance.main[0].endpoint,
  ) : ""
}
