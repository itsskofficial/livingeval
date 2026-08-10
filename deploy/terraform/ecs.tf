###############################################################################
# ECS Fargate: the API service behind an ALB, and the worker with no ingress.
###############################################################################

resource "aws_ecs_cluster" "main" {
  name = local.name

  setting {
    name  = "containerInsights"
    value = var.container_insights ? "enabled" : "disabled"
  }
}

resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${local.name}"
  retention_in_days = var.log_retention_days
}

###############################################################################
# IAM
###############################################################################

# The execution role is what ECS itself uses: pull the image, read the secret, write
# logs. It is not the role the application code runs as.
resource "aws_iam_role" "execution" {
  name = "${local.name}-execution"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "execution_managed" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "execution_secrets" {
  name = "read-config-secret"
  role = aws_iam_role.execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = [aws_secretsmanager_secret.app.arn]
    }]
  })
}

# The task role is what the application code runs as. Scoped to exactly the artifact
# bucket — an application that can read every bucket in the account is one bug away
# from being a data exfiltration path.
resource "aws_iam_role" "task" {
  name = "${local.name}-task"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "task_s3" {
  name = "artifacts-bucket"
  role = aws_iam_role.task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
        Resource = ["${aws_s3_bucket.artifacts.arn}/*"]
      },
      {
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = [aws_s3_bucket.artifacts.arn]
      },
    ]
  })
}

###############################################################################
# task definitions
###############################################################################

locals {
  image = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"

  env_list = [for k, v in local.common_env : { name = k, value = tostring(v) } if v != ""]

  # Every secret arrives by reference, resolved by the ECS agent at start-up. The value
  # never appears in the task definition, so `describe-task-definition` is safe to paste
  # into an issue.
  secret_list = concat(
    [
      { name = "LIVINGEVAL_DATABASE_URL", valueFrom = "${aws_secretsmanager_secret.app.arn}:LIVINGEVAL_DATABASE_URL::" },
      { name = "LIVINGEVAL_API_KEY", valueFrom = "${aws_secretsmanager_secret.app.arn}:LIVINGEVAL_API_KEY::" },
    ],
    var.openai_api_key == "" ? [] : [
      { name = "OPENAI_API_KEY", valueFrom = "${aws_secretsmanager_secret.app.arn}:OPENAI_API_KEY::" },
    ],
    var.anthropic_api_key == "" ? [] : [
      { name = "ANTHROPIC_API_KEY", valueFrom = "${aws_secretsmanager_secret.app.arn}:ANTHROPIC_API_KEY::" },
    ],
    var.supabase_service_role_key == "" ? [] : [
      { name = "SUPABASE_SERVICE_ROLE_KEY", valueFrom = "${aws_secretsmanager_secret.app.arn}:SUPABASE_SERVICE_ROLE_KEY::" },
    ],
    var.langfuse_public_key == "" ? [] : [
      { name = "LANGFUSE_PUBLIC_KEY", valueFrom = "${aws_secretsmanager_secret.app.arn}:LANGFUSE_PUBLIC_KEY::" },
      { name = "LANGFUSE_SECRET_KEY", valueFrom = "${aws_secretsmanager_secret.app.arn}:LANGFUSE_SECRET_KEY::" },
      { name = "LANGFUSE_HOST", valueFrom = "${aws_secretsmanager_secret.app.arn}:LANGFUSE_HOST::" },
    ],
  )

  log_config = {
    logDriver = "awslogs"
    options = {
      "awslogs-group"         = aws_cloudwatch_log_group.app.name
      "awslogs-region"        = var.region
      "awslogs-stream-prefix" = "ecs"
    }
  }
}

resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.api_cpu
  memory                   = var.api_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([{
    name             = "api"
    image            = local.image
    essential        = true
    environment      = local.env_list
    secrets          = local.secret_list
    portMappings     = [{ containerPort = 8000, protocol = "tcp" }]
    logConfiguration = local.log_config
    healthCheck = {
      command     = ["CMD-SHELL", "curl -fsS http://127.0.0.1:8000/healthz || exit 1"]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 30
    }
    # Two workers per task: enough to keep a core busy on an analysis request while
    # /v1/score stays responsive, without each replica holding four copies of a model.
    command = ["uvicorn", "livingeval.serve.asgi:app", "--host", "0.0.0.0",
    "--port", "8000", "--workers", "2", "--timeout-keep-alive", "65"]
  }])
}

resource "aws_ecs_task_definition" "worker" {
  family                   = "${local.name}-worker"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.worker_cpu
  memory                   = var.worker_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([{
    name             = "worker"
    image            = local.image
    essential        = true
    environment      = local.env_list
    secrets          = local.secret_list
    logConfiguration = local.log_config
    command = ["livingeval", "worker", "--interval", tostring(var.worker_interval_s),
    "--mine", tostring(var.worker_mine)]
  }])
}

# Migrations as a task you run once, not something the service does at start-up. Run it
# with `aws ecs run-task`; the deploy workflow does exactly that before rolling the
# service.
resource "aws_ecs_task_definition" "migrate" {
  family                   = "${local.name}-migrate"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "256"
  memory                   = "512"
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([{
    name             = "migrate"
    image            = local.image
    essential        = true
    environment      = local.env_list
    secrets          = local.secret_list
    logConfiguration = local.log_config
    command          = ["livingeval", "db", "migrate"]
  }])
}

###############################################################################
# load balancer
###############################################################################

resource "aws_lb" "main" {
  name               = substr(local.name, 0, 32)
  load_balancer_type = "application"
  subnets            = aws_subnet.public[*].id
  security_groups    = [aws_security_group.alb.id]
  idle_timeout       = 120 # analysis endpoints can take a while
}

resource "aws_lb_target_group" "api" {
  name        = substr("${local.name}-api", 0, 32)
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    path                = "/healthz"
    matcher             = "200"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }

  # Let an in-flight analysis request finish before the target is removed.
  deregistration_delay = 30
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  # With a certificate, plain HTTP redirects instead of serving. Without one it serves,
  # which is fine for a demo behind an API key and is not fine for real traffic.
  dynamic "default_action" {
    for_each = var.certificate_arn == "" ? [1] : []
    content {
      type             = "forward"
      target_group_arn = aws_lb_target_group.api.arn
    }
  }

  dynamic "default_action" {
    for_each = var.certificate_arn == "" ? [] : [1]
    content {
      type = "redirect"
      redirect {
        port        = "443"
        protocol    = "HTTPS"
        status_code = "HTTP_301"
      }
    }
  }
}

resource "aws_lb_listener" "https" {
  count             = var.certificate_arn == "" ? 0 : 1
  load_balancer_arn = aws_lb.main.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

###############################################################################
# services
###############################################################################

resource "aws_ecs_service" "api" {
  name            = "${local.name}-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.api_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.tasks.id]
    assign_public_ip = true # no NAT gateway; see the note in main.tf
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }

  health_check_grace_period_seconds = 60

  # A rolling deploy that keeps 100% up and can double briefly: with one replica that
  # means zero-downtime, and with more it means a bad image is caught before the old
  # one is gone.
  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  # CI updates the task definition; Terraform should not fight it on the next apply.
  lifecycle {
    ignore_changes = [task_definition, desired_count]
  }

  depends_on = [aws_lb_listener.http]
}

resource "aws_ecs_service" "worker" {
  name            = "${local.name}-worker"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.worker.arn
  desired_count   = var.worker_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.tasks.id]
    assign_public_ip = true
  }

  # Exactly one worker, replaced rather than doubled: two workers mining the same suite
  # at once would queue the same proposals twice.
  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100

  lifecycle {
    ignore_changes = [task_definition, desired_count]
  }
}
