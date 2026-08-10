###############################################################################
# Budget and alarms
#
# The portfolio plan says it explicitly: after expiry, AWS does not shut services
# down, it bills the card. A budget you actually receive email about is the cheapest
# insurance available, so it is created by default rather than left as an exercise.
###############################################################################

resource "aws_budgets_budget" "monthly" {
  count        = var.budget_alert_email == "" ? 0 : 1
  name         = "${local.name}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  # 50% is a nudge, 100% actual is a problem, 100% forecast is the one that arrives in
  # time to do something about it.
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 50
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_alert_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_alert_email]
  }
}

resource "aws_cloudwatch_metric_alarm" "api_5xx" {
  alarm_name          = "${local.name}-api-5xx"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "HTTPCode_Target_5XX_Count"
  namespace           = "AWS/ApplicationELB"
  period              = 300
  statistic           = "Sum"
  threshold           = 5
  alarm_description   = "The API is returning server errors."
  treat_missing_data  = "notBreaching"

  dimensions = {
    LoadBalancer = aws_lb.main.arn_suffix
    TargetGroup  = aws_lb_target_group.api.arn_suffix
  }
}

resource "aws_cloudwatch_metric_alarm" "no_healthy_targets" {
  alarm_name          = "${local.name}-no-healthy-targets"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 2
  metric_name         = "HealthyHostCount"
  namespace           = "AWS/ApplicationELB"
  period              = 60
  statistic           = "Average"
  threshold           = 1
  alarm_description   = "Nothing is serving. Usually a bad image or an unreachable database."
  treat_missing_data  = "breaching"

  dimensions = {
    LoadBalancer = aws_lb.main.arn_suffix
    TargetGroup  = aws_lb_target_group.api.arn_suffix
  }
}

###############################################################################
# Outputs - everything you need to use or hand to CI
###############################################################################

output "url" {
  description = "The dashboard. Add /review and /docs."
  value       = var.certificate_arn == "" ? "http://${aws_lb.main.dns_name}" : "https://${aws_lb.main.dns_name}"
}

output "ecr_repository_url" {
  description = "Push images here. The deploy workflow needs it."
  value       = aws_ecr_repository.app.repository_url
}

output "ecs_cluster" {
  value = aws_ecs_cluster.main.name
}

output "ecs_api_service" {
  value = aws_ecs_service.api.name
}

output "ecs_worker_service" {
  value = aws_ecs_service.worker.name
}

output "migrate_task_definition" {
  description = "Run before rolling the service: aws ecs run-task --task-definition <this>"
  value       = aws_ecs_task_definition.migrate.family
}

output "artifacts_bucket" {
  description = "Records, figures and exported datasets land here."
  value       = aws_s3_bucket.artifacts.bucket
}

output "secret_arn" {
  value = aws_secretsmanager_secret.app.arn
}

output "log_group" {
  value = aws_cloudwatch_log_group.app.name
}

output "task_security_group" {
  description = "Add this to the Supabase network restrictions if you enable them."
  value       = aws_security_group.tasks.id
}

output "region" {
  value = var.region
}

output "next_steps" {
  description = "What to run once terraform apply finishes."
  value       = <<-EOT

    1. Build and push the image:
         aws ecr get-login-password --region ${var.region} \
           | docker login --username AWS --password-stdin ${aws_ecr_repository.app.repository_url}
         docker build -f deploy/Dockerfile -t ${aws_ecr_repository.app.repository_url}:latest .
         docker push ${aws_ecr_repository.app.repository_url}:latest

    2. Run the migration task once:
         aws ecs run-task --cluster ${aws_ecs_cluster.main.name} \
           --task-definition ${aws_ecs_task_definition.migrate.family} \
           --launch-type FARGATE \
           --network-configuration 'awsvpcConfiguration={subnets=[${join(",", aws_subnet.public[*].id)}],securityGroups=[${aws_security_group.tasks.id}],assignPublicIp=ENABLED}'

    3. Roll the services:
         aws ecs update-service --cluster ${aws_ecs_cluster.main.name} --service ${aws_ecs_service.api.name} --force-new-deployment
         aws ecs update-service --cluster ${aws_ecs_cluster.main.name} --service ${aws_ecs_service.worker.name} --force-new-deployment

    4. Check it:
         curl ${var.certificate_arn == "" ? "http://${aws_lb.main.dns_name}" : "https://${aws_lb.main.dns_name}"}/healthz
         curl -H "Authorization: Bearer $LIVINGEVAL_API_KEY" \
              ${var.certificate_arn == "" ? "http://${aws_lb.main.dns_name}" : "https://${aws_lb.main.dns_name}"}/v1/status

    Logs:  aws logs tail ${aws_cloudwatch_log_group.app.name} --follow
  EOT
}
