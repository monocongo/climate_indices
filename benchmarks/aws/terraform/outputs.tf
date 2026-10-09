output "cluster_name" {
  description = "ECS cluster to start a benchmark task in."
  value       = aws_ecs_cluster.bench.name
}

output "task_definition_arn" {
  description = "Task definition to run, including the current revision."
  value       = aws_ecs_task_definition.bench.arn
}

output "log_group_name" {
  description = "Log group holding every task's stdout."
  value       = aws_cloudwatch_log_group.bench.name
}

output "subnet_ids" {
  description = "Default-VPC subnets the task runs in."
  value       = data.aws_subnets.default.ids
}

output "security_group_id" {
  description = "Egress-only security group attached to the task."
  value       = aws_security_group.task.id
}

output "account_id" {
  description = "Account the benchmark resources live in."
  value       = data.aws_caller_identity.current.account_id
}

output "instance_id" {
  description = "Benchmark EC2 instance, or empty when none is provisioned."
  value       = length(aws_instance.bench) > 0 ? aws_instance.bench[0].id : ""
}
