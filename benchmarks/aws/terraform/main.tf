terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = var.region
}

data "aws_caller_identity" "current" {}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }

  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

resource "aws_ecs_cluster" "bench" {
  name = var.name

  # Container Insights is billed and adds nothing to a benchmark's result.
  setting {
    name  = "containerInsights"
    value = "disabled"
  }
}

# Created here rather than by the task, because awslogs-create-group needs
# logs:CreateLogGroup, which the managed execution-role policy does not grant.
resource "aws_cloudwatch_log_group" "bench" {
  name              = "/ecs/${var.name}"
  retention_in_days = var.log_retention_days
}

data "aws_iam_policy_document" "ecs_tasks_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# Pulls the image and delivers the container's stdout to CloudWatch Logs.
resource "aws_iam_role" "execution" {
  name               = "${var.name}-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# The container needs no AWS API access today: it clones a public repository and
# prints its result. A distinct role keeps that explicit and leaves room for
# fixture or artifact buckets later.
resource "aws_iam_role" "task" {
  name               = "${var.name}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

# Egress only. The task clones and downloads over the internet and accepts no
# inbound connection, so no ingress rule exists.
resource "aws_security_group" "task" {
  name        = "${var.name}-task"
  description = "Egress-only group for the benchmark task"
  vpc_id      = data.aws_vpc.default.id

  egress {
    description = "toolchain and repository downloads"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_ecs_task_definition" "bench" {
  family                   = var.name
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = tostring(var.task_cpu)
  memory                   = tostring(var.task_memory)
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  runtime_platform {
    cpu_architecture        = "X86_64"
    operating_system_family = "LINUX"
  }

  ephemeral_storage {
    size_in_gib = var.ephemeral_gib
  }

  container_definitions = jsonencode([
    {
      name      = "bench"
      image     = var.image
      essential = true
      command   = ["bash", "-c", file("${path.module}/../bootstrap.sh")]
      environment = [
        { name = "STAGE", value = var.stage },
        { name = "GIT_REF", value = var.git_ref },
        { name = "REPEATS", value = tostring(var.repeats) },
        { name = "CPUS", value = var.cpus },
        { name = "NETCDF", value = var.netcdf },
        { name = "TAVG", value = var.tavg },
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.bench.name
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = var.stage
        }
      }
    }
  ])
}

# Opt-in periodic run. EventBridge Scheduler needs its own role, because starting
# a Fargate task passes the task's execution role to the ECS agent.
resource "aws_iam_role" "scheduler" {
  count = var.schedule_expression == null ? 0 : 1

  name = "${var.name}-scheduler"

  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Action = "sts:AssumeRole", Principal = { Service = "scheduler.amazonaws.com" } }]
  })
}

resource "aws_iam_role_policy" "scheduler" {
  count = var.schedule_expression == null ? 0 : 1

  name = "${var.name}-scheduler"
  role = aws_iam_role.scheduler[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "ecs:RunTask"
        Resource = aws_ecs_task_definition.bench.arn_without_revision
      },
      {
        Effect   = "Allow"
        Action   = "iam:PassRole"
        Resource = [aws_iam_role.execution.arn, aws_iam_role.task.arn]
      },
    ]
  })
}

resource "aws_scheduler_schedule" "bench" {
  count = var.schedule_expression == null ? 0 : 1

  name                = "${var.name}-schedule"
  schedule_expression = var.schedule_expression

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = aws_ecs_cluster.bench.arn
    role_arn = aws_iam_role.scheduler[0].arn

    ecs_parameters {
      task_definition_arn = aws_ecs_task_definition.bench.arn
      launch_type         = "FARGATE"
      task_count          = 1

      network_configuration {
        subnets          = data.aws_subnets.default.ids
        security_groups  = [aws_security_group.task.id]
        assign_public_ip = true
      }
    }
  }
}

# ---------------------------------------------------------------------------
# Optional EC2 instance for grid-scale runs.
#
# Fargate caps a single task at 8 vCPU under this account's quota and is not the
# right shape for a grid that needs tens of gigabytes, so the large-memory work
# runs here instead. Access is SSM-only: there is no key pair and the security
# group has no ingress rule, because Run Command and Session Manager are
# outbound connections from the instance.
# ---------------------------------------------------------------------------

data "aws_ssm_parameter" "al2023" {
  name = var.ec2_ami_ssm_parameter
}

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "instance" {
  count = var.ec2_instance_type == null ? 0 : 1

  name               = "${var.name}-instance"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

resource "aws_iam_role_policy_attachment" "instance_ssm" {
  count = var.ec2_instance_type == null ? 0 : 1

  role       = aws_iam_role.instance[0].name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "instance" {
  count = var.ec2_instance_type == null ? 0 : 1

  name = "${var.name}-instance"
  role = aws_iam_role.instance[0].name
}

resource "aws_security_group" "instance" {
  count = var.ec2_instance_type == null ? 0 : 1

  name        = "${var.name}-instance"
  description = "Egress-only group for the benchmark instance; SSM needs no ingress"
  vpc_id      = data.aws_vpc.default.id

  egress {
    description = "toolchain, repository and fixture downloads"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "bench" {
  count = var.ec2_instance_type == null ? 0 : 1

  ami                         = data.aws_ssm_parameter.al2023.value
  instance_type               = var.ec2_instance_type
  iam_instance_profile        = aws_iam_instance_profile.instance[0].name
  subnet_id                   = tolist(data.aws_subnets.default.ids)[0]
  vpc_security_group_ids      = [aws_security_group.instance[0].id]
  associate_public_ip_address = true

  # The watchdog in run_ec2.sh powers the instance off; this makes that final.
  # Without it an abandoned instance only stops and keeps billing its volumes.
  instance_initiated_shutdown_behavior = "terminate"

  metadata_options {
    http_tokens = "required"
  }

  root_block_device {
    volume_size           = var.ec2_root_gib
    volume_type           = "gp3"
    delete_on_termination = true
  }

  tags = {
    Name = "${var.name}-instance"
  }
}
