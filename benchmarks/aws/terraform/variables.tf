variable "region" {
  description = "AWS region. us-east-2 only: an Organizations SCP denies this account's EC2 API elsewhere."
  type        = string
  default     = "us-east-2"
}

variable "name" {
  description = "Name prefix for every resource this configuration owns."
  type        = string
  default     = "rust-bench"
}

variable "log_retention_days" {
  description = "How long a run's log is kept before it expires."
  type        = number
  default     = 7
}

variable "image" {
  description = "Container image. The toolchains are installed at run time, so a plain base image is enough."
  type        = string
  default     = "public.ecr.aws/amazonlinux/amazonlinux:2023"
}

variable "task_cpu" {
  description = <<-EOT
    Task CPU units (1024 = 1 vCPU). Fargate accepts only 256, 512, 1024, 2048, 4096, 8192
    and 16384, and the account's Fargate on-demand vCPU quota is 12, so 8192 is the
    largest single task until the quota increase is granted.
  EOT
  type        = number
  default     = 8192
}

variable "task_memory" {
  description = "Task memory in MiB. At 8192 CPU units Fargate accepts 16384-61440."
  type        = number
  default     = 16384
}

variable "ephemeral_gib" {
  description = "Ephemeral storage for the repo, toolchains, virtualenv and cargo target directory."
  type        = number
  default     = 50
}

variable "stage" {
  description = "Benchmark stage the task runs: routine, spread, or grid."
  type        = string
  default     = "routine"

  validation {
    condition     = contains(["routine", "spread", "grid"], var.stage)
    error_message = "stage must be one of: routine, spread, grid."
  }
}

variable "git_ref" {
  description = "Branch or commit of climate_indices to benchmark."
  type        = string
  default     = "main"
}

variable "repeats" {
  description = "Repetitions per backend. Ignored by the routine stage's own default when zero."
  type        = number
  default     = 15
}

variable "cpus" {
  description = "CPU set the measured process is pinned to, as taskset accepts it."
  type        = string
  default     = "0-7"
}

variable "schedule_expression" {
  description = "Optional EventBridge Scheduler expression for a periodic run, for example 'rate(7 days)'. Null disables it."
  type        = string
  default     = null
}

variable "netcdf" {
  description = "Prepared precipitation grid path inside the task, grid stage only."
  type        = string
  default     = ""
}

variable "tavg" {
  description = "Matching mean-temperature grid path inside the task, grid stage only."
  type        = string
  default     = ""
}
