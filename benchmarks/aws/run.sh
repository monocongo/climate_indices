#!/usr/bin/env bash
# Provision the benchmark resources with Terraform and run one stage as a Fargate task.
#
# This script never creates AWS resources directly: Terraform owns everything,
# and the AWS CLI is used only to start an existing task definition and read its
# log back.
#
# Usage:
#   ./benchmarks/aws/run.sh [stage] [git-ref] [repeats]
#
#   stage    routine (default) | spread | grid
#   git-ref  branch or commit to benchmark (default: main)
#   repeats  repetitions per backend (default: 15)
#
# Results are written to benchmarks/aws/results/<stage>-<timestamp>.log
set -euo pipefail

stage="${1:-routine}"
git_ref="${2:-main}"
repeats="${3:-15}"
region="${AWS_REGION:-us-east-2}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tf="${here}/terraform"

cd "${tf}"
terraform init -input=false >/dev/null
terraform apply -input=false -auto-approve \
  -var "stage=${stage}" -var "git_ref=${git_ref}" -var "repeats=${repeats}"

cluster="$(terraform output -raw cluster_name)"
task_definition="$(terraform output -raw task_definition_arn)"
security_group="$(terraform output -raw security_group_id)"
log_group="$(terraform output -raw log_group_name)"
subnets_json="$(terraform output -json subnet_ids)"

network_configuration="$(python3 -c '
import json, sys
print(json.dumps({"awsvpcConfiguration": {
    "subnets": json.loads(sys.argv[1]),
    "securityGroups": [sys.argv[2]],
    "assignPublicIp": "ENABLED",
}}))
' "${subnets_json}" "${security_group}")"

echo "starting ${stage} stage (ref ${git_ref}, repeats ${repeats}) on ${cluster}"
task="$(aws ecs run-task --region "${region}" --cluster "${cluster}" \
  --task-definition "${task_definition}" --launch-type FARGATE --count 1 \
  --network-configuration "${network_configuration}" \
  --query 'tasks[0].taskArn' --output text)"

if [[ "${task}" != arn:* ]]; then
  echo "run-task did not start a task; check the Fargate vCPU quota:" >&2
  aws ecs run-task --region "${region}" --cluster "${cluster}" \
    --task-definition "${task_definition}" --launch-type FARGATE --count 1 \
    --network-configuration "${network_configuration}" \
    --query 'failures[].reason' --output text >&2
  exit 1
fi

echo "task ${task##*/} running; waiting for it to stop"
aws ecs wait tasks-stopped --region "${region}" --cluster "${cluster}" --tasks "${task}"

read -r status exit_code stopped_reason < <(aws ecs describe-tasks --region "${region}" \
  --cluster "${cluster}" --tasks "${task}" \
  --query 'tasks[0].[lastStatus,containers[0].exitCode,stoppedReason]' --output text)

mkdir -p "${here}/results"
out="${here}/results/${stage}-$(date -u +%Y%m%dT%H%M%SZ).log"
aws logs get-log-events --region "${region}" --log-group-name "${log_group}" \
  --log-stream-name "${stage}/bench/${task##*/}" \
  --no-cli-pager --query 'events[].message' --output text | tr '\t' '\n' > "${out}"

echo "log: ${out}"
if [[ "${exit_code}" != "0" ]]; then
  echo "task failed: ${stopped_reason} (exit ${exit_code}); last lines:" >&2
  tail -20 "${out}" >&2
  exit 1
fi

tail -30 "${out}"
