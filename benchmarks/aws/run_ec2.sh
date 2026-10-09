#!/usr/bin/env bash
# Provision a benchmark EC2 instance with Terraform and run one stage on it over SSM.
#
# Used for grid-scale runs, which need more memory than a Fargate task can have:
# Fargate is capped at 8 vCPU by its own quota and cannot hold a CONUS-scale
# grid. Nothing here creates an instance directly; Terraform owns it, and the
# stage runs through SSM Run Command, so there is no key pair and no SSH.
#
# Usage:
#   ./benchmarks/aws/run_ec2.sh [stage] [git-ref] [repeats] [instance-type]
#
#   stage          grid_synthetic (default) | routine | spread | grid
#   git-ref        branch or commit to benchmark (default: main)
#   repeats        repetitions per backend (default: 2)
#   instance-type  default r7i.4xlarge (16 vCPU / 128 GiB, the vCPU quota ceiling)
#
# Environment:
#   GRID_ROWS, GRID_COLS  synthetic grid size for grid_synthetic (default 596 x 1385, CONUS-like)
#   CPUS                  taskset CPU set (default 0-15)
#   KEEP=1                leave the instance running instead of destroying it
set -euo pipefail

stage="${1:-grid_synthetic}"
git_ref="${2:-main}"
repeats="${3:-2}"
instance_type="${4:-r7i.4xlarge}"
region="${AWS_REGION:-us-east-2}"
grid_rows="${GRID_ROWS:-596}"
grid_cols="${GRID_COLS:-1385}"
grid_entries="${GRID_ENTRIES:-}"
grid_threads="${GRID_THREADS:-}"
cpus="${CPUS:-0-15}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tf="${here}/terraform"
raw_base="https://raw.githubusercontent.com/monocongo/climate_indices/${git_ref}/benchmarks/aws/bootstrap.sh"

cd "${tf}"
terraform init -input=false >/dev/null
terraform apply -input=false -auto-approve \
  -var "ec2_instance_type=${instance_type}" \
  -var "stage=${stage}" -var "git_ref=${git_ref}" -var "repeats=${repeats}" >/dev/null

instance_id="$(terraform output -raw instance_id)"
echo "${stage} on ${instance_type} (ref ${git_ref}, repeats ${repeats}); instance ${instance_id}"

cleanup() {
  if [[ "${KEEP:-0}" == "1" ]]; then
    echo "KEEP=1: instance ${instance_id} left running at the hourly rate; stop or destroy it when done"
  else
    echo "destroying ${instance_id}"
    cd "${tf}" && terraform destroy -input=false -auto-approve \
      -var "ec2_instance_type=${instance_type}" >/dev/null
  fi
}
trap cleanup EXIT

echo "waiting for the SSM agent to register"
for _ in $(seq 1 60); do
  status="$(aws ssm describe-instance-information --region "${region}" \
    --filters "Key=InstanceIds,Values=${instance_id}" \
    --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || echo None)"
  [[ "${status}" == "Online" ]] && break
  sleep 10
done
if [[ "${status}" != "Online" ]]; then
  echo "instance never registered with SSM (status ${status})" >&2
  exit 1
fi

# The bootstrap script is fetched from the ref under test rather than from this
# working tree, so the ref being benchmarked supplies its own runner. That needs
# the repository to stay public; a private fork would need the script staged in S3.
#
# executionTimeout is raised well above the one-hour default: the toolchain
# build plus a CONUS-scale grid run does not finish inside an hour.
commands="$(python3 -c '
import json, sys
ref, stage, repeats, cpus, rows, cols, entries, threads, url = sys.argv[1:10]
print(json.dumps([
    "set -euxo pipefail",
    "export HOME=/root",
    f"curl -sSfL {url} -o /tmp/bootstrap.sh",
    f"STAGE={stage} GIT_REF={ref} REPEATS={repeats} CPUS={cpus} GRID_ROWS={rows} GRID_COLS={cols}"
    f" GRID_ENTRIES={entries} GRID_THREADS={threads} bash /tmp/bootstrap.sh",
]))
' "${git_ref}" "${stage}" "${repeats}" "${cpus}" "${grid_rows}" "${grid_cols}" "${grid_entries}" "${grid_threads}" "${raw_base}")"

command_id="$(aws ssm send-command --region "${region}" \
  --instance-ids "${instance_id}" \
  --document-name AWS-RunShellScript \
  --comment "climate_indices ${stage} benchmark on ${git_ref}" \
  --timeout-seconds 10800 \
  --parameters "${commands}" \
  --query 'Command.CommandId' --output text)"
echo "command ${command_id}"

for _ in $(seq 1 1080); do
  status="$(aws ssm get-command-invocation --region "${region}" \
    --command-id "${command_id}" --instance-id "${instance_id}" \
    --query 'Status' --output text 2>/dev/null || echo Pending)"
  case "${status}" in
    Success | Failed | Cancelled | TimedOut) break ;;
  esac
  sleep 10
done

mkdir -p "${here}/results"
out="${here}/results/${stage}-ec2-$(date -u +%Y%m%dT%H%M%SZ).log"
aws ssm get-command-invocation --region "${region}" \
  --command-id "${command_id}" --instance-id "${instance_id}" \
  --query '[StandardOutputContent, StandardErrorContent]' --output text | tr '\t' '\n' > "${out}"

echo "status ${status}; log ${out}"
tail -40 "${out}"
[[ "${status}" == "Success" ]] || { echo "stage failed: ${status}" >&2; exit 1; }
