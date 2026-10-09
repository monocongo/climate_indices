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
#   GRID_ENTRIES          narrow the entries the grid run times (default: all six)
#   GRID_THREADS          narrow the thread counts (default: the harness default)
#   CPUS                  taskset CPU set (default 0-15)
#   SHUTDOWN_MINUTES      instance watchdog, in minutes (default 240); the instance powers off at
#                         the deadline and terminates itself, so an abandoned run cannot bill forever
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
percell_cells="${PERCELL_CELLS:-1000}"
percell_repeats="${PERCELL_REPEATS:-2}"
cpus="${CPUS:-0-15}"
shutdown_minutes="${SHUTDOWN_MINUTES:-240}"

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

# The stage runs detached, under nohup, writing a completion marker.
#
# Run Command kills a command when the AWS-RunShellScript document's own
# executionTimeout expires, which defaults to one hour and overrides
# --timeout-seconds. A CONUS-scale grid run does not finish in an hour, and a
# killed command loses the whole run, so the launcher returns immediately and the
# harness outlives it. The launcher also starts a shutdown watchdog, because an
# abandoned instance otherwise bills until someone notices.
#
# The bootstrap script is fetched from the ref under test rather than from this
# working tree, so the ref being benchmarked supplies its own runner. That needs
# the repository to stay public; a private fork would need the script staged in S3.
commands="$(python3 -c '
import json, sys
ref, stage, repeats, cpus, rows, cols, entries, threads, pcells, prepeats, url, shutdown = sys.argv[1:13]
env = (
    f"STAGE={stage} GIT_REF={ref} REPEATS={repeats} CPUS={cpus} GRID_ROWS={rows}"
    f" GRID_COLS={cols} GRID_ENTRIES={entries} GRID_THREADS={threads}"
    f" PERCELL_CELLS={pcells} PERCELL_REPEATS={prepeats}"
)
inner = (
    "export HOME=/root; curl -sSfL " + url + " -o /tmp/bootstrap.sh && "
    + env + " bash /tmp/bootstrap.sh > /tmp/stage.log 2>&1; echo $? > /tmp/EXIT"
)
print(json.dumps({
    "commands": [
        "set -x",
        "rm -f /tmp/EXIT /tmp/stage.log",
        f"nohup bash -c {json.dumps(inner)} > /dev/null 2>&1 &",
        f"shutdown -h +{shutdown} 2>/dev/null || true",
        "sleep 5; echo launched",
    ],
    "executionTimeout": ["120"],
}))
' "${git_ref}" "${stage}" "${repeats}" "${cpus}" "${grid_rows}" "${grid_cols}" \
  "${grid_entries}" "${grid_threads}" "${percell_cells}" "${percell_repeats}" "${raw_base}" "${shutdown_minutes}")"

command_id="$(aws ssm send-command --region "${region}" \
  --instance-ids "${instance_id}" \
  --document-name AWS-RunShellScript \
  --comment "climate_indices ${stage} benchmark on ${git_ref}" \
  --timeout-seconds 300 \
  --parameters "${commands}" \
  --query 'Command.CommandId' --output text)"
echo "launcher ${command_id}; stage is detached, so it survives Run Command timeouts"
echo "watchdog: instance powers off in ${shutdown_minutes} minutes if this script is abandoned"

# Poll for the marker, then read the report the stage produced.
poll_seconds=$(( (shutdown_minutes + 15) * 60 ))
deadline=$(( SECONDS + poll_seconds ))
exit_code=""
while (( SECONDS < deadline )); do
  probe="$(aws ssm send-command --region "${region}" \
    --instance-ids "${instance_id}" --document-name AWS-RunShellScript --timeout-seconds 60 \
    --parameters '{"commands":["cat /tmp/EXIT 2>/dev/null || echo RUNNING"],"executionTimeout":["45"]}' \
    --query 'Command.CommandId' --output text)"
  sleep 15
  answer="$(aws ssm get-command-invocation --region "${region}" \
    --command-id "${probe}" --instance-id "${instance_id}" \
    --query 'StandardOutputContent' --output text 2>/dev/null | tr -d '\t\n' || echo RUNNING)"
  if [[ "${answer}" =~ ^[0-9]+$ ]]; then
    exit_code="${answer}"
    break
  fi
  printf '.'
  sleep 30
done

if [[ -z "${exit_code}" ]]; then
  echo "stage did not finish within the watchdog window" >&2
  exit 1
fi

mkdir -p "${here}/results"
out="${here}/results/${stage}-ec2-$(date -u +%Y%m%dT%H%M%SZ).log"
fetch="$(aws ssm send-command --region "${region}" \
  --instance-ids "${instance_id}" --document-name AWS-RunShellScript --timeout-seconds 120 \
  --parameters '{"commands":["cat /tmp/stage.log"],"executionTimeout":["90"]}' \
  --query 'Command.CommandId' --output text)"
sleep 20
aws ssm get-command-invocation --region "${region}" --command-id "${fetch}" --instance-id "${instance_id}" \
  --query 'StandardOutputContent' --output text 2>/dev/null | tr '\t' '\n' > "${out}"

echo "stage exit ${exit_code}; log ${out}"
tail -40 "${out}"
[[ "${exit_code}" == "0" ]] || { echo "stage failed with exit ${exit_code}" >&2; exit 1; }
