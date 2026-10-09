# Running the Rust-vs-Python benchmarks on AWS

Provisioned and run as a Fargate task, not an EC2 instance. This directory holds
everything needed to reproduce a benchmark run on dedicated hardware, and the
reasoning behind each choice, so the run does not have to be rediscovered.

## Why Fargate and not EC2

On this AWS account EC2 cannot run the shapes these benchmarks need:

- The account is on the AWS **FREE plan**, which restricts `RunInstances` to
  free-tier-eligible instance types. In `us-east-2` that list is eight shapes,
  every one **2 vCPU and at most 8 GiB** (`m7i-flex.large`, `c7i-flex.large`,
  `t3`/`t4g`/`t8i` micro and small). Larger types are refused with
  `InvalidParameterCombination: not eligible for Free Tier`. A dry run
  (`--dry-run`) *passes* for those types, so only a real attempt is evidence.
- An AWS Organizations service control policy denies the EC2 API outside
  `us-east-2` (`ec2:DescribeInstanceTypes` is an explicit deny elsewhere), and
  denies the billing and account surfaces, so the plan cannot be upgraded
  through this role either.
- The biggest free-tier-eligible shape cannot run the existing CONUS
  nClimGrid case at all: that run peaks near 19 GB, and the prepared
  precipitation array alone is 825,460 cells x 528 months x 8 bytes ~ 3.5 GB.
  No amount of running time changes that.

Fargate does not select an instance type, so none of those restrictions apply.
It is also a better fit for the result we want: a dedicated, single-tenant
machine with a fixed CPU count, no co-tenant load, and no thermal drift.

## Quotas and the shape ceiling

| quota | value | effect |
| --- | --- | --- |
| Fargate on-demand vCPU resource count (`L-3032A538`) | 12 | largest valid task CPU is **8 vCPU** (Fargate accepts 256, 512, 1024, 2048, 4096, 8192, 16384 only) |
| Fargate Spot vCPU resource count (`L-36FBB829`) | 12 | Spot is available for restartable stages |

At 8 vCPU a task may take 16-60 GiB, which covers the CONUS case and the
routine run. Two tasks can run concurrently (8 + 4 = 12 vCPU). A quota increase
to 64 vCPU was requested with:

```bash
aws service-quotas request-service-quota-increase --service-code fargate \
  --quota-code L-3032A538 --desired-value 64 --region us-east-2
```

Once granted, `task_cpu = 16384` (16 vCPU / 32-120 GiB) becomes available for
the largest grids.

## Cost

Fargate on-demand in `us-east-2`: **$0.04048 per vCPU-hour** and
**$0.004445 per GiB-hour**, plus $0.000111 per GiB-hour of ephemeral storage
beyond 20 GiB.

| shape | $/hour | typical stage |
| --- | --- | --- |
| 8 vCPU / 16 GiB | ~0.40 | routine run, spread run |
| 8 vCPU / 32 GiB | ~0.47 | CONUS monthly grid |
| 8 vCPU / 60 GiB | ~0.59 | largest grid that fits at 8 vCPU |

A stage includes a ~10 minute bootstrap (Rust toolchain, `uv`, release build),
so budget ~25-40 minutes and **well under $1 per run**. Spend draws on the
account's credits.

## Gotchas found the hard way

1. **The minimal AL2023 container has `curl` but no `tar` or `unzip`.** The
   official `uv` and `rustup` installers cannot unpack their archives, and the
   failure surfaces only as a non-zero exit with the error lost at the end of
   the log. Install `tar unzip gzip` before either installer runs.
2. **No `lscpu` or `free` either.** Under `set -e` a probe for them kills the
   task with exit 127. Read `/proc/cpuinfo`, or install `util-linux` / `procps-ng`.
3. **Task exit 1 or 127 with a short log means a missing command**, not a
   benchmark failure. Check the log tail before suspecting the harness.
4. **Read the log after the task stops.** Querying CloudWatch Logs while the
   task is finishing returns a truncated stream.
5. **Log stream name is `<prefix>/<container-name>/<task-id>`**, where the
   prefix is what the task definition sets, not the log group.
6. **`awslogs-create-group` needs `logs:CreateLogGroup`**, which the managed
   `AmazonECSTaskExecutionRolePolicy` does not grant. Terraform creates the log
   group instead and the task definition does not ask for it.
7. **`ephemeralStorage` needs platform version 1.4.0+** (the default) and must
   be raised for a Rust build; the 20 GiB default is tight once the repo,
   toolchain, virtualenv, and `cargo` target directory are all present.
8. **Egress works with `assignPublicIp = ENABLED`** on a default-VPC subnet:
   `github.com`, `astral.sh`, and `sh.rustup.rs` all answer. Only the default
   subnet's route to an internet gateway is needed, and no inbound rule is
   required at all, so the task security group is egress-only.
9. **Fargate is charged for the whole task lifetime**, including image pull and
   bootstrap, so a failed stage still costs its minutes.
10. **`/proc/meminfo` shows the underlying microVM, not the task's memory
    limit.** A 16 GiB task reports ~32 GiB there, because `/proc` is not
    namespaced; read the cgroup limit for the budget the task actually has.
11. **The spread stage needs its script present in the benchmarked ref.** It
    runs `benchmarks/aws/spread.py` from the clone, so before this lands on
    `main` the ref must be passed explicitly, or the clone fails on an unknown
    branch and the task exits 128.

## Layout

| path | purpose |
| --- | --- |
| `terraform/` | all AWS resources: cluster, log group, IAM roles, security group, task definition, optional schedule |
| `bootstrap.sh` | runs inside the task: install toolchains, build the extension, run one stage |
| `spread.py` | written by `bootstrap.sh`; records every sample rather than the best, so the run's noise floor is measurable |
| `run.sh` | `terraform apply`, start the task, wait, print the log |

## Running a stage

```bash
# routine harness at the task's CPU count, best-of-15
./benchmarks/aws/run.sh routine

# every sample recorded, with medians and interquartile ranges
./benchmarks/aws/run.sh spread 30

# a specific revision
./benchmarks/aws/run.sh routine main

# before this lands on main, name the ref: the spread stage runs its own script
# from the clone, which only exists on the branch
./benchmarks/aws/run.sh spread ci/1324-benchmark-aws-terraform 30
```

Stages are selected by the `STAGE` environment variable inside the task, so a
new stage is one `case` arm in `bootstrap.sh`.

## Operating notes

- The routine harness and the spread run need no input fixtures. The full-grid
  stages do, and those fixtures are deliberately not in the repository; stage
  them in S3 and extend `bootstrap.sh` when a grid stage is added.
- Nothing here creates EC2 instances, key pairs, or SSH access. The task needs
  no inbound connectivity.
- `terraform destroy` removes everything; nothing bills while idle.

## What this replaces

The first attempts at this work created resources with ad-hoc `aws` CLI calls
(security group, key pair, cluster, log group, IAM roles, a probe task
definition). Those exist only to be superseded by this Terraform configuration,
which becomes the single source of truth for the account's benchmark resources.

## Grid-scale runs on EC2

Fargate is capped at 8 vCPU by its own quota and cannot hold a CONUS-scale grid,
so the large-memory stages run on an EC2 instance instead:

```bash
# verify the path cheaply: two entries, two thread counts, one sample
GRID_ROWS=300 GRID_COLS=700 GRID_ENTRIES=spi_gamma,palmer_pdsi GRID_THREADS=1,4 \
  ./benchmarks/aws/run_ec2.sh grid_synthetic ci/1324-benchmark-aws-terraform 1

# the full grid measurement at CONUS-like scale
./benchmarks/aws/run_ec2.sh grid_synthetic <ref> 2
```

The instance is `r7i.4xlarge` by default: 16 vCPU / 128 GiB at $1.0584/hr, which
is the largest shape the account's 16-vCPU standard quota allows. Terraform owns
it, and the stage runs through SSM Run Command, so there is no key pair and the
instance's security group has **no ingress rule at all** — Run Command and
Session Manager are outbound connections from the instance. `run_ec2.sh`
destroys the instance when the stage finishes unless `KEEP=1` is set.

Two things to know about this path:

- **The task fetches `bootstrap.sh` from the ref being benchmarked**, over
  `raw.githubusercontent.com`, so the ref supplies its own runner. That requires
  the repository to stay public; a private fork would need the script staged in S3.
- **The synthetic grid exists to make grid scale reproducible.** The real
  fixtures (CHIRPS, nClimGrid) are not in the repository, so `make_synthetic_grid.py`
  writes a prepared grid of the same shape and contract: January-first consecutive
  months covering the calibration window, `time`/`lat`/`lon` dims, and both inputs'
  SHA-256 recorded, as the real full-grid artifact does. Values are synthetic, so a
  result describes the gridded code path at that size, not any particular dataset.
  Swap in real prepared files with the `grid` stage and `NETCDF`/`TAVG` when the
  actual CONUS numbers are wanted.

For real data, the precedent is the nClimGrid preparation recipe in
`benchmarks/README.md` (NOAA's public NODD bucket: download, then
`cli_multiprocessing.py prepare`). That step belongs here as another stage; it has
not been written yet.

### Two ways a grid run fails, and how the runner handles them

- **`AWS-RunShellScript` has its own `executionTimeout`, defaulting to one hour**, and
  it overrides the `--timeout-seconds` passed to `send-command`. A CONUS-scale grid run
  does not finish in an hour, and a killed command loses the entire run (the harness
  writes its report only at the end, so nothing is salvageable). `run_ec2.sh` therefore
  runs the stage **detached under `nohup`**, returns from the launcher immediately, and
  polls for a completion marker file; the launcher also sets an explicit
  `executionTimeout` well below the practical ceiling.
- **An abandoned instance bills until someone notices.** The launcher starts a shutdown
  watchdog (`SHUTDOWN_MINUTES`, default 240), and the instance's
  `instance_initiated_shutdown_behavior` is `terminate`, so the watchdog deletes the
  instance rather than leaving a stopped one paying for its volumes. Killing the local
  script does not stop the remote stage, which is deliberate: the run survives the
  operator, and the watchdog bounds the cost either way.

Consequence worth knowing: because the stage is detached, a lost local session leaves the
benchmark running. Reattach with SSM rather than relaunching:

```bash
aws ssm send-command --region us-east-2 --instance-ids <id> --document-name AWS-RunShellScript \
  --parameters '{"commands":["cat /tmp/EXIT 2>/dev/null || echo RUNNING"],"executionTimeout":["45"]}'
```
