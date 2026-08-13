#!/usr/bin/env bash
# Launches a short-lived, GPU-free EC2 instance (data generation is CPU-
# bound -- doesn't need to burn g6.8xlarge GPU-instance-hours on it),
# generates TPC-H-derived Parquet data at the given scale factor via this
# repo's tools/generate_tpch.py, uploads it to the benchmark S3 bucket, and
# terminates itself. Not part of the main Terraform state (it's a one-shot
# batch job, not standing infrastructure) -- reads the bucket/subnet/
# security-group/IAM-profile it needs from `terraform output`, so run
# scripts/provision.sh first.
#
# Usage: generate_and_upload_data.sh --scale-factor 100 [--wait]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TERRAFORM_DIR="${SCRIPT_DIR}/../terraform"

SCALE_FACTOR=""
WAIT=false
# generate_tpch.py's row generation is single-threaded pure Python (one
# rng call per field per row, no multiprocessing) -- confirmed for real via
# `ps aux` on a running c6i.8xlarge instance (99.9% on exactly one of 32
# vCPUs, ~125K rows/sec). A big instance buys nothing here except letting
# fewer of these fit under the account's 64-vCPU on-demand quota at once;
# c6i.2xlarge (8 vCPU, 16GB RAM -- comfortably above the ~4.6GB RSS
# actually observed) is plenty and lets 8 of these run concurrently
# instead of 2, which matters when generating multiple compression
# variants of the same scale factor side by side.
INSTANCE_TYPE="c6i.2xlarge"
USE_SPOT=false
# Every compression choice (including snappy) writes to its own, fully
# self-describing tpch-data/sf<N>-<tag>/ prefix -- no unsuffixed "default"
# path. An earlier version left snappy unsuffixed for backward
# compatibility with data generated before this dimension existed, but
# that was confusing in practice (a real "I don't see sf100-snappy"
# question) once multiple compressions started being compared side by
# side, so every variant is explicit now, snappy included.
COMPRESSION="snappy"
# zstd only -- see generate_tpch.py's own --compression-level comment for
# why this needs to be settable at all (PyArrow's codec default, level 1,
# differs from libzstd's own upstream default, level 3).
COMPRESSION_LEVEL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --scale-factor) SCALE_FACTOR="$2"; shift 2 ;;
    --wait) WAIT=true; shift ;;
    --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
    --compression) COMPRESSION="$2"; shift 2 ;;
    --compression-level) COMPRESSION_LEVEL="$2"; shift 2 ;;
    # On-demand by default: generate_tpch.py uploads only once, at the very
    # end (no incremental checkpointing) -- a spot interruption partway
    # through a multi-hour SF3000/SF10000 generation would lose all of it,
    # not just resume. --spot is fine for quick SF100 validation runs
    # where a restart costs minutes, not hours.
    --spot) USE_SPOT=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done
if [ -z "$SCALE_FACTOR" ]; then
  echo "Usage: $0 --scale-factor <N> [--wait] [--instance-type <type>] [--compression none|snappy|zstd] [--compression-level <N>]" >&2
  exit 1
fi
case "$COMPRESSION" in
  none|snappy|zstd) ;;
  *) echo "--compression must be 'none', 'snappy', or 'zstd', got: $COMPRESSION" >&2; exit 1 ;;
esac
if [ -n "$COMPRESSION_LEVEL" ] && [ "$COMPRESSION" != "zstd" ]; then
  echo "--compression-level only applies to --compression zstd" >&2
  exit 1
fi
# Matches aws_benchmark_runner.py's compression_tag(): zstd with no
# explicit --compression-level defaults to "zstd" (PyArrow's own codec
# default, confirmed == level 1) rather than "zstd-l1" -- an explicit
# --compression-level 1 run still gets the fully-spelled-out "zstd-l1" tag
# (see below), so both spellings exist on purpose: "zstd" for "whatever
# PyArrow does by default," "zstd-l<N>" for "specifically this level."
if [ "$COMPRESSION" = "zstd" ] && [ -z "$COMPRESSION_LEVEL" ]; then
  COMPRESSION_TAG="zstd"
elif [ "$COMPRESSION" = "zstd" ]; then
  COMPRESSION_TAG="zstd-l${COMPRESSION_LEVEL}"
else
  COMPRESSION_TAG="$COMPRESSION"
fi
SF_DIR="sf${SCALE_FACTOR}-${COMPRESSION_TAG}"
# Plain string, not an array -- interpolated directly into the USER_DATA
# heredoc below (a remote shell script, not this script's own argv), which
# can't expand a local bash array.
COMPRESSION_LEVEL_FLAG=""
if [ -n "$COMPRESSION_LEVEL" ]; then
  COMPRESSION_LEVEL_FLAG="--compression-level ${COMPRESSION_LEVEL}"
fi

cd "$TERRAFORM_DIR"
BUCKET="$(terraform output -raw s3_bucket_name)"
# REGION/SSH_KEY: honor a pre-set env var before falling back to terraform
# output -- both are plain `var.*` passthroughs with no resource
# dependency of their own (see outputs.tf), so a `terraform apply -target`
# that only re-creates networking/IAM (e.g. after a partial teardown, to
# run just this data-gen step without standing up the full GPU/Spark/
# DuckDB/monitoring stack) never pulls them into state -- confirmed for
# real: `-target=output.aws_region` plans "No changes" since the output
# has no targetable resource to hang off of, not a mistake in the target
# list.
REGION="${REGION:-$(terraform output -raw aws_region)}"
SUBNET_ID="$(terraform output -raw subnet_id)"
SG_ID="$(terraform output -raw security_group_id)"
PROFILE="$(terraform output -raw iam_instance_profile_name)"
SSH_KEY="${SSH_KEY:-$(terraform output -raw ssh_key_name)}"
cd - >/dev/null

echo "=== Uploading tools/generate_tpch.py to s3://${BUCKET}/scripts/ ==="
aws s3 cp "${REPO_ROOT}/tools/generate_tpch.py" "s3://${BUCKET}/scripts/generate_tpch.py"

# Rough EBS sizing: real TPC-H-shaped Parquet+snappy runs ~0.35GB per
# scale-factor-unit (see scripts/estimate_cost.py's own documented
# estimate) -- generate_tpch.py writes Parquet directly (no intermediate
# raw-text staging, unlike real dbgen+a separate loader), so local disk
# only needs to hold the final output briefly before it's uploaded and can
# be cleaned up. 2x the estimate plus a 20GB floor for the OS/tooling.
EBS_SIZE_GB=$(python3 -c "print(max(20, int($SCALE_FACTOR * 0.35 * 2)))")

# --files: generate_tpch.py's own lineitem batching knob -- defaults to 1
# (every row of the whole scale factor generated as Python objects in a
# single in-memory batch before Arrow conversion, per that script's own
# comment on generate_lineitem_batch()/generate_orders_batch()). Left at
# the default, a real SF100 run (600M lineitem rows) OOM-killed a
# c6i.8xlarge (64GB RAM) at 63.8GB RSS before it ever reached the S3
# upload step -- confirmed via `dmesg` after the instance sat at ~0% CPU
# for 45+ minutes looking like it was still working. Sized against
# ORDERS_BATCH_ROWS (5,000,000), the same per-batch row count
# generate_tpch.py's own orders generation already uses safely, rather
# than a new guessed number.
LINEITEM_ROWS_PER_SF=6000000
BATCH_ROWS=5000000
LINEITEM_FILES=$(python3 -c "import math; print(max(1, math.ceil($LINEITEM_ROWS_PER_SF * $SCALE_FACTOR / $BATCH_ROWS)))")

USER_DATA=$(cat <<EOF
#!/bin/bash
set -euo pipefail
exec > >(tee /var/log/generate-data.log) 2>&1

# IMDSv2 (token-based) -- this account's instances default to HttpTokens:
# required, confirmed for real after a plain unauthenticated IMDSv1-style
# GET silently returned an empty body (401, swallowed by \$(...)), which
# made "aws ec2 terminate-instances --instance-ids ''" fail with
# InvalidInstanceID.Malformed and left two real instances running/billing
# indefinitely instead of self-terminating.
IMDS_TOKEN=\$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
INSTANCE_ID=\$(curl -s -H "X-aws-ec2-metadata-token: \${IMDS_TOKEN}" http://169.254.169.254/latest/meta-data/instance-id)

# Registered before any of the steps below that can fail, and unconditional
# -- a bash EXIT trap fires on every exit path (falling off the end,
# an explicit exit, or set -e's own automatic exit on an unhandled
# non-zero return), not just a clean success. A prior version only ever
# called terminate-instances as the literal last line, reached solely by
# every step above it succeeding in sequence -- so any failure partway
# through (e.g. generate_tpch.py OOM-killed) left the instance sitting
# "running", doing nothing, billing indefinitely, contrary to what this
# script prints ("self-terminates on completion or failure"). Confirmed
# for real: a live SF1000 run hit exactly this.
trap 'aws ec2 terminate-instances --region ${REGION} --instance-ids "\${INSTANCE_ID}"' EXIT

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends python3 python3-pip awscli
pip3 install --break-system-packages pyarrow

mkdir -p /data
aws s3 cp "s3://${BUCKET}/scripts/generate_tpch.py" /tmp/generate_tpch.py

echo "=== Generating TPC-H-derived data at SF${SCALE_FACTOR} (${COMPRESSION}${COMPRESSION_LEVEL:+ level ${COMPRESSION_LEVEL}}, ${LINEITEM_FILES} lineitem files) ==="
python3 /tmp/generate_tpch.py --scale-factor ${SCALE_FACTOR} --output /data \\
  --format parquet --compression ${COMPRESSION} ${COMPRESSION_LEVEL_FLAG} --row-group-rows 1000000 --files ${LINEITEM_FILES}

echo "=== Uploading to s3://${BUCKET}/tpch-data/${SF_DIR}/ ==="
aws s3 sync /data "s3://${BUCKET}/tpch-data/${SF_DIR}/"

# Explicit completion marker -- the local script's --wait polling looks
# for this rather than inferring completion from instance state (an
# instance terminating early due to an error would otherwise look
# indistinguishable from a successful, self-terminating run).
echo '{"scale_factor": ${SCALE_FACTOR}, "compression": "${COMPRESSION}", "compression_level": ${COMPRESSION_LEVEL:-null}, "status": "complete"}' | \\
  aws s3 cp - "s3://${BUCKET}/tpch-data/${SF_DIR}/.generation-complete"
# Trap above handles termination on exit -- nothing further needed here.
EOF
)

echo "=== Launching ${INSTANCE_TYPE} data-generation instance (SF${SCALE_FACTOR}, ${EBS_SIZE_GB}GB EBS) ==="
# On-demand by default (see the --spot flag's own comment above for why:
# generate_tpch.py has no incremental checkpointing, so a spot interruption
# on a long run loses everything). Previously this always passed
# --instance-market-options spot regardless of $USE_SPOT -- fixed to
# actually branch on the flag.
MARKET_OPTIONS_ARGS=()
if [ "$USE_SPOT" = true ]; then
  MARKET_OPTIONS_ARGS=(--instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}')
fi
INSTANCE_ID=$(aws ec2 run-instances \
  --region "$REGION" \
  --instance-type "$INSTANCE_TYPE" \
  --image-id "$(aws ec2 describe-images --region "$REGION" --owners 099720109477 \
      --filters "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-*-26.04-amd64-server-*" "Name=virtualization-type,Values=hvm" \
      --query 'sort_by(Images, &CreationDate)[-1].ImageId' --output text)" \
  --subnet-id "$SUBNET_ID" \
  --security-group-ids "$SG_ID" \
  --iam-instance-profile "Name=${PROFILE}" \
  --key-name "$SSH_KEY" \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${EBS_SIZE_GB},\"VolumeType\":\"gp3\"}}]" \
  "${MARKET_OPTIONS_ARGS[@]}" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=kernellake-bench-datagen-${SF_DIR}},{Key=Ephemeral,Value=true}]" \
  --user-data "$USER_DATA" \
  --query 'Instances[0].InstanceId' --output text)

if [ "$USE_SPOT" = true ]; then
  echo "Launched: ${INSTANCE_ID} (spot -- self-terminates on completion or failure to keep this cheap)"
else
  echo "Launched: ${INSTANCE_ID} (on-demand -- self-terminates on completion or failure)"
fi
echo "Progress: tail via SSM or check s3://${BUCKET}/tpch-data/${SF_DIR}/.generation-complete"

if [ "$WAIT" = true ]; then
  echo "=== Waiting for completion marker (--wait given) ==="
  while ! aws s3 ls "s3://${BUCKET}/tpch-data/${SF_DIR}/.generation-complete" >/dev/null 2>&1; do
    if ! aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null | grep -qv "terminated\|shutting-down"; then
      echo "Instance is no longer running and no completion marker was found -- generation likely failed." >&2
      echo "Check /var/log/generate-data.log on the instance (if still reachable) or CloudWatch." >&2
      exit 1
    fi
    sleep 30
  done
  echo "=== Data generation complete: s3://${BUCKET}/tpch-data/${SF_DIR}/ ==="
fi
