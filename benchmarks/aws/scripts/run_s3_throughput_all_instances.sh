#!/usr/bin/env bash
# Runs measure_s3_throughput.sh against every provisioned benchmark
# instance (KernelLake host(s), Spark host, DuckDB host) -- not just the
# KernelLake host alone, which is all the manual RUNBOOK.md step used to
# cover. Each instance measures its own real S3 GET throughput at
# increasing concurrency, over its own actual network path, labeled by
# role so the per-instance JSON outputs (and
# reporting/plot_s3_throughput.py's combined chart) can be told apart.
# Monitoring/Iceberg-catalog hosts are deliberately excluded -- they're
# infra support, not benchmark participants, so their own S3 throughput
# isn't a comparison point.
#
# Usage:
#   ./run_s3_throughput_all_instances.sh --scale-factor 100 --output-dir "$RUN_DIR"
#     [--bucket <bucket>] [--compression snappy] [--table lineitem]
#     [--concurrency-levels "1,4,8,16,32,64"] [--objects-per-level 20]
#
# --bucket defaults to `terraform output -raw s3_bucket_name` if omitted.
# Requires SSH access to every instance (same ssh_key_name/
# allowed_ssh_cidr as the rest of this harness) and the "ubuntu" user, same
# as every other scp/ssh step in docs/RUNBOOK.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAFORM_DIR="${SCRIPT_DIR}/../terraform"

BUCKET=""
SCALE_FACTOR=""
COMPRESSION="snappy"
COMPRESSION_LEVEL=""
TABLE="lineitem"
CONCURRENCY_LEVELS="1,4,8,16,32,64"
OBJECTS_PER_LEVEL=20
OUTPUT_DIR=""

while [ $# -gt 0 ]; do
  case "$1" in
    --bucket) BUCKET="$2"; shift 2 ;;
    --scale-factor) SCALE_FACTOR="$2"; shift 2 ;;
    --compression) COMPRESSION="$2"; shift 2 ;;
    --compression-level) COMPRESSION_LEVEL="$2"; shift 2 ;;
    --table) TABLE="$2"; shift 2 ;;
    --concurrency-levels) CONCURRENCY_LEVELS="$2"; shift 2 ;;
    --objects-per-level) OBJECTS_PER_LEVEL="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$SCALE_FACTOR" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: $0 --scale-factor <n> --output-dir <dir> [--bucket <bucket>] [--compression snappy] [--compression-level <N>] [--table lineitem] [--concurrency-levels \"1,4,8,16,32\"] [--objects-per-level 20]" >&2
  exit 1
fi

if [ -z "$BUCKET" ]; then
  BUCKET="$(cd "$TERRAFORM_DIR" && terraform output -raw s3_bucket_name)"
fi
mkdir -p "$OUTPUT_DIR"

COMPRESSION_LEVEL_FLAG=""
if [ -n "$COMPRESSION_LEVEL" ]; then
  COMPRESSION_LEVEL_FLAG="--compression-level '$COMPRESSION_LEVEL'"
fi

# `terraform output -raw` on a variable whose value is null (the
# enable_spark=false/enable_duckdb=false case, see spark_cluster.tf/
# duckdb_instance.tf) prints the literal string "null" rather than
# erroring or printing nothing -- checked for explicitly below, not just
# emptiness, so a disabled leg is skipped with a clear message instead of
# scp/ssh-ing to a host that doesn't exist.
tf_output_raw() {
  local value
  value="$(cd "$TERRAFORM_DIR" && terraform output -raw "$1" 2>/dev/null || true)"
  if [ "$value" = "null" ]; then
    printf ''
  else
    printf '%s' "$value"
  fi
}

# Runs measure_s3_throughput.sh against one host, labeled, writing its
# result into $OUTPUT_DIR -- the one piece of work repeated for every
# instance below.
run_one() {
  local host="$1" label="$2"
  if [ -z "$host" ]; then
    echo "=== Skipping ${label}: no instance (disabled or not provisioned) ===" >&2
    return
  fi
  echo "=== Measuring S3 throughput on ${label} (${host}) ===" >&2
  scp -o StrictHostKeyChecking=accept-new "${SCRIPT_DIR}/measure_s3_throughput.sh" "ubuntu@${host}:~/"
  ssh -o StrictHostKeyChecking=accept-new "ubuntu@${host}" \
    "./measure_s3_throughput.sh --bucket '$BUCKET' --scale-factor '$SCALE_FACTOR' \
     --compression '$COMPRESSION' ${COMPRESSION_LEVEL_FLAG} --table '$TABLE' \
     --concurrency-levels '$CONCURRENCY_LEVELS' --objects-per-level '$OBJECTS_PER_LEVEL' \
     --label '$label' --output s3-throughput.json"
  scp -o StrictHostKeyChecking=accept-new "ubuntu@${host}:~/s3-throughput.json" "${OUTPUT_DIR}/s3-throughput-${label}.json"
  echo "=== ${label}: wrote ${OUTPUT_DIR}/s3-throughput-${label}.json ===" >&2
}

# KernelLake: one label per replica (kernellake-0, kernellake-1, ...) --
# the M4 concurrency test runs more than one (see kernellake_instance_count
# in terraform/variables.tf), and each replica is its own real instance
# with its own real network path, not interchangeable.
#
# `terraform output -json` on an output the current state doesn't know
# about at all (never applied) exits non-zero with nothing on stdout --
# falls back to a literal "[]" so the python step below always gets valid
# JSON to parse, never an empty string it would choke on. That python step
# itself loops and prints one IP per line rather than
# "\n".join(...)-then-print: join() on an empty list still prints one
# blank line, which mapfile would read as a single empty-string element --
# a real bug caught by testing this against the current (unprovisioned)
# state directly, not assumed to be fine.
KL_JSON="$(cd "$TERRAFORM_DIR" && terraform output -json kernellake_instance_public_ips 2>/dev/null || echo '[]')"
mapfile -t KL_HOSTS < <(printf '%s' "$KL_JSON" | python3 -c '
import json, sys
for ip in (json.load(sys.stdin) or []):
    print(ip)
')
if [ "${#KL_HOSTS[@]}" -eq 0 ]; then
  echo "=== Skipping kernellake: no instances (kernellake_instance_count=0 or not provisioned) ===" >&2
else
  for i in "${!KL_HOSTS[@]}"; do
    run_one "${KL_HOSTS[$i]}" "kernellake-${i}"
  done
fi

run_one "$(tf_output_raw spark_host_public_ip)" "spark"
run_one "$(tf_output_raw duckdb_host_public_ip)" "duckdb"

echo "=== Done. Combine into one chart: python3 ../reporting/plot_s3_throughput.py --input-dir '$OUTPUT_DIR' --output '$OUTPUT_DIR/s3-throughput.png' ===" >&2
