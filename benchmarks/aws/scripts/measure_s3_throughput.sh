#!/bin/bash
# Raw S3 GET throughput from an EC2 host, independent of any query engine.
# Answers a narrower question than the full benchmark: at increasing
# request concurrency, how many MB/s can this *instance's own network
# path* pull from S3, and where does adding more parallel requests stop
# helping? That ceiling is the real number to compare a query engine's own
# observed scan throughput (scan_throughput_supporting.csv from
# reporting/generate_report.py) against, to tell whether S3 bandwidth
# itself is the bottleneck or whether the engine is leaving throughput on
# the table below that ceiling.
#
# Intended to run directly on the KernelLake EC2 host over SSH (same
# instance type/network path the real query benchmark uses), not from the
# orchestrator's own machine -- see docs/RUNBOOK.md. Uses the `aws` CLI
# directly (kernellake-host-init.sh installs it) rather than a `docker run`
# container, since the default IMDSv2 hop limit blocks a container's own
# network namespace from reaching instance-profile credentials unless the
# hop limit is explicitly raised.
#
# Usage:
#   ./measure_s3_throughput.sh --bucket <bucket> --scale-factor 100 \
#     [--compression snappy] [--table lineitem] \
#     [--concurrency-levels "1,4,8,16,32,64"] [--objects-per-level 20] \
#     --output s3_throughput.json
set -euo pipefail

BUCKET=""
SCALE_FACTOR=""
COMPRESSION="snappy"
TABLE="lineitem"
CONCURRENCY_LEVELS="1,4,8,16,32,64"
OBJECTS_PER_LEVEL=20
OUTPUT=""
REGION="us-east-1"

while [ $# -gt 0 ]; do
  case "$1" in
    --bucket) BUCKET="$2"; shift 2 ;;
    --scale-factor) SCALE_FACTOR="$2"; shift 2 ;;
    --compression) COMPRESSION="$2"; shift 2 ;;
    --table) TABLE="$2"; shift 2 ;;
    --concurrency-levels) CONCURRENCY_LEVELS="$2"; shift 2 ;;
    --objects-per-level) OBJECTS_PER_LEVEL="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$BUCKET" ] || [ -z "$SCALE_FACTOR" ] || [ -z "$OUTPUT" ]; then
  echo "Usage: $0 --bucket <bucket> --scale-factor <n> --output <path.json> [--compression snappy] [--table lineitem] [--concurrency-levels \"1,4,8,16,32\"] [--objects-per-level 20]" >&2
  exit 1
fi

PREFIX="tpch-data/sf${SCALE_FACTOR}-${COMPRESSION}/"
echo "=== Listing real objects under s3://${BUCKET}/${PREFIX} (table=${TABLE}) ===" >&2

# Real object keys + sizes via list-objects-v2 (ground truth, same as
# aws_benchmark_runner.py's s3_bytes_for_glob()), not an estimate -- filter
# to this table's own file prefix (e.g. "lineitem-") so a run against one
# table doesn't accidentally also pull part/orders/customer objects.
mapfile -t KEY_SIZE_PAIRS < <(
  aws s3api list-objects-v2 --bucket "$BUCKET" --prefix "${PREFIX}${TABLE}-" --region "$REGION" \
    --query "Contents[].[Key,Size]" --output text
)

if [ "${#KEY_SIZE_PAIRS[@]}" -eq 0 ]; then
  echo "No objects found under s3://${BUCKET}/${PREFIX}${TABLE}- -- has generate_and_upload_data.sh run for this scale factor/compression?" >&2
  exit 1
fi

echo "=== Found ${#KEY_SIZE_PAIRS[@]} object(s) ===" >&2

IFS=',' read -r -a LEVELS <<< "$CONCURRENCY_LEVELS"

RESULTS_JSON="[]"

for LEVEL in "${LEVELS[@]}"; do
  echo "=== Concurrency level: ${LEVEL} (${OBJECTS_PER_LEVEL} objects) ===" >&2

  TOTAL_BYTES=0
  PIDS=()
  TMP_DIR=$(mktemp -d)

  START=$(date +%s.%N)
  for ((i = 0; i < OBJECTS_PER_LEVEL; i++)); do
    PAIR="${KEY_SIZE_PAIRS[$((i % ${#KEY_SIZE_PAIRS[@]}))]}"
    KEY="${PAIR%%$'\t'*}"
    SIZE="${PAIR##*$'\t'}"
    TOTAL_BYTES=$((TOTAL_BYTES + SIZE))

    aws s3 cp "s3://${BUCKET}/${KEY}" - --region "$REGION" > /dev/null 2>"${TMP_DIR}/err.${i}" &
    PIDS+=($!)

    # Cap in-flight requests at $LEVEL -- wait for the oldest one before
    # launching another once the pool is full, rather than firing all
    # OBJECTS_PER_LEVEL at once regardless of the requested concurrency.
    if [ "${#PIDS[@]}" -ge "$LEVEL" ]; then
      wait "${PIDS[0]}"
      PIDS=("${PIDS[@]:1}")
    fi
  done
  wait
  END=$(date +%s.%N)
  rm -rf "$TMP_DIR"

  ELAPSED=$(echo "$END - $START" | bc)
  THROUGHPUT_MBPS=$(echo "scale=2; ($TOTAL_BYTES / 1000000) / $ELAPSED" | bc)
  THROUGHPUT_GBPS=$(echo "scale=3; ($TOTAL_BYTES * 8 / 1000000000) / $ELAPSED" | bc)

  echo "    ${TOTAL_BYTES} bytes in ${ELAPSED}s = ${THROUGHPUT_MBPS} MB/s (${THROUGHPUT_GBPS} Gbps)" >&2

  RESULTS_JSON=$(echo "$RESULTS_JSON" | python3 -c "
import json, sys
results = json.load(sys.stdin)
results.append({
    'concurrency': $LEVEL,
    'objects': $OBJECTS_PER_LEVEL,
    'total_bytes': $TOTAL_BYTES,
    'elapsed_seconds': $ELAPSED,
    'throughput_mbps': $THROUGHPUT_MBPS,
    'throughput_gbps': $THROUGHPUT_GBPS,
})
json.dump(results, sys.stdout)
")
done

python3 -c "
import json, sys

results = json.loads('''$RESULTS_JSON''')

# Simple plateau heuristic: the first concurrency level whose throughput
# is within 10% of the best throughput seen at any *lower* level -- i.e.
# where adding more parallel requests stopped meaningfully helping. Not a
# claim about *why* (could be S3 per-connection limits, this instance's
# own network cap, or client-side CPU/TLS overhead) -- just where the
# curve goes flat, for a human to interpret against the instance type's
# documented network bandwidth.
plateau_at = None
best_so_far = 0.0
for r in results:
    if best_so_far > 0 and r['throughput_mbps'] < best_so_far * 1.10:
        plateau_at = r['concurrency']
        break
    best_so_far = max(best_so_far, r['throughput_mbps'])

output = {
    'bucket': '$BUCKET',
    'scale_factor': $SCALE_FACTOR,
    'table': '$TABLE',
    'compression': '$COMPRESSION',
    'levels': results,
    'max_throughput_mbps': max(r['throughput_mbps'] for r in results),
    'max_throughput_gbps': max(r['throughput_gbps'] for r in results),
    'plateau_concurrency': plateau_at,
    'note': ('Throughput stopped improving meaningfully past concurrency=' + str(plateau_at) + ' -- compare max_throughput_mbps against this instance type\'s documented network bandwidth, and against the query engine\'s own observed scan_throughput_gbps in the same run\'s report, to tell whether S3 is the bottleneck.') if plateau_at else 'Throughput kept improving through the highest concurrency level tested -- try higher levels to find the real ceiling.',
}
with open('$OUTPUT', 'w') as f:
    json.dump(output, f, indent=2)
print(f'Wrote $OUTPUT', file=sys.stderr)
"

echo "=== Done ===" >&2
