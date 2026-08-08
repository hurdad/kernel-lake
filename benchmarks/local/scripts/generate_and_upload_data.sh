#!/usr/bin/env bash
# Generates TPC-H-derived Parquet data (via the project's own
# tools/generate_tpch.py) and uploads it to this stack's MinIO bucket, over
# MinIO's S3 API published on localhost:9000 (see docker-compose.yml).
# Run once after `docker compose up`, or again after `docker compose down
# -v` (which drops the minio-data volume).
set -euo pipefail

SCALE_FACTOR="${1:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUCKET="kernellake-bench"
PREFIX="tpch-sf${SCALE_FACTOR}"
DATA_DIR="$(mktemp -d)"
trap 'rm -rf "$DATA_DIR"' EXIT

export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
ENDPOINT_URL="http://localhost:9000"

# --files: generate_tpch.py's own lineitem batching knob -- defaults to 1
# (every row of the whole scale factor generated as Python objects in a
# single in-memory batch before Arrow conversion). Left at the default, a
# local SF10 run (60M lineitem rows) drove this dev box (15GB RAM, 4GB
# swap) into a swap-thrash bad enough to require a hard reboot. Sized
# against ORDERS_BATCH_ROWS (5,000,000), the same per-batch row count
# generate_tpch.py's own orders generation already uses safely, and the
# same fix already applied to ../../aws/scripts/generate_and_upload_data.sh
# after it OOM-killed a 64GB EC2 instance at SF100.
LINEITEM_ROWS_PER_SF=6000000
BATCH_ROWS=5000000
LINEITEM_FILES=$(python3 -c "import math; print(max(1, math.ceil($LINEITEM_ROWS_PER_SF * $SCALE_FACTOR / $BATCH_ROWS)))")

echo "=== Generating TPC-H-derived data at SF${SCALE_FACTOR} (${LINEITEM_FILES} lineitem files) -> ${DATA_DIR} ==="
python3 "$REPO_ROOT/tools/generate_tpch.py" \
  --scale-factor "$SCALE_FACTOR" \
  --output "$DATA_DIR" \
  --format parquet \
  --compression snappy \
  --row-group-rows 1000000 \
  --files "$LINEITEM_FILES"

echo "=== Uploading to s3://${BUCKET}/${PREFIX}/ (MinIO at ${ENDPOINT_URL}) ==="
aws --endpoint-url "$ENDPOINT_URL" s3 cp "$DATA_DIR" "s3://${BUCKET}/${PREFIX}/" --recursive

echo "=== Done. Table globs (for scripts/run_e2e.sh / manual queries): ==="
for table in lineitem part orders customer; do
  echo "  s3://${BUCKET}/${PREFIX}/${table}-*.parquet"
done
