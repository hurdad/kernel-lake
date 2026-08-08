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

echo "=== Generating TPC-H-derived data at SF${SCALE_FACTOR} -> ${DATA_DIR} ==="
python3 "$REPO_ROOT/tools/generate_tpch.py" \
  --scale-factor "$SCALE_FACTOR" \
  --output "$DATA_DIR" \
  --format parquet \
  --compression snappy \
  --row-group-rows 1000000

echo "=== Uploading to s3://${BUCKET}/${PREFIX}/ (MinIO at ${ENDPOINT_URL}) ==="
aws --endpoint-url "$ENDPOINT_URL" s3 cp "$DATA_DIR" "s3://${BUCKET}/${PREFIX}/" --recursive

echo "=== Done. Table globs (for scripts/run_e2e.sh / manual queries): ==="
for table in lineitem part orders customer; do
  echo "  s3://${BUCKET}/${PREFIX}/${table}-*.parquet"
done
