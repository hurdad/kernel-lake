#!/usr/bin/env bash
# Thin wrapper around `terraform apply` -- always shows the plan; only
# actually applies (spends real money) with an explicit --yes. See
# ../docs/RUNBOOK.md for when to run this (after reviewing
# estimate_cost.py's output for the relevant milestone) and
# ../README.md's cost-warnings section.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAFORM_DIR="${SCRIPT_DIR}/../terraform"
MONITORING_DIR="${SCRIPT_DIR}/../monitoring"

APPLY=false
for arg in "$@"; do
  case "$arg" in
    --yes) APPLY=true ;;
    *) ;;
  esac
done

cd "$TERRAFORM_DIR"
terraform init -input=false
terraform validate

echo "=== terraform plan ==="
terraform plan -out=tfplan

if [ "$APPLY" != true ]; then
  echo
  echo "Plan-only run (no --yes given). Review the plan above, and"
  echo "../scripts/estimate_cost.py's dollar estimate, before re-running"
  echo "with --yes to actually create these resources and start spending."
  exit 0
fi

echo "=== terraform apply (--yes given -- creating real, billed AWS resources) ==="
terraform apply -input=false tfplan

BUCKET="$(terraform output -raw s3_bucket_name)"
GRAFANA_URL="$(terraform output -raw grafana_url)"

echo "=== Uploading Grafana dashboards to S3 (monitoring-init.sh polls for these) ==="
aws s3 sync "${MONITORING_DIR}/grafana/dashboards/" "s3://${BUCKET}/monitoring/grafana/dashboards/"

echo
echo "=== Provisioning complete ==="
echo "S3 bucket:    s3://${BUCKET}"
echo "Grafana:      ${GRAFANA_URL}  (anonymous viewer access -- give the monitoring"
echo "              instance a few minutes to finish boot + dashboard sync)"
terraform output
echo
echo "Next: generate benchmark data (scripts/generate_and_upload_data.sh) if this"
echo "bucket doesn't already have it, then run the benchmark (runner/aws_benchmark_runner.py)."
echo "Remember to run scripts/teardown.sh when done -- these instances bill hourly."
