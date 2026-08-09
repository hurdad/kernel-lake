#!/usr/bin/env bash
# `terraform destroy` wrapper. The S3 data bucket is kept by default
# (aws_s3_bucket.benchmark_data's own force_destroy = false, see
# ../terraform/s3.tf) -- a plain teardown.sh run destroys every billed EC2
# instance but leaves the bucket (and a `terraform destroy` error naming
# it, which is expected and fine) so TB-scale generated data is never
# silently deleted. Pass --purge-data to actually empty and remove it too.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAFORM_DIR="${SCRIPT_DIR}/../terraform"

PURGE_DATA=false
for arg in "$@"; do
  case "$arg" in
    --purge-data) PURGE_DATA=true ;;
    *) ;;
  esac
done

cd "$TERRAFORM_DIR"

if [ ! -f terraform.tfstate ] && [ -z "$(terraform state list 2>/dev/null || true)" ]; then
  echo "No Terraform state found in ${TERRAFORM_DIR} -- nothing to tear down."
  exit 0
fi

# Safety check: confirm every resource Terraform is about to destroy
# actually belongs to this project (the Ephemeral=true default tag set in
# ../terraform/main.tf's provider block) -- catches the case of an
# accidentally-shared/misconfigured state file pointing at resources this
# script shouldn't touch.
echo "=== Resources Terraform will destroy ==="
terraform plan -destroy -no-color | tee /tmp/kernellake-bench-destroy-plan.txt
if ! grep -q "Ephemeral" /tmp/kernellake-bench-destroy-plan.txt 2>/dev/null; then
  echo "WARNING: destroy plan doesn't visibly reference the Ephemeral tag this project's" >&2
  echo "resources all carry -- double check this state file belongs to this benchmark" >&2
  echo "before continuing." >&2
fi

if [ "$PURGE_DATA" = true ]; then
  BUCKET="$(terraform output -raw s3_bucket_name 2>/dev/null || true)"
  if [ -n "$BUCKET" ]; then
    echo "=== --purge-data given: emptying s3://${BUCKET} before destroy ==="
    aws s3 rm "s3://${BUCKET}" --recursive
  fi
fi

echo "=== terraform destroy ==="
terraform destroy -auto-approve

echo
echo "Teardown complete."
if [ "$PURGE_DATA" != true ]; then
  echo "S3 data bucket was left in place (pass --purge-data to also remove it)."
fi
