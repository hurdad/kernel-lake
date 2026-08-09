#!/usr/bin/env bash
# Launches a short-lived EC2 instance that reads an already-generated flat
# TPC-H dataset from S3 (generate_and_upload_data.sh must have already run
# for the same --scale-factor/--source-compression) and writes it into real
# Iceberg tables via runner/generate_and_upload_iceberg_data.py, against the
# standalone REST catalog instance (terraform/iceberg_catalog.tf). Then
# terminates itself, same self-terminating pattern as
# generate_and_upload_data.sh.
#
# Usage: generate_and_upload_iceberg_data.sh --scale-factor 100 [--wait]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TERRAFORM_DIR="${SCRIPT_DIR}/../terraform"

SCALE_FACTOR=""
WAIT=false
INSTANCE_TYPE="r6i.2xlarge"
SOURCE_COMPRESSION="snappy"
while [ $# -gt 0 ]; do
  case "$1" in
    --scale-factor) SCALE_FACTOR="$2"; shift 2 ;;
    --wait) WAIT=true; shift ;;
    --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
    --source-compression) SOURCE_COMPRESSION="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done
if [ -z "$SCALE_FACTOR" ]; then
  echo "Usage: $0 --scale-factor <N> [--wait] [--instance-type <type>] [--source-compression snappy|zstd]" >&2
  exit 1
fi

cd "$TERRAFORM_DIR"
BUCKET="$(terraform output -raw s3_bucket_name)"
REGION="$(terraform output -raw aws_region)"
SUBNET_ID="$(terraform output -raw subnet_id)"
SG_ID="$(terraform output -raw security_group_id)"
PROFILE="$(terraform output -raw iam_instance_profile_name)"
SSH_KEY="$(terraform output -raw ssh_key_name)"
CATALOG_URI="$(terraform output -raw iceberg_catalog_uri)"
cd - >/dev/null

# Fail fast with a clear message rather than launching an instance that
# will fail once it tries to actually connect -- this whole script needs
# terraform/iceberg_catalog.tf's aws_instance.iceberg_catalog to already
# be up (see docs/RUNBOOK.md).
if [ -z "$CATALOG_URI" ] || [ "$CATALOG_URI" = "http://:8181" ]; then
  echo "iceberg_catalog_uri terraform output is empty -- apply aws_instance.iceberg_catalog first" >&2
  exit 1
fi
WAREHOUSE="s3://${BUCKET}/warehouse/"

echo "=== Uploading generate_and_upload_iceberg_data.py to s3://${BUCKET}/scripts/ ==="
aws s3 cp "${SCRIPT_DIR}/../runner/generate_and_upload_iceberg_data.py" "s3://${BUCKET}/scripts/generate_and_upload_iceberg_data.py"

USER_DATA=$(cat <<EOF
#!/bin/bash
set -euo pipefail
exec > >(tee /var/log/generate-iceberg-data.log) 2>&1
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends python3 python3-pip python3-dev build-essential awscli
pip3 install --break-system-packages 'pyiceberg[sql-sqlite,pyiceberg-core]' pyarrow boto3

aws s3 cp "s3://${BUCKET}/scripts/generate_and_upload_iceberg_data.py" /tmp/generate_and_upload_iceberg_data.py

echo "=== Writing Iceberg tables at SF${SCALE_FACTOR} (source: ${SOURCE_COMPRESSION}) ==="
python3 /tmp/generate_and_upload_iceberg_data.py \\
  --s3-bucket ${BUCKET} --scale-factor ${SCALE_FACTOR} \\
  --source-compression ${SOURCE_COMPRESSION} \\
  --catalog-uri ${CATALOG_URI} --warehouse ${WAREHOUSE} --aws-region ${REGION}

# Explicit completion marker -- same rationale as
# generate_and_upload_data.sh's own (an instance terminating early on
# error looks indistinguishable from success otherwise).
echo '{"scale_factor": ${SCALE_FACTOR}, "status": "complete"}' | \\
  aws s3 cp - "s3://${BUCKET}/tpch-iceberg/sf${SCALE_FACTOR}/.generation-complete"

# IMDSv2 (token-based) -- see generate_and_upload_data.sh's own comment on
# this same fix: a plain unauthenticated IMDSv1-style GET silently returns
# an empty body against this account's instances (HttpTokens: required),
# which left two real instances running/billing indefinitely instead of
# self-terminating.
IMDS_TOKEN=\$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
INSTANCE_ID=\$(curl -s -H "X-aws-ec2-metadata-token: \${IMDS_TOKEN}" http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --region ${REGION} --instance-ids "\${INSTANCE_ID}"
EOF
)

echo "=== Launching ${INSTANCE_TYPE} Iceberg-writer instance (SF${SCALE_FACTOR}) ==="
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
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":30,"VolumeType":"gp3"}}]' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=kernellake-bench-iceberg-sf${SCALE_FACTOR}},{Key=Ephemeral,Value=true}]" \
  --user-data "$USER_DATA" \
  --query 'Instances[0].InstanceId' --output text)

echo "Launched: ${INSTANCE_ID} (on-demand -- self-terminates on completion or failure)"
echo "Progress: check s3://${BUCKET}/tpch-iceberg/sf${SCALE_FACTOR}/.generation-complete"

if [ "$WAIT" = true ]; then
  echo "=== Waiting for completion marker (--wait given) ==="
  while ! aws s3 ls "s3://${BUCKET}/tpch-iceberg/sf${SCALE_FACTOR}/.generation-complete" >/dev/null 2>&1; do
    if ! aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null | grep -qv "terminated\|shutting-down"; then
      echo "Instance is no longer running and no completion marker was found -- generation likely failed." >&2
      echo "Check /var/log/generate-iceberg-data.log on the instance (if still reachable) or CloudWatch." >&2
      exit 1
    fi
    sleep 30
  done
  echo "=== Iceberg data generation complete: catalog namespace tpch_sf${SCALE_FACTOR} ==="
fi
