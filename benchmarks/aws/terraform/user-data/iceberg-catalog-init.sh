#!/bin/bash
# EC2 user-data for the Iceberg REST catalog instance. Rendered by
# Terraform's templatefile() (iceberg_catalog.tf) -- plain dollar-brace
# interpolation is Terraform-injected. Only a genuine dollar-brace
# sequence needs a doubled-dollar escape for a literal one -- plain
# command substitution ($(...)) and bare variable references need no
# escaping, since Terraform's templatefile() never looks at them. An
# earlier version of this file doubled every bash-side dollar sign
# regardless, which rendered as literal, unevaluated bash-PID-plus-text on
# a real instance boot -- caught only via `terraform apply` + SSH, not
# `terraform validate`/`plan` (HCL/Terraform syntax only, not the
# rendered script's own bash validity).
set -euo pipefail
exec > >(tee /var/log/iceberg-catalog-init.log) 2>&1

echo "=== Iceberg REST catalog init starting $(date -u) ==="

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends docker.io docker-compose-v2
systemctl enable --now docker

mkdir -p /opt/kernellake-bench
cd /opt/kernellake-bench

# tabulario/iceberg-rest: the reference REST catalog implementation.
# CATALOG_IO__IMPL=S3FileIO + AWS_REGION, no explicit AWS_ACCESS_KEY_ID/
# SECRET -- picks up this instance's own IAM instance profile via the AWS
# SDK for Java v2's default credential chain, same "default" convention
# already used by kernellake-server's own S3Section (config.hpp).
cat > docker-compose.yml <<EOF
services:
  iceberg-rest:
    image: tabulario/iceberg-rest:latest
    network_mode: host
    environment:
      - AWS_REGION=${aws_region}
      - CATALOG_WAREHOUSE=s3://${s3_bucket}/warehouse/
      - CATALOG_IO__IMPL=org.apache.iceberg.aws.s3.S3FileIO
      - CATALOG_S3_REGION=${aws_region}
    restart: unless-stopped
EOF

echo "=== Pulling image ==="
docker compose pull

echo "=== Starting Iceberg REST catalog ==="
docker compose up -d

echo "=== Iceberg REST catalog init complete $(date -u) ==="
