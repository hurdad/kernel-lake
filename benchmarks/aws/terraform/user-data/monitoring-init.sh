#!/bin/bash
# EC2 user-data for the Prometheus/Grafana monitoring instance. Rendered by
# Terraform's templatefile() (monitoring_instance.tf) -- plain dollar-brace
# interpolation is Terraform-injected (prometheus_config, itself already
# the *fully rendered* prometheus.yml from ../monitoring/prometheus.yml.tftpl).
# Only a genuine dollar-brace sequence needs a doubled-dollar escape for a
# literal one -- plain command substitution ($(...)) and bare variable
# references need no escaping, since Terraform's templatefile() never
# looks at them. An earlier version of this file doubled every bash-side
# dollar sign regardless, which rendered as literal, unevaluated
# bash-PID-plus-text on a real instance boot -- caught only via `terraform
# apply` + SSH, not `terraform validate`/`plan` (HCL/Terraform syntax
# only, not the rendered script's own bash validity).
set -euo pipefail
exec > >(tee /var/log/monitoring-init.log) 2>&1

echo "=== Monitoring instance init starting $(date -u) ==="

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends docker.io docker-compose-v2 awscli
systemctl enable --now docker

mkdir -p /opt/kernellake-bench/grafana/provisioning/datasources
mkdir -p /opt/kernellake-bench/grafana/provisioning/dashboards
mkdir -p /opt/kernellake-bench/grafana/dashboards
cd /opt/kernellake-bench

cat > prometheus.yml <<'PROMEOF'
${prometheus_config}
PROMEOF

cat > grafana/provisioning/datasources/datasources.yml <<'EOF'
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

cat > grafana/provisioning/dashboards/dashboards.yml <<'EOF'
apiVersion: 1
providers:
  - name: kernellake-bench
    folder: KernelLake Benchmark
    type: file
    updateIntervalSeconds: 30
    options:
      path: /var/lib/grafana/dashboards
EOF

cat > docker-compose.yml <<'EOF'
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    ports:
      - "9090:9090"
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - ./grafana/dashboards:/var/lib/grafana/dashboards:ro
    ports:
      - "3000:3000"
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer
    restart: unless-stopped
EOF

docker compose pull
docker compose up -d

# Dashboard JSONs live in this repo (monitoring/grafana/dashboards/*.json),
# not embedded in user-data (EC2's 16KB user-data limit, plus wanting
# dashboard edits to not require re-launching the instance). provision.sh
# uploads them to s3://<bucket>/monitoring/grafana/dashboards/ right after
# `terraform apply` returns -- this loop tolerates that happening a few
# seconds/minutes after this instance has already booted, rather than
# requiring a strict ordering guarantee between the two.
echo "=== Waiting for dashboard JSONs in S3 (up to 5 minutes) ==="
for _ in $(seq 1 20); do
  if aws s3 sync "s3://${s3_bucket}/monitoring/grafana/dashboards/" grafana/dashboards/ 2>/dev/null \
     && [ -n "$(ls -A grafana/dashboards/ 2>/dev/null)" ]; then
    echo "Dashboards synced."
    break
  fi
  sleep 15
done

echo "=== Monitoring instance init complete $(date -u) ==="
