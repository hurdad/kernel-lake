#!/bin/bash
# EC2 user-data for a KernelLake GPU benchmark host. Rendered by Terraform's
# templatefile() (kernellake_instance.tf) -- Terraform-injected values use
# plain dollar-brace interpolation. Terraform's templatefile() only treats a
# dollar-brace sequence as special (needing a doubled-dollar escape for a
# literal one) -- plain bash command substitution and bare variable
# references need no escaping at all, since Terraform never looks at them.
# Confirmed the hard way: an earlier version of this file doubled every
# bash-side dollar sign on the mistaken assumption that ALL of them needed
# escaping, which rendered as literal, unevaluated bash-PID-plus-text on
# every real instance boot (caught only by an actual `terraform apply` and
# SSH into the running instance, not by `terraform validate`/`plan`, which
# only check HCL/Terraform syntax, never the rendered script's own bash
# validity).
set -euo pipefail
exec > >(tee /var/log/kernellake-host-init.log) 2>&1

echo "=== KernelLake host init starting $(date -u) ==="

# Real fallback, not just a check: ami.tf tries a Deep Learning Base AMI
# for Ubuntu 26.04 first, but that AMI may not exist yet for a release
# this new (unverified from here -- see ami.tf's own comment). If
# kernellake_ami_id ends up pointing at a *plain* Ubuntu 26.04 image
# instead (no NVIDIA driver/CUDA/Docker/nvidia-container-toolkit
# pre-baked), this installs all four from scratch rather than failing.
export DEBIAN_FRONTEND=noninteractive

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found -- installing (plain-AMI fallback path)"
  apt-get update
  apt-get install -y --no-install-recommends ca-certificates curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
  apt-get update
  apt-get install -y --no-install-recommends docker-ce docker-ce-cli containerd.io docker-compose-plugin
  systemctl enable --now docker
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "docker compose plugin not found -- installing"
  apt-get update && apt-get install -y --no-install-recommends docker-compose-plugin
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA driver not found -- installing from scratch (plain-AMI fallback path)"
  apt-get update
  apt-get install -y --no-install-recommends build-essential linux-headers-$(uname -r)
  # ubuntu-drivers-common's `ubuntu-drivers install` picks the recommended
  # driver for the detected GPU (the L4 here) automatically, avoiding a
  # hardcoded driver-version pin that would drift out of date.
  apt-get install -y --no-install-recommends ubuntu-drivers-common
  ubuntu-drivers install
  echo "NVIDIA driver installed -- this instance needs a reboot to load it."
  echo "Re-running the rest of this script after reboot via a systemd unit."
  cat > /etc/systemd/system/kernellake-host-init-resume.service <<'EOF'
[Unit]
Description=Resume kernellake-host-init.sh after NVIDIA driver reboot
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/var/lib/cloud/instance/scripts/part-001
ExecStartPost=/bin/systemctl disable kernellake-host-init-resume.service

[Install]
WantedBy=multi-user.target
EOF
  systemctl enable kernellake-host-init-resume.service
  reboot
  exit 0
fi

if ! docker info 2>/dev/null | grep -qi nvidia; then
  echo "nvidia-container-toolkit not found -- installing from scratch (plain-AMI fallback path)"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  apt-get install -y --no-install-recommends nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

mkdir -p /opt/kernellake-bench
cd /opt/kernellake-bench

# --- kernellake-server config -------------------------------------------
# credentials_kind: "default" (S3Section's own default, config.hpp:75-83)
# resolves this instance's IAM instance profile automatically via the AWS
# SDK's default credential chain -- no access key ever touches this file.
cat > kernellake-server.yaml <<CONFIGEOF
engine:
  backend: gpu
storage:
  s3:
    credentials_kind: default
server:
  port: 31337
observability:
  enabled: true
  service_name: kernellake-server
  otlp_protocol: grpc
  otlp_endpoint: "${otlp_endpoint}"
iceberg:
  catalogs:
    bench:
      catalog_uri: "${iceberg_catalog_uri}"
      warehouse: "${iceberg_warehouse}"
      credentials_kind: none
CONFIGEOF

# --- OTel Collector config: OTLP gRPC receiver -> Prometheus exporter ---
# Mirrors ../monitoring/otel-collector-config.yaml (kept in sync by hand,
# same "no build-time link between infra config copies" convention this
# repo's own vendored proto files already use) -- receives kernellake-server's
# real OTel metrics on :4317 and exposes them for the central Prometheus
# instance to scrape on :8889/metrics.
cat > otel-collector-config.yaml <<'OTELEOF'
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
exporters:
  prometheus:
    endpoint: 0.0.0.0:8889
processors:
  batch: {}
service:
  pipelines:
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [prometheus]
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: []
OTELEOF

# --- docker-compose: kernellake-server + otel-collector + node_exporter +
#     dcgm-exporter, all on this one instance -----------------------------
cat > docker-compose.yml <<COMPOSEEOF
services:
  kernellake-server:
    image: ${kernellake_docker_image}
    entrypoint: ["/opt/kernellake/bin/kernellake-server"]
    command: ["--config", "/config/kernellake-server.yaml"]
    network_mode: host
    volumes:
      - ./kernellake-server.yaml:/config/kernellake-server.yaml:ro
    environment:
      - AWS_REGION=${aws_region}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
    depends_on:
      - otel-collector

  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    command: ["--config=/etc/otel-collector-config.yaml"]
    network_mode: host
    volumes:
      - ./otel-collector-config.yaml:/etc/otel-collector-config.yaml:ro
    restart: unless-stopped

  node-exporter:
    image: prom/node-exporter:latest
    network_mode: host
    pid: host
    volumes:
      - /:/host:ro,rslave
    command:
      - "--path.rootfs=/host"
      - "--web.listen-address=:9100"
    restart: unless-stopped

  dcgm-exporter:
    image: nvcr.io/nvidia/k8s/dcgm-exporter:latest
    network_mode: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
COMPOSEEOF

echo "=== Pulling images ==="
docker compose pull

echo "=== Starting kernellake-server + monitoring sidecars ==="
docker compose up -d

echo "=== KernelLake host init complete $(date -u) ==="
