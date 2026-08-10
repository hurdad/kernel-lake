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

# awscli: for scripts/measure_s3_throughput.sh (raw S3 GET throughput,
# run directly on this host over SSH -- see docs/RUNBOOK.md), independent
# of anything kernellake-server itself does. Native apt package rather
# than a `docker run` container, since the default IMDSv2 hop limit (1)
# blocks a container's own network namespace from reaching the instance
# metadata service for credentials unless the hop limit is explicitly
# raised -- this instance's own IAM role works out of the box for a
# process running directly on the host, no such caveat.
apt-get update
apt-get install -y --no-install-recommends awscli

mkdir -p /opt/kernellake-bench
cd /opt/kernellake-bench

# --- NVMe cache tier: format + mount the instance's local NVMe instance
# store (if present) and point kernellake-server's storage.cache at it --
# ephemeral, high-throughput local SSD for repeat-scan reads, separate
# from the root EBS volume (see include/kernellake/common/config.hpp's
# CacheSection and docs/ARCHITECTURE.md's "NVMe cache tier" section for
# the feature itself). Detected via /dev/disk/by-id/*Instance_Storage*,
# not a hardcoded device name or assumed instance-type spec -- Nitro
# instances present the root EBS volume as an nvme* device too, so name
# alone can't tell them apart, and not every instance type/size actually
# ships local instance storage, so this must be checked for real at boot,
# not assumed from the instance type string. Multiple instance-store
# devices (some instance types ship more than one) are striped with mdadm
# RAID0 for combined capacity/throughput -- this cache has no redundancy
# requirement of its own, S3 is still the durable source of truth.
NVME_CACHE_HOST_DIR=""
NVME_CACHE_MAX_BYTES=0
NVME_CACHE_ENABLED=false

mapfile -t INSTANCE_STORE_DEVICES < <(
  for link in /dev/disk/by-id/*Instance_Storage*; do
    [ -e "$link" ] || continue
    readlink -f "$link"
  done | sort -u
)

if [ "$${#INSTANCE_STORE_DEVICES[@]}" -gt 0 ]; then
  echo "=== Found $${#INSTANCE_STORE_DEVICES[@]} local NVMe instance-store device(s): $${INSTANCE_STORE_DEVICES[*]} ==="
  NVME_CACHE_HOST_DIR="/mnt/kernellake-nvme-cache"
  mkdir -p "$NVME_CACHE_HOST_DIR"

  if [ "$${#INSTANCE_STORE_DEVICES[@]}" -eq 1 ]; then
    CACHE_BLOCK_DEVICE="$${INSTANCE_STORE_DEVICES[0]}"
  else
    apt-get install -y --no-install-recommends mdadm
    CACHE_BLOCK_DEVICE="/dev/md/kernellake-cache"
    yes | mdadm --create "$CACHE_BLOCK_DEVICE" --level=0 \
      --raid-devices="$${#INSTANCE_STORE_DEVICES[@]}" "$${INSTANCE_STORE_DEVICES[@]}"
    udevadm settle
  fi

  # Some AMIs (e.g. AWS's Deep Learning AMIs) auto-mount instance store on
  # boot via their own fstab/udev rule -- unmount first if so, otherwise
  # this mkfs would format a device the kernel considers busy and fail.
  if mount | grep -q "^$CACHE_BLOCK_DEVICE "; then
    echo "=== $CACHE_BLOCK_DEVICE was already mounted by the AMI -- unmounting before reformatting ==="
    umount "$CACHE_BLOCK_DEVICE"
  fi

  mkfs.xfs -f "$CACHE_BLOCK_DEVICE"
  mount "$CACHE_BLOCK_DEVICE" "$NVME_CACHE_HOST_DIR"
  chmod 1777 "$NVME_CACHE_HOST_DIR"  # container runs as non-root; world-writable+sticky, same convention as /tmp

  # 90% of the real formatted filesystem size, not a guessed constant --
  # leaves headroom for filesystem overhead/metadata so the cache's own
  # eviction logic (LRU-by-mtime, src/storage/nvme_object_cache.cpp) never
  # races a hard ENOSPC from the OS itself.
  FS_BYTES=$(df --output=size -B1 "$NVME_CACHE_HOST_DIR" | tail -1 | tr -d ' ')
  NVME_CACHE_MAX_BYTES=$(( FS_BYTES * 90 / 100 ))
  NVME_CACHE_ENABLED=true
  echo "=== NVMe cache mounted at $NVME_CACHE_HOST_DIR ($FS_BYTES formatted bytes, $NVME_CACHE_MAX_BYTES byte cache budget) ==="
else
  echo "=== No local NVMe instance-store device found on this instance -- storage.cache stays disabled ==="
  NVME_CACHE_HOST_DIR="/opt/kernellake-bench/cache-disabled-placeholder"
  mkdir -p "$NVME_CACHE_HOST_DIR"
fi

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
  cache:
    enabled: $${NVME_CACHE_ENABLED}
    directory: /cache
    max_size_bytes: $${NVME_CACHE_MAX_BYTES}
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
      - $${NVME_CACHE_HOST_DIR}:/cache
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
