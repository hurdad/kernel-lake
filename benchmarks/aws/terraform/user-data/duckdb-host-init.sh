#!/bin/bash
# EC2 user-data for the dedicated DuckDB benchmark host. Rendered by
# Terraform's templatefile() (duckdb_instance.tf) -- plain dollar-brace
# interpolation is Terraform-injected; see spark-node-init.sh's own
# comment for the escaping rule this follows.
#
# This instance runs no long-lived service of its own -- it just needs
# Python/DuckDB/boto3 installed and ready. runner/duckdb_query_loop.py is
# scp'd here and invoked over SSH per run (see docs/RUNBOOK.md), the same
# way the KernelLake/Spark hosts are driven from the orchestrator rather
# than pulling their own work.
set -euo pipefail
exec > >(tee /var/log/duckdb-host-init.log) 2>&1

echo "=== DuckDB host init starting $(date -u) ==="

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends python3 python3-pip curl

pip3 install --break-system-packages duckdb boto3

# node_exporter, same as the KernelLake/Spark hosts, for host-level
# CPU/memory/network/disk metrics in the same central Prometheus -- see
# spark-node-init.sh's own comment for why the release is pinned rather
# than fetched via a "latest" redirect URL.
curl -fsSL -o /tmp/node_exporter.tar.gz \
  https://github.com/prometheus/node_exporter/releases/download/v1.12.1/node_exporter-1.12.1.linux-amd64.tar.gz \
  || echo "node_exporter download failed, continuing without it (host metrics won't appear for this node)"
if [ -f /tmp/node_exporter.tar.gz ]; then
  tar xzf /tmp/node_exporter.tar.gz -C /tmp
  cp /tmp/node_exporter-*/node_exporter /usr/local/bin/
  cat > /etc/systemd/system/node_exporter.service <<'EOF'
[Unit]
Description=node_exporter
[Service]
ExecStart=/usr/local/bin/node_exporter --web.listen-address=:9100
Restart=always
[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable --now node_exporter
fi

mkdir -p /opt/kernellake-bench

echo "=== DuckDB host init complete $(date -u) ==="
