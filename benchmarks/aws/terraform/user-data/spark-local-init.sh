#!/bin/bash
# EC2 user-data for the dedicated single-node PySpark benchmark host
# (spark_cluster.tf), running Spark in local[*] mode -- no standalone
# master/worker daemons at all, unlike this project's earlier multi-node
# spark-node-init.sh (removed alongside this file's introduction; see
# spark_cluster.tf's own comment for why). Rendered by Terraform's
# templatefile() -- plain dollar-brace interpolation is Terraform-injected;
# a genuine dollar-brace bash sequence needs a doubled-dollar escape for a
# literal one (e.g. $${SPARK_HOME} below).
set -euo pipefail
exec > >(tee /var/log/spark-local-init.log) 2>&1

echo "=== Spark local[*] host init starting $(date -u) ==="

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends openjdk-17-jre-headless python3 python3-pip curl unzip

# Official AWS CLI v2 installer, NOT the Ubuntu 26.04 apt package -- see
# duckdb-host-init.sh's own comment for the real, confirmed-live
# "badly formed help string" bug this avoids (apt's aws-cli/2.31.35
# "source" build breaks every `aws s3api` call, including the one
# ../../scripts/measure_s3_throughput.sh needs to list real objects).
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
/tmp/aws/install

# pyspark's own pip package bundles enough of the Spark distribution to
# run in local[*] mode -- no separate daemon scripts needed at all in this
# mode (unlike the old standalone-cluster setup, which needed
# sbin/spark-daemon.sh). pandas: runner/pyspark_query_loop.py's
# run_pyspark_query() calls spark.sql(...).toPandas() -- a real, confirmed
# gap on a fresh instance otherwise (ImportError: Pandas >= 1.0.5 must be
# installed), not an assumed pyspark dependency. pyarrow: that same
# function's pa.Table.from_pandas(...) call right after toPandas() --
# another real, confirmed-live gap (ModuleNotFoundError on this benchmark's
# very first real run against the rewritten local[*] topology), same
# category as duckdb-host-init.sh's own pyarrow fix. boto3: any
# Python-side S3 access (the aws CLI itself is installed separately
# above).
pip3 install --break-system-packages pyspark==3.5.3 boto3 pandas pyarrow

SPARK_HOME=$(python3 -c "import pyspark, os; print(os.path.dirname(pyspark.__file__))")
echo "SPARK_HOME=$${SPARK_HOME}"

# Real disk-backed shuffle-spill dir, same reasoning as the old
# standalone-cluster init script: /tmp can be RAM-backed tmpfs with only a
# few GB total on some AMIs, and a SF100 3-way join's shuffle spill blows
# through that well before the real ~90GB-free root EBS volume is
# anywhere close to full. runner/pyspark_query_loop.py's own
# new_spark_session() points spark.local.dir here explicitly.
mkdir -p /var/spark-tmp
chmod 1777 /var/spark-tmp

# node_exporter, same as every other benchmark host, for host-level CPU/
# memory/network/disk metrics in the central Prometheus -- see
# ../../monitoring/prometheus.yml.tftpl's "spark-host-node" job. No JMX
# exporter here (unlike the old master/worker daemons): queries run as
# short-lived SSH-invoked Python processes
# (runner/pyspark_query_loop.py), same shape as the DuckDB host, with
# nothing long-lived to instrument via JMX.
# Pinned to a specific release (not /releases/latest/download/, which
# 404's the moment a newer release ships and the "latest" redirect's
# actual asset filename no longer matches a hardcoded old one -- confirmed
# for real on this project's KernelLake/Spark host init scripts before
# this convention was adopted everywhere).
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

echo "=== Spark local[*] host init complete $(date -u) ==="
