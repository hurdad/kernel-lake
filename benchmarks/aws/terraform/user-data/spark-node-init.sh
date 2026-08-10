#!/bin/bash
# EC2 user-data for a self-managed Spark standalone node (master or
# worker). Rendered by Terraform's templatefile() (spark_cluster.tf) --
# plain dollar-brace interpolation is Terraform-injected. Only a genuine
# dollar-brace sequence needs a doubled-dollar escape for a literal one
# (e.g. $${SPARK_HOME} below, bash's own brace-variable syntax) -- plain
# command substitution ($(...)) and bare variable references need no
# escaping, since Terraform's templatefile() never looks at them. An
# earlier version of this file doubled every bash-side dollar sign
# regardless, which rendered as literal, unevaluated bash-PID-plus-text on
# a real instance boot -- caught only via `terraform apply` + SSH, not
# `terraform validate`/`plan` (HCL/Terraform syntax only, not the rendered
# script's own bash validity).
set -euo pipefail
exec > >(tee /var/log/spark-node-init.log) 2>&1

echo "=== Spark node init (role=${role}) starting $(date -u) ==="

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends openjdk-17-jre-headless python3 python3-pip curl

# pyspark's own pip package bundles enough of the Spark distribution to
# run standalone master/worker daemons (sbin/spark-daemon.sh, the jars) --
# simpler and more reproducible than hand-downloading a Spark tarball from
# an Apache mirror (mirror URLs/availability drift over time; pip's
# package pinning doesn't). It does NOT include the start-master.sh/
# start-worker.sh wrapper scripts themselves (confirmed for real, not
# assumed -- see this file's own start-up section below for the fix:
# spark-daemon.sh directly, which is what those wrappers call anyway).
# pandas: aws_benchmark_runner.py's run_pyspark_query() calls
# spark.sql(...).toPandas() -- a real, confirmed-live gap
# (ImportError: Pandas >= 1.0.5 must be installed) on a fresh instance,
# not an assumed pyspark dependency (pyspark's own pip package doesn't
# pull it in).
pip3 install --break-system-packages pyspark==3.5.3 boto3 pandas

SPARK_HOME=$(python3 -c "import pyspark, os; print(os.path.dirname(pyspark.__file__))")
echo "SPARK_HOME=$${SPARK_HOME}"

# Real disk-backed shuffle-spill dir for the Worker (and, transitively,
# the executors it launches). SPARK_LOCAL_DIRS (a Worker-process
# environment variable) overrides spark.local.dir (a driver-side
# SparkConf entry, e.g. runner/aws_benchmark_runner.py's
# new_spark_session()) entirely in Spark standalone mode -- confirmed for
# real that setting only the driver-side config had zero effect,
# because /tmp (java.io.tmpdir's default, and what the Worker used
# instead) turned out to be RAM-backed tmpfs on this AMI with only ~8GB
# total, not real disk -- a SF100 3-way join's shuffle spill blew through
# it ("No space left on device") well before the real ~90GB-free root EBS
# volume was anywhere close to full.
mkdir -p /var/spark-tmp
chmod 1777 /var/spark-tmp

# node_exporter, same as the KernelLake host, for host-level CPU/memory/
# network/disk metrics in the same central Prometheus.
# Pinned to a specific release (not /releases/latest/download/, which
# redirects to whatever the current latest tag is) -- the previous version
# here mixed the two: a "latest" URL path with a hardcoded old filename
# (1.8.2), which 404's the moment a newer release ships, since the
# "latest" release's actual asset filename no longer matches. Confirmed
# for real (a live 404 on a running instance) before this fix, and
# confirmed the real current latest tag (v1.12.1) via the same redirect
# this bug relied on, rather than guessing a new pin.
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

# Spark's own JMX metrics (executor/task-level detail beyond node_exporter's
# host-level view) via the built-in metrics servlet + a Prometheus JMX
# exporter javaagent -- exposed on :7777, scraped by the central Prometheus
# same as node_exporter/dcgm-exporter.
curl -fsSL -o /opt/jmx_prometheus_javaagent.jar \
  https://repo1.maven.org/maven2/io/prometheus/jmx/jmx_prometheus_javaagent/0.20.0/jmx_prometheus_javaagent-0.20.0.jar \
  || echo "JMX exporter download failed, continuing without Spark-internal metrics"
# No hostPort here: confirmed for real (a live "SEVERE: Configuration
# error: When running jmx_exporter as a Java agent, you must not configure
# 'jmxUrl' or 'hostPort' because you don't want to monitor a remote JVM"
# failure, which silently prevented spark-daemon.sh from ever starting the
# master/worker process at all) that hostPort is only valid for the
# exporter's *standalone* mode (monitoring a separate, remote JVM) -- a
# javaagent already runs inside the JVM it's monitoring and picks that up
# automatically, so setting hostPort here isn't just redundant, it's a
# hard startup error.
cat > /opt/jmx-exporter-config.yaml <<'EOF'
startDelaySeconds: 0
lowercaseOutputName: true
EOF

SPARK_DAEMON_JAVA_OPTS="-javaagent:/opt/jmx_prometheus_javaagent.jar=7777:/opt/jmx-exporter-config.yaml"
export SPARK_DAEMON_JAVA_OPTS

# start-master.sh/start-worker.sh, not just spark-daemon.sh directly:
# confirmed for real that pyspark's pip wheel does NOT ship those two
# wrapper scripts (sbin/ only has spark-config.sh, spark-daemon.sh,
# start-history-server.sh, stop-history-server.sh -- no start-master.sh/
# start-worker.sh/start-slave.sh, unlike the full Apache tarball
# distribution). spark-daemon.sh IS shipped and is exactly what those
# wrapper scripts call internally (start-master.sh is a thin wrapper
# around `spark-daemon.sh start org.apache.spark.deploy.master.Master ...`,
# same for start-worker.sh/Worker) -- verified end to end on a real
# instance (master + all 3 workers registered ALIVE) before fixing this
# file, not assumed.
if [ "${role}" = "master" ]; then
  OWN_IP="$(hostname -I | awk '{print $1}')"
  echo "=== Starting Spark standalone master ==="
  "$${SPARK_HOME}/sbin/spark-daemon.sh" start org.apache.spark.deploy.master.Master 1 --host "$${OWN_IP}"

  # Also start a worker on the master itself, registered against its own
  # master URI. This makes spark_worker_count=0 a genuine, self-sufficient
  # single-node standalone cluster (real executors to run queries on)
  # instead of a master with nothing registered to it, which would hang
  # forever waiting for resources on the first job submitted. Harmless
  # when spark_worker_count > 0 too -- just one more worker alongside the
  # dedicated ones, so the cost-matched multi-node mode is unaffected.
  #
  # The worker needs its own SPARK_DAEMON_JAVA_OPTS with a different JMX
  # port than the master's (both daemons otherwise inherit the same
  # exported env var, so the second one to start fatally crashes on a
  # port collision -- confirmed for real on a live instance:
  # "java.net.BindException: Address already in use" -> a JVM
  # "ASSERTION FAILED"/native fatal error, not just a graceful skip).
  echo "=== Also starting a Spark worker on the master node itself ==="
  SPARK_DAEMON_JAVA_OPTS="-javaagent:/opt/jmx_prometheus_javaagent.jar=7778:/opt/jmx-exporter-config.yaml" \
    SPARK_LOCAL_DIRS="/var/spark-tmp" \
    "$${SPARK_HOME}/sbin/spark-daemon.sh" start org.apache.spark.deploy.worker.Worker 1 "spark://$${OWN_IP}:7077"
else
  echo "=== Starting Spark standalone worker, joining master at ${master_ip} ==="
  SPARK_LOCAL_DIRS="/var/spark-tmp" \
    "$${SPARK_HOME}/sbin/spark-daemon.sh" start org.apache.spark.deploy.worker.Worker 1 "spark://${master_ip}:7077"
fi

echo "=== Spark node init (role=${role}) complete $(date -u) ==="
