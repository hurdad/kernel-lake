# Dedicated PySpark benchmark host -- single-node, CPU-only, running Spark
# in local[*] mode (runner/pyspark_query_loop.py), not a distributed
# standalone cluster. Mirrors duckdb_instance.tf's own shape exactly: one
# instance, one $/hour to attribute a query's cost against, node-count-
# matched against the single KernelLake GPU host and the single DuckDB
# host (see ../README.md's "What this measures" and docs/RUNBOOK.md).
#
# A real multi-node standalone cluster (separate master + worker
# instances, coordinating over spark://<master>:7077) was the original
# design here -- replaced with this single local[*] host because
# local[*] sidesteps a whole class of standalone-mode-specific problems
# this project's own history hit for real on a live run (Master/Worker
# daemon JMX-port collisions when co-located, SPARK_LOCAL_DIRS silently
# overriding driver-side config) and because a distributed cluster isn't
# what "one instance per engine" calls for. See
# runner/pyspark_query_loop.py's own module docstring for the query-loop
# side of this.
#
# m7i.4xlarge by default: cost-matched to the KernelLake GPU host's real,
# live on-demand rate (confirmed via the AWS Pricing API: g6.xlarge
# $0.8048/hr vs. m7i.4xlarge $0.8064/hr, 0.2% apart) -- unlike
# duckdb_instance_type's own historical default, this one *is* meant as a
# same-hourly-cost comparison point. Network performance is not an exact
# match (g6.xlarge is rated "Up to 10 Gigabit", m7i.4xlarge "Up to 12.5
# Gigabit" -- no CPU instance in this price class is rated as low as
# 10 Gigabit, confirmed via a real `aws ec2 describe-instance-types`
# sweep across m7i/c7i/m6i/c6i) -- scripts/measure_s3_throughput.sh run
# against this host (see docs/RUNBOOK.md) measures the real achieved
# throughput rather than relying on that rating.
resource "aws_instance" "spark_host" {
  count = var.enable_spark ? 1 : 0

  ami                         = data.aws_ami.ubuntu_26_04.id
  instance_type               = var.spark_instance_type
  subnet_id                   = local.subnet_id
  vpc_security_group_ids      = [aws_security_group.benchmark.id]
  iam_instance_profile        = aws_iam_instance_profile.benchmark_host.name
  key_name                    = var.ssh_key_name
  associate_public_ip_address = true

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
  }

  # No vars needed: unlike the old master/worker script, spark-local-init.sh
  # only installs the runtime -- the S3 bucket/region are passed to
  # runner/pyspark_query_loop.py as CLI args at SSH-invocation time
  # instead (see docs/RUNBOOK.md), matching duckdb-host-init.sh's own
  # convention.
  user_data = templatefile("${path.module}/user-data/spark-local-init.sh", {})

  tags = {
    Name = "${var.name_prefix}-spark"
    Role = "spark"
  }
}

output "spark_host_public_ip" {
  value = length(aws_instance.spark_host) > 0 ? aws_instance.spark_host[0].public_ip : null
}

output "spark_host_id" {
  value = length(aws_instance.spark_host) > 0 ? aws_instance.spark_host[0].id : null
}
