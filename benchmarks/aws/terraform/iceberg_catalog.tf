# Real Iceberg REST catalog (tabulario/iceberg-rest, the reference
# implementation), for the Iceberg-table-format comparison dimension
# (benchmarks/aws/README.md / docs/RUNBOOK.md -- data partitioned by
# years(l_shipdate)/years(o_orderdate), to actually exercise KernelLake's
# manifest-level partition pruning (src/iceberg/partition_pruning.cpp),
# unlike a plain Hive-partitioned directory layout, which only makes
# partition columns queryable and does NOT skip files -- confirmed by
# reading the actual pruning code before building this).
#
# Deliberately its own standalone instance, not folded into the monitoring
# instance: monitoring_instance.tf's user_data directly references
# aws_instance.kernellake[*]/spark_master/spark_worker's private IPs (for
# Prometheus scrape targets), which makes it un-appliable until those
# GPU/Spark instances exist. The Iceberg catalog has no such dependency --
# both the data-generation instance (writing tables via pyiceberg) and,
# later, kernellake-server/Spark (reading them) only need this instance's
# IP and port 8181, so it can stand up and start hosting real Iceberg
# tables immediately, independent of GPU-instance quota approval.
#
# Catalog metadata backend: the image's default in-memory JDBC (H2) store
# -- resets on container restart, but the actual Iceberg data/metadata
# files it tracks live durably on S3 regardless. Fine for this benchmark's
# ephemeral, single-session lifetime; not intended to survive a restart.

resource "aws_instance" "iceberg_catalog" {
  ami                         = data.aws_ami.ubuntu_26_04.id
  instance_type               = var.iceberg_catalog_instance_type
  subnet_id                   = local.subnet_id
  vpc_security_group_ids      = [aws_security_group.benchmark.id]
  iam_instance_profile        = aws_iam_instance_profile.benchmark_host.name
  key_name                    = var.ssh_key_name
  associate_public_ip_address = true

  root_block_device {
    volume_size = 30
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/user-data/iceberg-catalog-init.sh", {
    s3_bucket  = local.s3_bucket_name
    aws_region = var.aws_region
  })

  tags = {
    Name = "${var.name_prefix}-iceberg-catalog"
    Role = "iceberg-catalog"
  }
}

output "iceberg_catalog_uri" {
  value = "http://${aws_instance.iceberg_catalog.private_ip}:8181"
}

output "iceberg_catalog_public_ip" {
  value = aws_instance.iceberg_catalog.public_ip
}
