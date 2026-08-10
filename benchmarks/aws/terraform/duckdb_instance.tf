# Dedicated DuckDB benchmark host -- single-node, CPU-only, non-GPU,
# node-count-matched against the single KernelLake GPU host (see
# ../README.md's "What this measures" and docs/RUNBOOK.md). Previously
# DuckDB ran in-process on whichever machine happened to run the
# orchestrator script, which had no dedicated $/hour to attribute a
# query's cost against -- this instance exists so DuckDB gets a real,
# cost-tracked box the same way KernelLake and Spark do.
#
# m7i.xlarge by default: same type as the Spark master, not an attempt to
# cost-match the GPU host's own rate -- DuckDB is a single-threaded-ish,
# in-process CPU engine, and oversizing it doesn't change what's being
# measured (its own single-node ceiling), just the cost line. Override
# duckdb_instance_type if a stricter "same vCPU/RAM as the KernelLake
# host" comparison is wanted instead.
resource "aws_instance" "duckdb_host" {
  count = var.enable_duckdb ? 1 : 0

  ami                         = data.aws_ami.ubuntu_26_04.id
  instance_type               = var.duckdb_instance_type
  subnet_id                   = local.subnet_id
  vpc_security_group_ids      = [aws_security_group.benchmark.id]
  iam_instance_profile        = aws_iam_instance_profile.benchmark_host.name
  key_name                    = var.ssh_key_name
  associate_public_ip_address = true

  root_block_device {
    volume_size = 30
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/user-data/duckdb-host-init.sh", {})

  tags = {
    Name = "${var.name_prefix}-duckdb"
    Role = "duckdb"
  }
}

output "duckdb_host_public_ip" {
  value = length(aws_instance.duckdb_host) > 0 ? aws_instance.duckdb_host[0].public_ip : null
}

output "duckdb_host_id" {
  value = length(aws_instance.duckdb_host) > 0 ? aws_instance.duckdb_host[0].id : null
}
