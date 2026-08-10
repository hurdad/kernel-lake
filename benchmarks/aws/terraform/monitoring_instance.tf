# Small, cheap instance running Prometheus + Grafana (docker-compose).
# Depends explicitly on every KernelLake/Spark instance so their IPs are
# known before this instance's user-data renders prometheus.yml's scrape
# targets -- avoids a partially-populated config from an apply-ordering race.

resource "aws_instance" "monitoring" {
  ami                         = data.aws_ami.ubuntu_26_04.id
  instance_type               = var.monitoring_instance_type
  subnet_id                   = local.subnet_id
  vpc_security_group_ids      = [aws_security_group.benchmark.id]
  iam_instance_profile        = aws_iam_instance_profile.benchmark_host.name
  key_name                    = var.ssh_key_name
  associate_public_ip_address = true

  root_block_device {
    volume_size = 50
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/user-data/monitoring-init.sh", {
    s3_bucket = local.s3_bucket_name
    prometheus_config = templatefile("${path.module}/../monitoring/prometheus.yml.tftpl", {
      kernellake_ips  = aws_instance.kernellake[*].private_ip
      # spark_master/duckdb_host are now optional (enable_spark/enable_duckdb,
      # see variables.tf) -- "" when disabled rather than indexing a 0-count
      # resource, and prometheus.yml.tftpl skips the matching scrape job
      # entirely when it sees an empty string.
      spark_master_ip = length(aws_instance.spark_master) > 0 ? aws_instance.spark_master[0].private_ip : ""
      spark_worker_ips = aws_instance.spark_worker[*].private_ip
      duckdb_host_ip  = length(aws_instance.duckdb_host) > 0 ? aws_instance.duckdb_host[0].private_ip : ""
    })
  })

  depends_on = [aws_instance.kernellake, aws_instance.spark_master, aws_instance.spark_worker, aws_instance.duckdb_host]

  tags = {
    Name = "${var.name_prefix}-monitoring"
    Role = "monitoring"
  }
}

output "monitoring_public_ip" {
  value = aws_instance.monitoring.public_ip
}

output "grafana_url" {
  value = "http://${aws_instance.monitoring.public_ip}:3000"
}
