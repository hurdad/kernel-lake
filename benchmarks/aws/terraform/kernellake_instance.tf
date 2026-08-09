# One or more independent KernelLake GPU hosts. Each is a fully
# independent kernellake-server -- there is no coordination between them
# (see README.md's "What this does not measure"); running more than 1 is
# only ever for the M4 concurrency test, never to make a single query
# faster.

resource "aws_instance" "kernellake" {
  count = var.kernellake_instance_count

  ami                         = local.kernellake_ami_id
  instance_type               = var.kernellake_instance_type
  subnet_id                   = local.subnet_id
  vpc_security_group_ids      = [aws_security_group.benchmark.id]
  iam_instance_profile        = aws_iam_instance_profile.benchmark_host.name
  key_name                    = var.ssh_key_name
  associate_public_ip_address = true

  # Local NVMe (g6.8xlarge ships with instance store) is left as-is here --
  # kernellake-server reads directly from S3 (config.hpp's S3Section,
  # credentials_kind="default" resolves this instance's own IAM role
  # automatically, no credential wiring needed), so no local data
  # partitioning/mounting is required for the benchmark itself.
  root_block_device {
    volume_size = 100
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/user-data/kernellake-host-init.sh", {
    kernellake_docker_image = var.kernellake_docker_image
    s3_bucket                = local.s3_bucket_name
    aws_region                = var.aws_region
    # Points kernellake-server's own ObservabilitySection.otlp_endpoint at
    # the OTel Collector sidecar this same user-data script also starts --
    # see monitoring/otel-collector-config.yaml and the plan's own
    # "OTel -> Prometheus" section. localhost since both containers run on
    # the same instance.
    otlp_endpoint = "localhost:4317"
    # Points kernellake-server's IcebergCatalogSection at the standalone
    # REST catalog instance (iceberg_catalog.tf) -- registered under the
    # name "bench", so queries address tables as
    # read_iceberg('bench.tpch.<table>'). See generate_and_upload_iceberg_data.py
    # for what actually writes those tables.
    iceberg_catalog_uri = "http://${aws_instance.iceberg_catalog.private_ip}:8181"
    iceberg_warehouse    = "s3://${local.s3_bucket_name}/warehouse/"
  })

  tags = {
    Name = "${var.name_prefix}-kernellake-${count.index}"
    Role = "kernellake-host"
  }
}

output "kernellake_instance_public_ips" {
  value = aws_instance.kernellake[*].public_ip
}

output "kernellake_instance_ids" {
  value = aws_instance.kernellake[*].id
}
