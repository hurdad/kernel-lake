# Self-managed Spark standalone cluster (not EMR) on m7i instances, for
# direct control over cost-matching against the KernelLake GPU host's own
# hourly cost -- see scripts/estimate_cost.py, which sizes
# spark_worker_count/spark_worker_instance_type against live AWS Pricing
# API lookups rather than a hardcoded guess.

resource "aws_instance" "spark_master" {
  count = var.enable_spark ? 1 : 0

  ami                         = data.aws_ami.ubuntu_26_04.id
  instance_type               = var.spark_master_instance_type
  subnet_id                   = local.subnet_id
  vpc_security_group_ids      = [aws_security_group.benchmark.id]
  iam_instance_profile        = aws_iam_instance_profile.benchmark_host.name
  key_name                    = var.ssh_key_name
  associate_public_ip_address = true

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/user-data/spark-node-init.sh", {
    role       = "master"
    master_ip  = "" # the master doesn't need its own IP to start itself
    s3_bucket   = local.s3_bucket_name
    aws_region   = var.aws_region
  })

  tags = {
    Name = "${var.name_prefix}-spark-master"
    Role = "spark-master"
  }
}

resource "aws_instance" "spark_worker" {
  count = var.enable_spark ? var.spark_worker_count : 0

  ami                         = data.aws_ami.ubuntu_26_04.id
  instance_type               = var.spark_worker_instance_type
  subnet_id                   = local.subnet_id
  vpc_security_group_ids      = [aws_security_group.benchmark.id]
  iam_instance_profile        = aws_iam_instance_profile.benchmark_host.name
  key_name                    = var.ssh_key_name
  associate_public_ip_address = true

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
  }

  # Depends on the master explicitly (rather than relying on apply
  # ordering alone) so the worker's own init script always has a real
  # master private IP to register with by the time it runs, not a
  # not-yet-provisioned one.
  user_data = templatefile("${path.module}/user-data/spark-node-init.sh", {
    role       = "worker"
    master_ip  = aws_instance.spark_master[0].private_ip
    s3_bucket   = local.s3_bucket_name
    aws_region   = var.aws_region
  })

  depends_on = [aws_instance.spark_master]

  tags = {
    Name = "${var.name_prefix}-spark-worker-${count.index}"
    Role = "spark-worker"
  }
}

output "spark_master_public_ip" {
  value = length(aws_instance.spark_master) > 0 ? aws_instance.spark_master[0].public_ip : null
}

output "spark_worker_public_ips" {
  value = aws_instance.spark_worker[*].public_ip
}
