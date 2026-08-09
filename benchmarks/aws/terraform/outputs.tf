# Per-resource outputs (instance IPs, Grafana URL) live alongside their
# resources (kernellake_instance.tf, spark_cluster.tf,
# monitoring_instance.tf) -- these are the remaining general ones the
# runner/scripts also need.

output "s3_bucket_name" {
  value = aws_s3_bucket.benchmark_data.bucket
}

output "s3_bucket_arn" {
  value = aws_s3_bucket.benchmark_data.arn
}

output "security_group_id" {
  value = aws_security_group.benchmark.id
}

output "aws_region" {
  value = var.aws_region
}

output "subnet_id" {
  value = local.subnet_id
}

output "iam_instance_profile_name" {
  value = aws_iam_instance_profile.benchmark_host.name
}

output "ssh_key_name" {
  value = var.ssh_key_name
}
