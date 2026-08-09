resource "random_id" "bucket_suffix" {
  count       = var.s3_bucket_name == null ? 1 : 0
  byte_length = 4
}

locals {
  s3_bucket_name = var.s3_bucket_name != null ? var.s3_bucket_name : "${var.name_prefix}-${data.aws_caller_identity.current.account_id}-${random_id.bucket_suffix[0].hex}"
}

resource "aws_s3_bucket" "benchmark_data" {
  bucket = local.s3_bucket_name

  # Kept false by default -- a `terraform destroy` must not silently delete
  # potentially-expensive-to-regenerate TB-scale benchmark data. See
  # ../scripts/teardown.sh: the bucket is only ever emptied via an explicit
  # `--purge-data` flag, which sets this to true in a targeted follow-up
  # apply before destroying, not as this resource's own default.
  force_destroy = false

  tags = {
    Name = "${var.name_prefix}-data"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "benchmark_data" {
  bucket = aws_s3_bucket.benchmark_data.id

  rule {
    id     = "abort-incomplete-multipart-uploads"
    status = "Enabled"
    filter {} # applies to every object in the bucket, not just a prefix
    # A multi-GB/TB upload (SF1000+) that fails partway through leaves
    # incomplete multipart parts that still cost storage $ until cleaned up
    # -- 7 days is generous for retrying a failed upload before that
    # happens automatically.
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

resource "aws_s3_bucket_public_access_block" "benchmark_data" {
  bucket = aws_s3_bucket.benchmark_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
