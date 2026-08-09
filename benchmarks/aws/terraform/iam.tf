# One shared instance profile for every instance this project creates
# (KernelLake hosts, Spark nodes, the data-generation instance, the
# monitoring instance) -- all need S3 access to the benchmark bucket
# (read for query hosts, read+write for the generator), and CloudWatch
# agent permissions for host-level metrics. Scoped to just this bucket,
# not full S3 access.

data "aws_iam_policy_document" "assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "benchmark_host" {
  name_prefix        = "${var.name_prefix}-"
  assume_role_policy = data.aws_iam_policy_document.assume_role.json
}

data "aws_iam_policy_document" "s3_access" {
  statement {
    sid       = "ListBucket"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.benchmark_data.arn]
  }
  statement {
    sid     = "ReadWriteObjects"
    actions = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload"]
    resources = ["${aws_s3_bucket.benchmark_data.arn}/*"]
  }
}

resource "aws_iam_role_policy" "s3_access" {
  name_prefix = "${var.name_prefix}-s3-"
  role        = aws_iam_role.benchmark_host.id
  policy      = data.aws_iam_policy_document.s3_access.json
}

resource "aws_iam_role_policy_attachment" "cloudwatch_agent" {
  role       = aws_iam_role.benchmark_host.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

# Read-only EC2 describe permissions -- scripts/estimate_cost.py and
# cost_model.py look up this run's own instances' launch times for
# instance-hour accounting (see runner/cost_model.py's own comment) rather
# than requiring the caller to track wall-clock start times by hand.
#
# ec2:TerminateInstances, scoped to only this project's own Ephemeral=true
# instances (never a blanket grant -- a role that can terminate ANY
# instance in the account is a real blast-radius risk) -- for
# generate_and_upload_data.sh/generate_and_upload_iceberg_data.py's
# self-terminating ephemeral data-gen instances. This was missing
# entirely until a real self-terminate call failed with a genuine
# UnauthorizedOperation once an unrelated IMDSv2 bug (which had been
# producing a malformed-instance-id error first) was fixed -- the missing
# permission was masked by that earlier bug and only surfaced once the
# call actually reached AWS's own authorization check.
data "aws_iam_policy_document" "ec2_describe" {
  statement {
    actions   = ["ec2:DescribeInstances", "pricing:GetProducts"]
    resources = ["*"]
  }
  statement {
    actions   = ["ec2:TerminateInstances"]
    resources = ["arn:aws:ec2:*:*:instance/*"]
    condition {
      test     = "StringEquals"
      variable = "aws:ResourceTag/Ephemeral"
      values   = ["true"]
    }
  }
}

resource "aws_iam_role_policy" "ec2_describe" {
  name_prefix = "${var.name_prefix}-ec2-describe-"
  role        = aws_iam_role.benchmark_host.id
  policy      = data.aws_iam_policy_document.ec2_describe.json
}

resource "aws_iam_instance_profile" "benchmark_host" {
  name_prefix = "${var.name_prefix}-"
  role        = aws_iam_role.benchmark_host.name
}
