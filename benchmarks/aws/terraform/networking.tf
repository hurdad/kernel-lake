# Creates a minimal, dedicated VPC (public subnet + Internet Gateway, no
# NAT Gateway) unless var.vpc_id/var.subnet_id are given -- deliberately
# NOT relying on the account's default VPC existing at all: confirmed for
# real, during this project's own M0 verification, that a real AWS account
# can have no default VPC (newer/security-hardened accounts often don't),
# which `data "aws_vpc" "default" { default = true }` hard-fails on with
# "no matching EC2 VPC found" rather than returning an empty/absent result
# Terraform could branch on. No NAT Gateway needed here specifically
# because every instance this project creates already gets a public IP
# directly (`associate_public_ip_address = true` on each aws_instance) --
# a plain Internet Gateway + public route table is enough, and unlike a
# NAT Gateway has no hourly cost of its own.

resource "aws_vpc" "benchmark" {
  count                = var.vpc_id == null ? 1 : 0
  cidr_block           = "10.90.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "${var.name_prefix}-vpc"
  }
}

locals {
  vpc_id = var.vpc_id != null ? var.vpc_id : aws_vpc.benchmark[0].id
}

resource "aws_internet_gateway" "benchmark" {
  count  = var.vpc_id == null ? 1 : 0
  vpc_id = aws_vpc.benchmark[0].id

  tags = {
    Name = "${var.name_prefix}-igw"
  }
}

resource "aws_subnet" "benchmark" {
  count                   = var.subnet_id == null && var.vpc_id == null ? 1 : 0
  vpc_id                  = aws_vpc.benchmark[0].id
  cidr_block              = "10.90.1.0/24"
  map_public_ip_on_launch = true
  # Pinned explicitly -- left to AWS's default AZ selection, this landed in
  # us-east-1e for this account, which doesn't support g6.8xlarge or
  # m7i.xlarge (confirmed via a real failed RunInstances call during this
  # project's own M1 apply). us-east-1a is one of the AZs AWS's own error
  # message listed as supporting both instance types.
  availability_zone = "${var.aws_region}a"

  tags = {
    Name = "${var.name_prefix}-subnet"
  }
}

resource "aws_route_table" "benchmark" {
  count  = var.vpc_id == null ? 1 : 0
  vpc_id = aws_vpc.benchmark[0].id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.benchmark[0].id
  }

  tags = {
    Name = "${var.name_prefix}-rt"
  }
}

resource "aws_route_table_association" "benchmark" {
  count          = var.subnet_id == null && var.vpc_id == null ? 1 : 0
  subnet_id      = aws_subnet.benchmark[0].id
  route_table_id = aws_route_table.benchmark[0].id
}

# Gateway endpoint for S3 -- free (no hourly/per-GB charge, unlike Interface
# endpoints), and keeps all TPC-H scan traffic (the thing this whole harness
# measures) on AWS's internal network instead of hairpinning out through the
# Internet Gateway, which would otherwise sit as an unmeasured bandwidth
# bottleneck between "S3 throughput" and "what the benchmark actually
# reports." Only created alongside our own VPC/route-table (var.vpc_id ==
# null) -- if the caller supplied an existing VPC, its own S3 endpoint
# story (if any) is their concern, not this harness's.
resource "aws_vpc_endpoint" "s3" {
  count             = var.vpc_id == null ? 1 : 0
  vpc_id            = aws_vpc.benchmark[0].id
  service_name      = "com.amazonaws.${var.aws_region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [aws_route_table.benchmark[0].id]

  tags = {
    Name = "${var.name_prefix}-s3-endpoint"
  }
}

# If the caller supplied vpc_id but not subnet_id, fall back to that VPC's
# first available subnet (mirrors the original default-VPC-reuse design,
# just scoped to a caller-supplied VPC instead of assuming a default one).
data "aws_subnets" "caller_vpc" {
  count = var.vpc_id != null && var.subnet_id == null ? 1 : 0
  filter {
    name   = "vpc-id"
    values = [var.vpc_id]
  }
}

locals {
  subnet_id = (
    var.subnet_id != null ? var.subnet_id :
    var.vpc_id == null ? aws_subnet.benchmark[0].id :
    data.aws_subnets.caller_vpc[0].ids[0]
  )
}

resource "aws_security_group" "benchmark" {
  name_prefix = "${var.name_prefix}-"
  description = "KernelLake AWS benchmark harness -- SSH (scoped), and internal traffic between benchmark hosts (Flight SQL, Spark, Prometheus scraping, OTLP)."
  vpc_id      = local.vpc_id

  # SSH: scoped to the caller-supplied CIDR only, never left open.
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  # All other ports are internal-only: allowed from other instances in this
  # same security group (self-referencing), never from the public internet.
  # Covers: kernellake-server Flight SQL (31337), OTel Collector OTLP
  # receiver + Prometheus exporter (4317, 8889), node_exporter (9100),
  # dcgm-exporter (9400), Spark master/worker/UI/history (7077, 8080, 8081,
  # 4040, 18080), Spark JMX exporter (7777), Prometheus (9090), Grafana
  # (3000).
  ingress {
    description = "Internal benchmark-harness traffic"
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    self        = true
  }

  egress {
    description = "All outbound (S3, package installs, Docker image pulls, AWS APIs)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.name_prefix}-sg"
  }
}
