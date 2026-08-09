variable "aws_region" {
  description = "AWS region for all resources. S3 bucket and every EC2 instance share this region to avoid cross-region transfer cost."
  type        = string
  default     = "us-east-1"
}

variable "name_prefix" {
  description = "Prefix for all resource names/tags, so multiple concurrent benchmark runs (or runs by different people) don't collide."
  type        = string
  default     = "kernellake-bench"
}

variable "ssh_key_name" {
  description = "Name of an existing EC2 key pair, for SSH access to instances (data generation, log inspection, debugging). No default -- must be supplied explicitly."
  type        = string
}

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed to SSH into any instance (e.g. \"203.0.113.4/32\" for a single IP). No default -- must be supplied explicitly, never left open to 0.0.0.0/0 by accident."
  type        = string

  validation {
    condition     = var.allowed_ssh_cidr != "0.0.0.0/0"
    error_message = "allowed_ssh_cidr must not be 0.0.0.0/0 -- scope it to your own IP or a bastion's CIDR."
  }
}

variable "subnet_id" {
  description = "Subnet to launch instances into. If null, the default VPC's first available subnet is used (see networking.tf) -- avoids the cost/complexity of a dedicated VPC+NAT gateway for short-lived benchmark infrastructure."
  type        = string
  default     = null
}

variable "vpc_id" {
  description = "VPC to create the security group in. If null, the account's default VPC is used (see networking.tf)."
  type        = string
  default     = null
}

# --- KernelLake host(s) -----------------------------------------------

variable "kernellake_instance_type" {
  description = "EC2 instance type for the KernelLake GPU benchmark host(s). g6.8xlarge per the benchmark spec: 1x NVIDIA L4 (24GB), 32 vCPUs, 128GB RAM."
  type        = string
  default     = "g6.8xlarge"
}

variable "kernellake_instance_count" {
  description = "Number of independent kernellake-server instances to run. 1 for the base latency/cost-per-query benchmark; 1/2/4/8 across separate applies for the M4 concurrency test (see docs/RUNBOOK.md) -- each instance is fully independent (no distributed execution), so this is horizontal replica count, not a cluster size."
  type        = number
  default     = 1

  validation {
    condition     = var.kernellake_instance_count >= 1 && var.kernellake_instance_count <= 8
    error_message = "kernellake_instance_count must be between 1 and 8 (the concurrency test's own documented range)."
  }
}

variable "kernellake_docker_image" {
  description = "Docker image reference for the kernellake-server runtime-gpu image. Defaults to the image this repo's own CI (.github/workflows/docker-publish.yml) already builds and pushes to GHCR on every push to main (confirmed via a real successful run this session) -- no manual build/push needed unless testing local changes, in which case build docker/Dockerfile's runtime-gpu target yourself and override this. Assumes the GHCR package is public (the default for a public source repo, which hurdad/kernel-lake is) -- if a pull fails with an auth error on the KernelLake host, either make the package public in the repo's Packages settings, or add a `docker login ghcr.io` step (with a PAT) to kernellake-host-init.sh."
  type        = string
  default     = "ghcr.io/hurdad/kernel-lake-gpu:latest"
}

variable "kernellake_ami_id" {
  description = "AMI for the KernelLake GPU host(s). Defaults to AWS's own Deep Learning AMI (Ubuntu 24.04-based, NVIDIA driver + CUDA 12.x + Docker + nvidia-container-toolkit pre-baked) looked up by name in ami.tf -- set explicitly to override (e.g. to test the from-scratch stock-Ubuntu install path)."
  type        = string
  default     = null
}

# --- Spark cluster -------------------------------------------------------

variable "spark_master_instance_type" {
  description = "EC2 instance type for the Spark standalone master."
  type        = string
  default     = "m7i.xlarge"
}

variable "spark_worker_instance_type" {
  description = "EC2 instance type for each Spark standalone worker."
  type        = string
  default     = "m7i.4xlarge"
}

variable "spark_worker_count" {
  description = "Number of Spark worker instances. Sized (instance type x count) against live AWS Pricing API lookups in scripts/estimate_cost.py to approximate the KernelLake host's own hourly cost -- see docs/COST_ESTIMATES.md for the actual comparison once computed, this default is a starting point, not a verified match."
  type        = number
  default     = 3
}

# --- Iceberg REST catalog ---------------------------------------------

variable "iceberg_catalog_instance_type" {
  description = "EC2 instance type for the Iceberg REST catalog (tabulario/iceberg-rest). Small and cheap -- it only tracks table metadata pointers, not query execution."
  type        = string
  default     = "t3.small"
}

# --- Monitoring ------------------------------------------------------------

variable "monitoring_instance_type" {
  description = "EC2 instance type for the Prometheus/Grafana monitoring instance. Small and cheap -- it aggregates metrics, it doesn't run queries."
  type        = string
  default     = "t3.large"
}

# --- S3 ----------------------------------------------------------------

variable "s3_bucket_name" {
  description = "Name for the benchmark data S3 bucket. If null, a name is generated from name_prefix + account ID (globally unique bucket names required)."
  type        = string
  default     = null
}
