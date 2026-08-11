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

variable "kernellake_subnet_id" {
  description = "Overrides just the KernelLake instance's subnet (not local.subnet_id, which every other resource shares) -- for routing around a real, confirmed AZ-specific GPU InsufficientInstanceCapacity error without forcing every other resource (e.g. the Iceberg catalog instance, whose subnet_id is ForceNew) to move AZs too. If null, falls back to local.subnet_id like everything else."
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
  description = "EC2 instance type for the KernelLake GPU benchmark host(s). g6.xlarge (1x NVIDIA L4 24GB, 4 vCPUs, 16GB RAM, local NVMe instance storage -- confirmed via a real `aws ec2 describe-instance-types` lookup) is the current default because it's the only size that fits this account's real G/VT vCPU quota (8, as of 2026-08-09 -- see docs/COST_ESTIMATES.md and the pending appeal on the quota increase case). g6.4xlarge (16 vCPU, same L4) and g6e.16xlarge (64 vCPU, L40S) are the next two steps of a planned instance-size sweep once quota allows -- override this variable to run them, don't change the default until then."
  type        = string
  default     = "g6.xlarge"
}

variable "kernellake_nvme_cache_enabled" {
  description = "Whether kernellake-host-init.sh wires the detected local NVMe instance store into kernellake-server's storage.cache (CacheSection, include/kernellake/common/config.hpp) at all. true (the default) formats/mounts it and enables the cache exactly as before; false skips that whole step and always renders storage.cache.enabled: false into kernellake-server.yaml, regardless of whether the instance actually has local NVMe -- e.g. to get a benchmark run's numbers with every read genuinely hitting S3, not masked by a warm repeat-scan cache."
  type        = bool
  default     = true
}

variable "kernellake_instance_count" {
  description = "Number of independent kernellake-server instances to run. 1 for the base latency/cost-per-query benchmark; 1/2/4/8 across separate applies for the M4 concurrency test (see docs/RUNBOOK.md) -- each instance is fully independent (no distributed execution), so this is horizontal replica count, not a cluster size. 0 stands the rest of the stack up (Spark/DuckDB/monitoring/Iceberg catalog) without the GPU host at all -- e.g. for testing/tuning the Spark or DuckDB leg alone, so the (comparatively expensive) GPU instance isn't sitting there idle and billing while it is."
  type        = number
  default     = 1

  validation {
    condition     = var.kernellake_instance_count >= 0 && var.kernellake_instance_count <= 8
    error_message = "kernellake_instance_count must be between 0 and 8 (the concurrency test's own documented range; 0 means no GPU host at all)."
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

# --- Spark host -----------------------------------------------------------

variable "enable_spark" {
  description = "Whether to provision the dedicated PySpark host at all. false skips it entirely -- e.g. for testing/tuning the KernelLake or DuckDB leg alone, so the Spark instance isn't sitting there idle and billing while it is. See kernellake_instance_count's own comment for the equivalent GPU-host toggle."
  type        = bool
  default     = true
}

variable "spark_instance_type" {
  description = "EC2 instance type for the dedicated single-node PySpark host (local[*] mode, runner/pyspark_query_loop.py -- not a distributed standalone cluster, see spark_cluster.tf's own comment). Defaults to m7i.4xlarge, cost-matched to the KernelLake GPU host's real on-demand rate (confirmed live via the AWS Pricing API: g6.xlarge $0.8048/hr vs. m7i.4xlarge $0.8064/hr, 0.2% apart, as of the pricing lookup that picked this default) -- unlike duckdb_instance_type's own historical default, this one *is* meant as a same-hourly-cost comparison point. Network performance is not an exact match (g6.xlarge is rated \"Up to 10 Gigabit\", m7i.4xlarge \"Up to 12.5 Gigabit\" -- confirmed via `aws ec2 describe-instance-types` that no CPU instance in this price class is rated as low as 10 Gigabit); scripts/measure_s3_throughput.sh run against every instance (see docs/RUNBOOK.md) measures real achieved throughput instead of relying on that rating."
  type        = string
  default     = "m7i.4xlarge"
}

# --- DuckDB host --------------------------------------------------------

variable "enable_duckdb" {
  description = "Whether to provision the dedicated DuckDB host at all. false skips it entirely -- e.g. for testing/tuning the KernelLake or Spark leg alone, so the DuckDB instance isn't sitting there idle and billing while it is. See kernellake_instance_count's own comment for the equivalent GPU-host toggle."
  type        = bool
  default     = true
}

variable "duckdb_instance_type" {
  description = "EC2 instance type for the dedicated DuckDB benchmark host (single-node, CPU-only). Defaults to the same type as the Spark host, m7i.4xlarge -- cost-matched to the KernelLake GPU host's real on-demand rate, the same live Pricing API lookup spark_instance_type's own comment describes (this variable used to default to m7i.xlarge specifically as *not* an attempt at that match; changed once a real cost-matched type was confirmed live)."
  type        = string
  default     = "m7i.4xlarge"
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
