# KernelLake AWS benchmark harness -- infrastructure as code.
#
# Nothing here runs on `terraform init`/`validate`/`plan` alone -- those are
# the M0 (zero-cost) verification steps documented in ../../docs/RUNBOOK.md.
# Real AWS resources (and real cost) only start with `terraform apply`,
# which ../../scripts/provision.sh gates behind an explicit --yes and a
# printed cost estimate (../../scripts/estimate_cost.py).

terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "kernellake-aws-benchmark"
      ManagedBy   = "terraform"
      # Every instance this project creates is meant to be short-lived --
      # this tag is what scripts/teardown.sh's own safety check greps for
      # before destroying anything, so a resource created by hand outside
      # Terraform (and thus untagged) is never accidentally swept up.
      Ephemeral   = "true"
    }
  }
}

data "aws_caller_identity" "current" {}
