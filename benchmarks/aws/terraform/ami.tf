# Ubuntu 26.04, matching kernel-lake's own Docker image baseline
# (docker/Dockerfile's runtime-gpu/runtime-cpu targets both build FROM
# ubuntu:26.04, not 24.04 -- see docs/ARCHITECTURE.md's "Ubuntu 26.04
# baseline" section) -- newer system Abseil that Arrow's bundled
# dependencies need, per that doc.
#
# AWS's Deep Learning Base AMI (NVIDIA driver + CUDA 12.x + Docker +
# nvidia-container-toolkit pre-baked) is tried first, since a release as
# new as Ubuntu 26.04 may not have one published yet -- this is a real,
# unverified-from-here risk, not a settled fact. If this data source finds
# no match, `terraform plan`/`apply` fails loudly with "no matching AMI
# found" (Terraform's own data-source behavior, not silently falling back
# to something else) -- at that point, set `kernellake_ami_id` explicitly
# to a plain Ubuntu 26.04 AMI (e.g. `data.aws_ami.ubuntu_26_04.id` below).
# Either way, kernellake-host-init.sh's own NVIDIA-driver-and-toolkit
# install step is a real, working fallback (not just a warning) for
# exactly this case -- see that script's own comment.
data "aws_ami" "deep_learning_base" {
  count       = var.kernellake_ami_id == null ? 1 : 0
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 26.04)*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

locals {
  kernellake_ami_id = var.kernellake_ami_id != null ? var.kernellake_ami_id : data.aws_ami.deep_learning_base[0].id
}

# Plain Ubuntu 26.04 -- for the Spark and monitoring instances (no GPU
# needed there), and as the explicit `kernellake_ami_id` fallback target
# noted above if no Deep Learning Base AMI exists yet for 26.04.
data "aws_ami" "ubuntu_26_04" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    # Ubuntu's AMI name codename for 26.04 wasn't confirmed at the time
    # this was written (too new to have verified against a real account) --
    # "noble" was 24.04's; 26.04 will have its own. Verify the actual name
    # via `aws ec2 describe-images --owners 099720109477 --filters
    # "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-*-26.04-amd64-server-*"`
    # during M0's own verification pass and correct this filter if it
    # doesn't match, rather than assume.
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-*-26.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}
