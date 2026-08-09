# Runbook

Exact command sequence for every milestone. See `../README.md` for the
headline metrics this exists to produce and what it deliberately doesn't
measure, and `COST_ESTIMATES.md` for real observed costs once each
milestone has actually run.

**Prerequisites** (once, before M0):
- AWS credentials configured (`aws sts get-caller-identity` should succeed).
- Terraform >= 1.5, Python 3.11+ with `boto3`, an existing EC2 key pair
  name, and your own IP/CIDR for SSH access.
- **G/VT-family EC2 vCPU service quota >= 32** (covers `g6.8xlarge`, which
  needs 32 vCPUs). New/unused AWS accounts often default to a quota of 0
  for this instance family (confirmed for real during this project's own
  M1 attempt: `RunInstances` failed with `VcpuLimitExceeded` after the
  Spark cluster had already been created and started billing). Check and
  request ahead of time, since approval isn't instant:
  ```bash
  aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA \
    --query 'Quota.Value' --output text
  # If less than 32:
  aws service-quotas request-service-quota-increase --service-code ec2 \
    --quota-code L-DB2E81BA --desired-value 32
  ```
- **No manual Docker build/push needed**: `kernellake_docker_image` defaults
  to `ghcr.io/hurdad/kernel-lake-gpu:latest`, which this repo's own CI
  (`.github/workflows/docker-publish.yml`) already builds and pushes on
  every push to `main` (confirmed via a real successful run). Only build
  and push your own image (and override `kernellake_docker_image`) if
  you're testing local changes not yet on `main`:
  ```bash
  cd ../../.. # repo root
  docker build -t <your-registry>/kernellake:runtime-gpu -f docker/Dockerfile --target runtime-gpu .
  docker push <your-registry>/kernellake:runtime-gpu
  ```

## M0 -- zero-cost skeleton

No AWS resources created. Confirms the Terraform is internally consistent
before anything real is spent.

```bash
cd terraform
terraform init
terraform validate
terraform plan \
  -var="ssh_key_name=<your-key-pair-name>" \
  -var="allowed_ssh_cidr=<your-ip>/32"
# Review the plan. Creates nothing. (kernellake_docker_image needs no
# override unless testing a local build -- see Prerequisites above.)
```

## M1 -- small validated run

Proves the whole pipeline works end to end (provisioning, data generation,
both engines, metrics collection, reporting) at manageable cost before
scaling up.

```bash
cd ../scripts
python3 estimate_cost.py --milestone m1
# Review the estimate, then:

cat > ../terraform/terraform.tfvars <<EOF
ssh_key_name             = "<your-key-pair-name>"
allowed_ssh_cidr          = "<your-ip>/32"
kernellake_docker_image   = "<ECR-URI>/kernellake:runtime-gpu"
kernellake_instance_count = 1
spark_worker_count        = 3
EOF

./provision.sh --yes
# Wait a few minutes for instances to finish boot (NVIDIA driver install,
# if the Deep Learning Base AMI wasn't available for Ubuntu 26.04 yet --
# see terraform/ami.tf's own comment -- adds a reboot cycle here).

./generate_and_upload_data.sh --scale-factor 100 --wait
```

Run the benchmark (from a machine with network access to the provisioned
instances -- SSH to the Spark master or monitoring instance, or set up SSH
port-forwarding for ports 31337 and 7077 from your local machine):

```bash
cd ../runner
KL_HOST=$(cd ../terraform && terraform output -json kernellake_instance_public_ips | jq -r '.[0]')
SPARK_HOST=$(cd ../terraform && terraform output -raw spark_master_public_ip)
BUCKET=$(cd ../terraform && terraform output -raw s3_bucket_name)

python3 aws_benchmark_runner.py \
  --kernellake-host "$KL_HOST" --spark-master-host "$SPARK_HOST" \
  --s3-bucket "$BUCKET" --scale-factor 100 \
  --query all --iterations 2 --output ../results-sf100.json
```

Compute real cost (after the run, before teardown -- so instance uptime
reflects the actual run window):

```bash
KL_ID=$(cd ../terraform && terraform output -json kernellake_instance_ids | jq -r '.[0]')
# ... gather spark/monitoring instance IDs similarly from terraform output ...
python3 cost_model.py \
  --kernellake-instance-ids "$KL_ID" \
  --spark-instance-ids "<master-id>,<worker-id-1>,<worker-id-2>,<worker-id-3>" \
  --spark-instance-types "m7i.xlarge,m7i.4xlarge,m7i.4xlarge,m7i.4xlarge" \
  --monitoring-instance-id "<monitoring-id>" \
  --output ../cost-sf100.json
```

Generate the report:

```bash
cd ../reporting
python3 aggregate_results.py --benchmark-results ../results-sf100.json --output ../aggregated.json
python3 generate_report.py --input ../aggregated.json --cost-json ../cost-sf100.json --output-dir ../report/
python3 generate_dashboard.py --input ../aggregated.json --cost-json ../cost-sf100.json --output ../dashboard.html
```

Review `../report/report.md` and `../dashboard.html`. Confirm the headline
numbers (latency speedup ratio, cost per completed query) look sane before
proceeding -- this is the actual validation gate for M1, not just "did the
commands exit zero."

```bash
cd ../scripts
./teardown.sh  # keeps the S3 data; pass --purge-data to also remove it
```

## M2 -- full SF100 report

Same as M1, but with the full 6-query set, treated as the first
publication-candidate report. Repetition count stays at the default
(`--iterations 2`) -- a higher count (5 was tried once, real EC2 GPU-hour
cost made a full `--query all` x 2 modes x cold-mode-restarts-per-rep run
take too long in practice) is only worth it for a specific query/mode
whose 2-sample median looks noisy, not as a blanket default.

## M3 -- larger scale factors

For each of SF1000, SF3000, SF10000, in order:

```bash
python3 ../scripts/estimate_cost.py --milestone m3-sf1000  # or m3-sf3000 / m3-sf10000
# Review, then:
../scripts/generate_and_upload_data.sh --scale-factor 1000 --wait
# (SF3000/SF10000: consider a larger --instance-type for generate_and_upload_data.sh,
# and expect multi-hour runtimes -- see estimate_cost.py's own data_gen_hours table)
```

Then repeat the M1 benchmark-run/cost/report sequence with
`--scale-factor 1000` (etc.) in place of `100`.

## M4 -- concurrency headline test

The most expensive single step (up to 8x simultaneous `g6.8xlarge`
instance-hours) -- time-boxed, torn down immediately after.

```bash
python3 ../scripts/estimate_cost.py --milestone m4
# Review, then re-provision with more replicas:
cd ../terraform
terraform apply -var="kernellake_instance_count=8" -var-file=terraform.tfvars

cd ../runner
KL_HOSTS=$(cd ../terraform && terraform output -json kernellake_instance_public_ips | jq -r 'join(",")')
python3 scaling_test.py --kernellake-hosts "$KL_HOSTS" --s3-bucket "$BUCKET" \
  --scale-factor 1000 --concurrent-clients 16 --duration-seconds 120 \
  --output ../scaling-8replicas.json
```

Repeat for 1/2/4 replicas (re-`apply` with a lower
`kernellake_instance_count` each time), then aggregate all four
`scaling-*replicas.json` files via `reporting/aggregate_results.py`'s
`--scaling-results` flag before the final report/dashboard generation.

**Teardown immediately after this test** -- 8 GPU instances running is the
most expensive state this project can be in.

## M5 -- final polish

Aggregate every scale factor + the full scaling-test sweep into one final
`report.md`/`dashboard.html`, finalize this file and `COST_ESTIMATES.md`
with real observed numbers from the milestones above.
