# Runbook

Exact command sequence for every milestone. See `../README.md` for the
headline metrics this exists to produce and what it deliberately doesn't
measure, and `COST_ESTIMATES.md` for real observed costs once each
milestone has actually run.

**Prerequisites** (once, before M0):
- AWS credentials configured (`aws sts get-caller-identity` should succeed).
- Terraform >= 1.5, Python 3.11+ with `boto3`, an existing EC2 key pair
  name, and your own IP/CIDR for SSH access.
- **G/VT-family EC2 vCPU service quota.** This single quota (code
  `L-DB2E81BA`) covers every `g6.*`/`g6e.*` GPU instance type, not just
  the one you're launching -- check the real current value before
  assuming any given instance type will launch:
  ```bash
  aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA \
    --query 'Quota.Value' --output text
  aws service-quotas list-requested-service-quota-change-history \
    --service-code ec2 --query "RequestedQuotas[?QuotaCode=='L-DB2E81BA']"
  ```
  `kernellake_instance_type` defaults to `g6.xlarge` (4 vCPUs) specifically
  because it's the only size that reliably fits a low/default quota.
  Larger sizes need more: `g6.4xlarge` needs 16, `g6.8xlarge` needs 32,
  `g6e.16xlarge` needs 64. New/unused AWS accounts often default to a
  quota of 0-8 for this family (confirmed for real: a `g6.8xlarge` attempt
  once failed with `VcpuLimitExceeded` *after* the Spark cluster had
  already started billing, since Terraform provisions in parallel -- and a
  32-vCPU increase request was later "partially approved" with the new
  limit equal to the old one, i.e. a real denial, requiring a manual
  appeal with a detailed use case reopening the Support Center case
  rather than a second API request -- `request-service-quota-increase`
  rejects a second request outright while one is still open). Request
  ahead of time and expect it to take a manual round-trip, not an instant
  approval:
  ```bash
  aws service-quotas request-service-quota-increase --service-code ec2 \
    --quota-code L-DB2E81BA --desired-value <needed-for-your-instance-type>
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
all engines, metrics collection, reporting) at manageable cost before
scaling up.

**Every run's output lives in its own dated directory** --
`runs/<timestamp>-sf<N>/` (gitignored, see `.gitignore` -- real per-run
numbers aren't source, `docs/COST_ESTIMATES.md` is where the durable
summary goes once a run is done). Set this up first:

```bash
cd ../scripts  # benchmarks/aws/scripts
RUN_DIR="../runs/$(date +%Y%m%d-%H%M%S)-sf100"
mkdir -p "$RUN_DIR"
```

**Selective bring-up**: if you're only testing/tuning one engine, skip
provisioning the others so they aren't idle and billing while you do --
`-var="enable_spark=false"` and/or `-var="enable_duckdb=false"` skip those
entirely, `-var="kernellake_instance_count=0"` skips the GPU host. The
full three-way comparison below needs all of them.

**Single-node Spark**: `spark_worker_count=0` (with `enable_spark=true`)
gives a self-sufficient single-node standalone cluster -- the master also
runs its own worker (see spark_cluster.tf). This still runs a real
standalone cluster (Master + Worker daemons), which has real footguns
confirmed live on this project: co-located daemons need distinct JMX
ports, and `SPARK_LOCAL_DIRS` (a Worker-process env var) silently
overrides `spark.local.dir` (the driver-side config), which bit a real
run when `/tmp` turned out to be a small RAM-backed tmpfs. For a
single-node run, `--spark-master-host local` (`aws_benchmark_runner.py`)
sidesteps all of that by running Spark in-process (`local[*]`) instead --
prefer it over standing up the standalone daemons when
`spark_worker_count=0` is what you're doing anyway.

```bash
python3 estimate_cost.py --milestone m1
# Review the estimate, then:

cat > ../terraform/terraform.tfvars <<EOF
ssh_key_name             = "<your-key-pair-name>"
allowed_ssh_cidr          = "<your-ip>/32"
kernellake_docker_image   = "<ECR-URI>/kernellake:runtime-gpu"
kernellake_instance_count = 1
# kernellake_instance_type defaults to g6.xlarge -- see Prerequisites'
# G/VT quota note before overriding to a larger size.
spark_worker_count        = 3
EOF

./provision.sh --yes
# Wait a few minutes for instances to finish boot (NVIDIA driver install,
# if the Deep Learning Base AMI wasn't available for Ubuntu 26.04 yet --
# see terraform/ami.tf's own comment -- adds a reboot cycle here).

./generate_and_upload_data.sh --scale-factor 100 --wait
```

**NVMe cache**: `kernellake-host-init.sh` detects, formats, and mounts the
GPU instance's local NVMe instance storage automatically at boot and
enables `storage.cache` against it (falls back to disabled with a clear
log line if the instance type has none -- check
`/var/log/kernellake-host-init.log` on the host to confirm which path it
took). No action needed here, but it's real infra state worth confirming
once per instance type, not assuming: SSH in and check
`df -h /mnt/kernellake-nvme-cache` and `docker compose logs kernellake-server`
for cache hit/miss counters, or watch `kernellake.storage.cache.*` in
Grafana. Expect this to show up as improved *warm*-mode numbers
specifically (repeat scans of the same data), not cold-mode ones.

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
  --query all --iterations 2 --output "$RUN_DIR/results.json"
```

**DuckDB** now runs on its own dedicated, cost-tracked host
(`terraform/duckdb_instance.tf`) rather than in-process -- scp the query
loop + query files over, run it via SSH, and copy the results back into
this run's directory:

```bash
DUCKDB_HOST=$(cd ../terraform && terraform output -raw duckdb_host_public_ip)
scp duckdb_query_loop.py ../../tpch/queries "ubuntu@${DUCKDB_HOST}:~/" -r
ssh "ubuntu@${DUCKDB_HOST}" \
  "python3 duckdb_query_loop.py --s3-bucket $BUCKET --scale-factor 100 \
   --query all --iterations 2 --output duckdb-results.json"
scp "ubuntu@${DUCKDB_HOST}:~/duckdb-results.json" "$RUN_DIR/duckdb-results.json"
```

(The older `--duckdb` flag on `aws_benchmark_runner.py` still works as a
quick in-process check with no dedicated infra, but its numbers won't be
cost-tracked -- see README.md.)

**Raw S3 throughput** (optional, but useful context for interpreting scan
throughput numbers below): run directly on the KernelLake host itself, so
it measures the same instance/network path the real queries used.

```bash
scp ../scripts/measure_s3_throughput.sh "ubuntu@${KL_HOST}:~/"
ssh "ubuntu@${KL_HOST}" \
  "./measure_s3_throughput.sh --bucket $BUCKET --scale-factor 100 --output s3-throughput.json"
scp "ubuntu@${KL_HOST}:~/s3-throughput.json" "$RUN_DIR/s3-throughput.json"
```

Compute real cost (after the run, before teardown -- so instance uptime
reflects the actual run window):

```bash
KL_ID=$(cd ../terraform && terraform output -json kernellake_instance_ids | jq -r '.[0]')
DUCKDB_ID=$(cd ../terraform && terraform output -raw duckdb_host_id)
# ... gather spark/monitoring instance IDs similarly from terraform output ...
python3 cost_model.py \
  --kernellake-instance-ids "$KL_ID" \
  --spark-instance-ids "<master-id>,<worker-id-1>,<worker-id-2>,<worker-id-3>" \
  --spark-instance-types "m7i.xlarge,m7i.4xlarge,m7i.4xlarge,m7i.4xlarge" \
  --duckdb-instance-id "$DUCKDB_ID" \
  --monitoring-instance-id "<monitoring-id>" \
  --output "$RUN_DIR/cost.json"
```

Generate the report (Markdown/CSV, plus a PDF with the same tables and the
written analysis/caveats, for sharing outside a repo checkout):

```bash
cd ../reporting
python3 aggregate_results.py --benchmark-results "$RUN_DIR/results.json" --output "$RUN_DIR/aggregated.json"
python3 generate_report.py --input "$RUN_DIR/aggregated.json" --cost-json "$RUN_DIR/cost.json" \
  --duckdb-results "$RUN_DIR/duckdb-results.json" --output-dir "$RUN_DIR/report/"
python3 generate_pdf_report.py --input "$RUN_DIR/aggregated.json" --cost-json "$RUN_DIR/cost.json" \
  --duckdb-results "$RUN_DIR/duckdb-results.json" --output "$RUN_DIR/report.pdf"
python3 generate_dashboard.py --input "$RUN_DIR/aggregated.json" --cost-json "$RUN_DIR/cost.json" \
  --output "$RUN_DIR/dashboard.html"
```

Review `$RUN_DIR/report/report.md`, `$RUN_DIR/report.pdf`, and
`$RUN_DIR/dashboard.html`. Confirm the headline numbers (latency speedup
ratio, cost per completed query) look sane before proceeding -- this is
the actual validation gate for M1, not just "did the commands exit zero."
Once a run is worth keeping, record its real headline numbers in
`docs/COST_ESTIMATES.md`'s prose/table (the `$RUN_DIR` contents themselves
are gitignored, ephemeral).

```bash
cd ../scripts
./teardown.sh  # keeps the S3 data; pass --purge-data to also remove it
```

**Instance-size sweep**: once G/VT quota allows (see Prerequisites),
`g6.xlarge` -> `g6.4xlarge` -> `g6e.16xlarge` is the planned next step --
repeat this M1 sequence with `-var="kernellake_instance_type=g6.4xlarge"`
(etc.) in place of the default, each into its own `$RUN_DIR`. `g6e.16xlarge`
carries an L40S GPU rather than the L4 the other two have, so treat that
leg as a different-GPU-generation data point, not a same-GPU size
comparison.

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

The most expensive single step (up to 8x simultaneous GPU-instance-hours,
at whichever `kernellake_instance_type` is configured) -- time-boxed, torn
down immediately after.

```bash
python3 ../scripts/estimate_cost.py --milestone m4
# Review, then re-provision with more replicas:
cd ../terraform
terraform apply -var="kernellake_instance_count=8" -var-file=terraform.tfvars

cd ../runner
RUN_DIR="../runs/$(date +%Y%m%d-%H%M%S)-m4-concurrency"
mkdir -p "$RUN_DIR"
KL_HOSTS=$(cd ../terraform && terraform output -json kernellake_instance_public_ips | jq -r 'join(",")')
python3 scaling_test.py --kernellake-hosts "$KL_HOSTS" --s3-bucket "$BUCKET" \
  --scale-factor 1000 --concurrent-clients 16 --duration-seconds 120 \
  --output "$RUN_DIR/scaling-8replicas.json"
```

Repeat for 1/2/4 replicas (re-`apply` with a lower
`kernellake_instance_count` each time, same `$RUN_DIR`), then aggregate
all four `scaling-*replicas.json` files via `reporting/aggregate_results.py`'s
`--scaling-results` flag before the final report/dashboard generation.

**Teardown immediately after this test** -- 8 GPU instances running is the
most expensive state this project can be in.

## M5 -- final polish

Aggregate every scale factor + the full scaling-test sweep into one final
`report.md`/`dashboard.html`, finalize this file and `COST_ESTIMATES.md`
with real observed numbers from the milestones above.
