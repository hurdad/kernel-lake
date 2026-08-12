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
  (the `docker-publish` job in `.github/workflows/ci.yml`) already builds
  and pushes on every push to `main` (confirmed via a real successful
  run). Only build and push your own image (and override
  `kernellake_docker_image`) if you're testing local changes not yet on
  `main`:
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

**Every engine runs on its own single, dedicated, cost-tracked host** --
KernelLake (`terraform/kernellake_instance.tf`), Spark
(`terraform/spark_cluster.tf`), and DuckDB (`terraform/duckdb_instance.tf`)
are each one instance, no distributed cluster for any of them. Spark runs
in local[*] mode (`runner/pyspark_query_loop.py`), not a real
`spark://<master>:7077` standalone cluster -- that sidesteps a whole class
of standalone-mode-specific footguns confirmed live on this project's
earlier multi-node topology: co-located Master/Worker daemons needing
distinct JMX ports, and `SPARK_LOCAL_DIRS` (a Worker-process env var)
silently overriding `spark.local.dir` (the driver-side config), which bit
a real run when `/tmp` turned out to be a small RAM-backed tmpfs. See
`runner/pyspark_query_loop.py`'s own module docstring. Spark and DuckDB
are both cost-matched to the KernelLake GPU host's real on-demand rate
(`m7i.4xlarge`, confirmed live via the AWS Pricing API against
`g6.xlarge` -- see `spark_instance_type`/`duckdb_instance_type` in
`terraform/variables.tf`); their network performance rating is not an
exact match (no CPU instance in this price class is rated as low as
`g6.xlarge`'s "Up to 10 Gigabit" -- see the Raw S3 throughput step below,
which measures what each host actually achieves instead of relying on
that rating).

**Overriding `kernellake_instance_type` needs a matching
`spark_instance_type`/`duckdb_instance_type` override too** -- the
`m7i.4xlarge` default above is priced against `g6.xlarge` specifically,
not whatever GPU size you actually launch. Confirmed live for `g6.2xlarge`
($0.9776/hr): the closest CPU match across every family checked
(`m7a`/`r7i`/`r7a`/`m6a`/`m6i`/`c7i`/`c6i`/`i4i`/`z1d`, several sizes each)
is `r6i.4xlarge` ($1.008/hr, 3.1% apart) -- notably not another `m7i` size,
since AWS instance sizes jump in big steps (4xlarge to 8xlarge roughly
doubles) with nothing in between. `r6i.4xlarge` also has double the RAM
(128GB vs. `m7i.4xlarge`'s 64GB), real headroom for Spark's `local[*]` JVM
after a live `SparkOutOfMemoryError` on the 64GB host -- see
`pyspark_query_loop.py`'s `new_spark_session()` comment for that fix.
Skipping this re-match doesn't error, it just silently compares KernelLake
against CPU hosts priced for a different, cheaper GPU tier -- confirmed
for real: the same SF100 queries measured 45.8x median speedup against
`g6.xlarge`-priced hosts and 9.9x against genuinely `g6.2xlarge`-priced
ones. Both numbers are real; only one matches the GPU size actually
running. Re-check live pricing for your own instance_type rather than
reusing either of these -- on-demand rates drift.

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
EOF

./provision.sh --yes
# Wait a few minutes for instances to finish boot (NVIDIA driver install,
# if the Deep Learning Base AMI wasn't available for Ubuntu 26.04 yet --
# see terraform/ami.tf's own comment -- adds a reboot cycle here).

./generate_and_upload_data.sh --scale-factor 100 --wait
```

**Compression comparison** (optional): `generate_and_upload_data.sh --compression` writes
each choice to its own fully-tagged `tpch-data/sf<N>-<tag>/` prefix, so multiple
compressions of the same scale factor can coexist and be pointed at independently via
every reader's own `--compression`/`--compression-level` flags
(`aws_benchmark_runner.py`, `duckdb_query_loop.py`, `pyspark_query_loop.py`,
`measure_s3_throughput.sh`). Generation is single-threaded pure Python (confirmed via a
real `ps aux` on a running instance -- one core pegged, ~125K rows/sec regardless of
instance size), so a small instance type generates just as fast as a large one and lets
multiple variants run concurrently instead of competing for the account's 64-vCPU
on-demand quota:

```bash
export REGION="$(cd ../terraform && terraform output -raw aws_region)"
export SSH_KEY="$(cd ../terraform && terraform output -raw ssh_key_name)"
./generate_and_upload_data.sh --scale-factor 100 --compression none                       # tpch-data/sf100-none/
./generate_and_upload_data.sh --scale-factor 100 --compression snappy                     # tpch-data/sf100-snappy/
./generate_and_upload_data.sh --scale-factor 100 --compression zstd                       # tpch-data/sf100-zstd/ (PyArrow's own default level, confirmed == 1)
./generate_and_upload_data.sh --scale-factor 100 --compression zstd --compression-level 3 # tpch-data/sf100-zstd-l3/ (libzstd's own upstream default)
```

(The `REGION`/`SSH_KEY` env var overrides above are only needed if `terraform output` is
missing them -- e.g. after a partial `terraform apply -target` that re-created just the
networking/IAM resources this script needs without standing up the full GPU/Spark/
DuckDB/monitoring stack. Omit them for a normal full `provision.sh` apply.)

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
`kernellake_nvme_cache_enabled=false` disables it entirely (every read
genuinely hits S3) -- useful when comparing against Spark/DuckDB, neither
of which has an equivalent cache.

**Cache-off combined with a smaller instance type can look like a full
regression, not just a warm-mode difference** -- confirmed for real:
`kernellake_nvme_cache_enabled=false` on `g6.xlarge` (rather than
`g6.2xlarge`) returned KernelLake medians 10x *slower* than the cache-on
`g6.2xlarge` baseline, to the point cold and warm mode were nearly
identical and KernelLake measured slower than PySpark/DuckDB outright.
Root-caused via a direct S3 throughput probe (`measure_s3_throughput.sh`):
`g6.xlarge` plateaus at ~265 MB/s, vs. ~830 MB/s on the CPU hosts it was
being compared against -- `g6.xlarge`'s smaller network allocation, not a
code regression. Re-running the identical queries with cache back on and
`g6.2xlarge` restored numbers within a few percent of the original
baseline. If a run looks anomalously slow, check both
`kernellake_nvme_cache_enabled` and `kernellake_instance_type` before
assuming something broke -- this combination reproduces the symptom
exactly.

Run the KernelLake leg (from a machine with network access to the
provisioned instances -- SSH to the monitoring instance, or set up SSH
port-forwarding for port 31337 from your local machine):

```bash
cd ../runner
KL_HOST=$(cd ../terraform && terraform output -json kernellake_instance_public_ips | jq -r '.[0]')
BUCKET=$(cd ../terraform && terraform output -raw s3_bucket_name)

python3 aws_benchmark_runner.py \
  --kernellake-host "$KL_HOST" \
  --s3-bucket "$BUCKET" --scale-factor 100 \
  --query all --iterations 2 --output "$RUN_DIR/results.json"
```

**Spark** runs on its own dedicated host, the same shape as DuckDB below --
scp the query loop + query files over, run it via SSH, and copy the
results back into this run's directory:

```bash
SPARK_HOST=$(cd ../terraform && terraform output -raw spark_host_public_ip)
scp pyspark_query_loop.py ../../tpch/queries "ubuntu@${SPARK_HOST}:~/" -r
ssh "ubuntu@${SPARK_HOST}" \
  "python3 pyspark_query_loop.py --s3-bucket $BUCKET --scale-factor 100 \
   --query all --iterations 2 --output pyspark-results.json"
scp "ubuntu@${SPARK_HOST}:~/pyspark-results.json" "$RUN_DIR/pyspark-results.json"
```

**DuckDB** runs on its own dedicated host the same way:

```bash
DUCKDB_HOST=$(cd ../terraform && terraform output -raw duckdb_host_public_ip)
scp duckdb_query_loop.py ../../tpch/queries "ubuntu@${DUCKDB_HOST}:~/" -r
ssh "ubuntu@${DUCKDB_HOST}" \
  "python3 duckdb_query_loop.py --s3-bucket $BUCKET --scale-factor 100 \
   --query all --iterations 2 --output duckdb-results.json"
scp "ubuntu@${DUCKDB_HOST}:~/duckdb-results.json" "$RUN_DIR/duckdb-results.json"
```

(`--spark-master-host local` and `--duckdb` on `aws_benchmark_runner.py`
still work as quick in-process checks with no dedicated infra, but their
numbers won't be cost-tracked -- see README.md.)

**Raw S3 throughput** (optional, but useful context for interpreting scan
throughput numbers below): run on *every* provisioned benchmark instance
(each KernelLake replica, the Spark host, the DuckDB host), not just
KernelLake -- each measures its own real S3 GET throughput over its own
actual network path, which is what the network-rating mismatch noted
above (`g6.xlarge`'s "Up to 10 Gigabit" vs. `m7i.4xlarge`'s "Up to 12.5
Gigabit") needs an empirical answer for, not just a spec comparison.

```bash
../scripts/run_s3_throughput_all_instances.sh --scale-factor 100 --output-dir "$RUN_DIR"
```

Combine every instance's result into one chart:

```bash
python3 ../reporting/plot_s3_throughput.py --input-dir "$RUN_DIR" --output "$RUN_DIR/s3-throughput.png"
```

Compute real cost (after the run, before teardown -- so instance uptime
reflects the actual run window):

```bash
KL_ID=$(cd ../terraform && terraform output -json kernellake_instance_ids | jq -r '.[0]')
DUCKDB_ID=$(cd ../terraform && terraform output -raw duckdb_host_id)
SPARK_ID=$(cd ../terraform && terraform output -raw spark_host_id)
MONITORING_ID=$(cd ../terraform && terraform output -raw monitoring_instance_id)
python3 cost_model.py \
  --kernellake-instance-ids "$KL_ID" \
  --spark-instance-ids "$SPARK_ID" \
  --spark-instance-types "m7i.4xlarge" \
  --duckdb-instance-id "$DUCKDB_ID" \
  --duckdb-instance-type "m7i.4xlarge" \
  --monitoring-instance-id "$MONITORING_ID" \
  --output "$RUN_DIR/cost.json"
```

Generate the report (Markdown/CSV, plus a PDF with the same tables and the
written analysis/caveats, for sharing outside a repo checkout):

```bash
cd ../reporting
python3 aggregate_results.py --benchmark-results "$RUN_DIR/results.json" --output "$RUN_DIR/aggregated.json"
python3 generate_report.py --input "$RUN_DIR/aggregated.json" --cost-json "$RUN_DIR/cost.json" \
  --duckdb-results "$RUN_DIR/duckdb-results.json" --pyspark-results "$RUN_DIR/pyspark-results.json" \
  --output-dir "$RUN_DIR/report/"
python3 generate_pdf_report.py --input "$RUN_DIR/aggregated.json" --cost-json "$RUN_DIR/cost.json" \
  --duckdb-results "$RUN_DIR/duckdb-results.json" --pyspark-results "$RUN_DIR/pyspark-results.json" \
  --output "$RUN_DIR/report.pdf"
python3 generate_dashboard.py --input "$RUN_DIR/aggregated.json" --cost-json "$RUN_DIR/cost.json" \
  --output "$RUN_DIR/dashboard.html"
```

`--pyspark-results` is not optional now that Spark runs on its own dedicated
host rather than in-process (see `merge_pyspark_results()` in
`reporting/generate_report.py`) -- omitting it silently produces a report
with no PySpark column at all rather than an error, a real gap only found
by actually running this sequence end to end. `generate_dashboard.py` has
no equivalent `--duckdb-results`/`--pyspark-results` flags yet (a
pre-existing gap, not new today) -- it reads only `aggregated.json`
directly, so the dashboard currently shows KernelLake-only numbers even
when the Markdown/PDF reports show all three engines.

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

**If teardown hangs on `terraform destroy` for an instance/IGW/VPC** that's
already gone in `aws ec2 describe-instances` reality: this is a real,
reproducible pattern, not necessarily a stuck `terraform destroy` process
-- confirmed for real (twice, same session) that the underlying resource
was already fully deleted while Terraform's own state/polling loop kept
reporting "Still destroying..." for many more minutes. Re-issuing the
delete directly against the still-reported resource (e.g. `aws ec2
terminate-instances --instance-ids <id>`) is safe and idempotent even if
it's already gone, and Terraform's own destroy eventually catches up. If a
VPC specifically won't finish destroying, check for a subnet Terraform
doesn't know about (see below) before assuming it's just slow.

**AZ-specific `InsufficientInstanceCapacity`**: real, encountered live
for both `g6.2xlarge` and `g6.4xlarge` in `us-east-1a` (this project's
default subnet AZ, pinned in `terraform/networking.tf` for an unrelated
instance-type/AZ-support reason -- see that file's own comment) --
confirmed via `aws cloudtrail lookup-events
--lookup-attributes AttributeKey=EventName,AttributeValue=RunInstances`
that AWS was rejecting the launch outright, not Terraform hanging. AWS's
own error message suggests other AZs, but **that suggestion is generic
boilerplate, not a real-time capacity signal** -- confirmed for real: `us-east-1a`'s
error recommended `us-east-1b`, and `us-east-1b`'s own error (from an
actual `aws ec2 run-instances` attempt there, not just reading the
message) recommended `us-east-1a` right back. The only reliable check is
an actual launch attempt in the AZ you're considering.

`kernellake_subnet_id` (in `terraform/variables.tf`) overrides just the
KernelLake instance's subnet without touching the shared `local.subnet_id`
every other resource uses -- deliberately narrow, since a shared
`subnet_id` override would force-replace the Iceberg catalog instance too
(its `subnet_id` is `ForceNew`, ordinarily no reason to move AZs). To use
a different AZ for one run:

```bash
# No vpc_id output exists -- read it out of state directly, same way as
# ROUTE_TABLE_ID below.
VPC_ID=$(cd ../terraform && terraform state show 'aws_vpc.benchmark[0]' | grep -m1 '^    id ' | awk '{print $3}' | tr -d '"')
SUBNET_ID=$(aws ec2 create-subnet --region us-east-1 --vpc-id "$VPC_ID" \
  --cidr-block 10.90.3.0/24 --availability-zone us-east-1c \
  --query 'Subnet.SubnetId' --output text)
aws ec2 modify-subnet-attribute --region us-east-1 --subnet-id "$SUBNET_ID" --map-public-ip-on-launch
ROUTE_TABLE_ID=$(cd ../terraform && terraform state show 'aws_route_table.benchmark[0]' | grep -m1 '^    id ' | awk '{print $3}' | tr -d '"')
aws ec2 associate-route-table --region us-east-1 --route-table-id "$ROUTE_TABLE_ID" --subnet-id "$SUBNET_ID"
# Then add to terraform.tfvars: kernellake_subnet_id = "<the new subnet id>"
```

**This subnet is created outside Terraform, so Terraform never destroys
it either** -- a real, confirmed VPC-deletion hang at teardown time: the
manually-created subnet blocked `aws_vpc.benchmark`'s own destroy for
10+ minutes with no error, just silent "Still destroying...". Delete it
manually (`aws ec2 delete-subnet --subnet-id <id>`) before/during
teardown if you used this workaround -- check `aws ec2 describe-subnets
--filters "Name=vpc-id,Values=<vpc-id>"` for anything Terraform's own
`terraform state list` doesn't know about.

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
