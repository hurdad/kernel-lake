# Cost estimates vs. real observed costs

`../scripts/estimate_cost.py`'s output is an *estimate* (live on-demand
pricing, but approximated data-generation duration and Parquet size --
see that script's own comments for exactly which numbers are guessed vs.
looked up live). This file is where the **real** number replaces the
estimate, once a milestone has actually run and `runner/cost_model.py`
has computed real cost from actual instance-hours.

Until a milestone has actually run, its row says "not yet run" -- never a
fabricated number standing in for a real one.

| Milestone | Scale factor | Estimated cost | Real observed cost | Notes |
|---|---|---|---|---|
| dev validation | SF10 | n/a (below any formal milestone) | **$1.25** (kernellake $0.57, spark $0.66) | 2026-08-09. Real run: 6 queries x cold/warm x 2 iterations, KernelLake vs. real PySpark, 1x g6.2xlarge + 1x m7i.xlarge + 3x m7i.4xlarge + monitoring + iceberg catalog. All 12 KernelLake/PySpark result comparisons matched. See `benchmarks/aws/report-sf10/report.md` (gitignored -- regenerate via `reporting/generate_report.py` against `results-sf10.json`/`cost-sf10.json`, also gitignored, or see this file's own summary below). |
| M1 | SF100 | run `estimate_cost.py --milestone m1` for current pricing | not yet run | |
| M2 | SF100 | run `estimate_cost.py --milestone m2` for current pricing | not yet run | |
| M3 | SF1000 | run `estimate_cost.py --milestone m3-sf1000` | not yet run | |
| M3 | SF3000 | run `estimate_cost.py --milestone m3-sf3000` | not yet run | |
| M3 | SF10000 | run `estimate_cost.py --milestone m3-sf10000` | not yet run | Largest single cost in the project by data volume |
| M4 | SF1000, 8 replicas | run `estimate_cost.py --milestone m4` | not yet run | Largest single cost by simultaneous instance-hours |

## SF10 dev validation run (2026-08-09) -- real results

Not a formal milestone (SF10 is below M1's SF100), but a real, complete
run worth recording: confirmed the `device_read`/`device_read_async` S3
scan-throughput fix (see `docs/GPU_OPTIMIZATIONS.md`) end to end against
the real PySpark comparison, and validated the new
`cost_efficiency_per_query` KPI (`runner/cost_model.py`'s
`QueryCostEfficiency`) against a real run for the first time.

**Latency** (median seconds, 2 iterations): KernelLake beat PySpark on
8 of 12 query/mode combinations (1.2x-1.6x, mostly cold mode); PySpark won
the other 4 (Q1/Q6/Q19 warm, 0.68x-0.95x) -- at SF10 the data volume is
small enough that per-query fixed overhead (Flight SQL round-trip, Arrow
serialization) matters more than raw scan speed, and a warmed-up JVM is
competitive at this scale. Not a clean GPU-wins story, reported as
measured rather than the more flattering subset.

**Cost per completed query**: KernelLake $0.048 vs. PySpark $0.055 (1.15x
cheaper) -- KernelLake wins here even where it lost on latency, since it's
one instance's cost against four.

**Cost efficiency (new KPI)**: the opposite result -- PySpark is more
TB-scanned-per-dollar in most cells (e.g. Q6 warm: PySpark 11.0 TB/$ vs.
KernelLake 4.2 TB/$). The GPU instance's real on-demand rate ($0.98/hr)
vs. the Spark cluster's real blended rate ($0.52/hr) is large enough to
outweigh KernelLake's speed advantage for pure cost-efficiency at this
scale, even on the queries it won on latency. Whether this flips at larger
scale factors (more data amortizing the GPU's fixed per-query overhead) is
exactly what the SF100 run below tests.

## Data-generation duration (for `estimate_cost.py`'s own
`data_gen_hours` table, and to correct it once real numbers exist)

| Scale factor | Estimated duration | Real observed duration |
|---|---|---|
| SF100 | ~1h (guess) | not yet run |
| SF1000 | ~3h (guess) | not yet run |
| SF3000 | ~8h (guess) | not yet run |
| SF10000 | ~20h (guess) | not yet run |

## AMI availability

`terraform/ami.tf` tries AWS's Deep Learning Base AMI for Ubuntu 26.04
first, falling back to a plain Ubuntu 26.04 image (in which case
`kernellake-host-init.sh`'s from-scratch NVIDIA driver install path runs,
adding a reboot cycle to first boot -- see that script's own comments).

- **Deep Learning Base AMI found for Ubuntu 26.04?** Yes -- confirmed via
  a real `terraform plan` in the target account/region (`us-east-1`):
  `ami-0887f7d803698e7c3` ("Deep Learning Base OSS Nvidia Driver GPU AMI
  (Ubuntu 26.04)*"). The from-scratch driver-install fallback path is not
  expected to be exercised by default, but stays available/working via
  `kernellake_ami_id` regardless.
- **Plain Ubuntu 26.04 AMI (for Spark/monitoring instances)?** Yes --
  `ami-02ebdb11bae1b2486` (Canonical-owned), also confirmed via the same
  `terraform plan`.
- **`terraform plan` end-to-end (no default VPC in this account)?**
  Confirmed clean: `networking.tf` creates a self-sufficient VPC/subnet/
  Internet Gateway (no NAT Gateway, no dependency on a default VPC) --
  21 resources planned, 0 errors, 0 warnings.
