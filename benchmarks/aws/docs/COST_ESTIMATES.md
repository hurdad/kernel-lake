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
| dev validation | SF100 (pre-fix) | n/a (below M1 -- Q3 excluded, see notes) | **$2.46** (kernellake $0.90, spark $1.52) | 2026-08-09, same infra as the SF10 row above. Q3 (3-way join) excluded: real RMM OOM on the g6.2xlarge's single L4 (24GB) -- "Exceeded memory limit (failed to allocate 1.432GiB)" -- a genuine finding, not a bug from this session's other fixes (see `docs/GPU_OPTIMIZATIONS.md` opportunity #4, `pass_read_limit_bytes` retuning). The other 5 queries: all 10 KernelLake/PySpark comparisons matched, but **PySpark won every single one, 2-4x** -- a full reversal from SF10, where KernelLake won 8/12. See "SF100 latency reversal" below for the real breakdown of why. |
| dev validation | SF100 (post-fix) | n/a (same run, after the `resolve_table()` parallelization fix) | **$4.64 cumulative session total** (kernellake $1.47, spark $3.07 -- see notes, not directly comparable to the pre-fix row) | 2026-08-09, same infra, rerun after deploying the `std::async` metadata-inspection fix (`docs/GPU_OPTIMIZATIONS.md`). All 10 comparisons still matched; KernelLake got real-, meaningfully faster per query (Q6 cold 22.7s -> 15.9s) but **PySpark still won every comparison, 2-3x** -- a narrower gap, not a reversal-of-the-reversal. Cost per completed query improved to 2.08x cheaper for KernelLake ($0.147 vs. $0.307, up from 1.7x pre-fix). This row's dollar total is the full session's cumulative instance-hours (`cost_model.py` measures from each instance's real `LaunchTime` to teardown), not an isolated cost for just this rerun -- it includes the SF10 and pre-fix SF100 runs' EC2 time too, since the same instances stayed up throughout. See "SF100 post-fix rerun" below. |
| M1 | SF100 | run `estimate_cost.py --milestone m1` for current pricing | not yet run (SF100 above is a dev validation subset, not the formal M1 methodology -- Q3 missing, only 5/6 queries, 2 not 3+ iterations) | |
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

## SF100 dev validation run (2026-08-09) -- real results, and a real reversal

Same infra as the SF10 run above (g6.2xlarge + 1x m7i.xlarge + 3x
m7i.4xlarge). Q3 (3-way join) excluded -- see the table row above for the
real RMM OOM that caused it.

**Latency: PySpark won every single query, 2-4x** (e.g. Q6 cold:
KernelLake 22.7s vs. PySpark 5.3s), a full reversal from SF10 where
KernelLake won 8/12. Root cause investigated directly (not guessed): ran
`kernellake benchmark tpch` standalone against the same real SF100 Q1 data
for its own decode/compute/wall breakdown --

```
wall_seconds: 28.0s, gpu_execution_seconds: 17.5s, parquet_decoding_seconds: 7.9s,
peak_gpu_memory_bytes: 4.3 GiB (well under the L4's 24 GiB -- not the SF100 Q3 OOM's cause here)
```

Three things stack up against KernelLake at this scale, none of them the
S3-read fix's fault (decode itself, 7.9s for ~11.6GB compressed lineitem,
is reasonable -- roughly the same per-byte cost as the SF10 numbers
above): (1) ~9.6s of non-decode GPU compute for a two-column GROUP BY/SUM
over ~600M rows -- whether that's proportionate or itself has headroom is
unmeasured; (2) ~10.5s unaccounted for outside gpu_execution_seconds
entirely -- plausibly file discovery/footer-reading across 120 S3 objects
(10x SF10's 12), not measured directly here; (3) structural: this account's
GPU quota only allows g6.2xlarge (8 vCPUs/32GB host RAM), not the
originally-specified g6.8xlarge (32 vCPUs/128GB) -- same single L4 GPU
either way, but far less host-side parallelism/RAM for whatever isn't
actually running on the GPU, against a Spark cluster with 48 vCPUs/192GB
spread across 4 real instances. **Cost per completed query still favors
KernelLake** (1.7x cheaper, $0.090 vs. $0.152) purely because it's one
instance's cost against four -- but the latency picture at SF100 is
unambiguous and not currently a story worth telling as a KernelLake win.
Not chased further in this session (would need dedicated GPU-compute and
file-discovery profiling, real further EC2 cost/time); flagged here as a
real, reproducible finding rather than smoothed over.

## SF100 post-fix rerun (2026-08-09) -- narrower gap, not closed

Same infra, same 5 queries (Q3 still excluded, same real OOM), rerun after
deploying `src/io/table_resolution.cpp`'s `std::async` metadata-inspection
fix (see `docs/GPU_OPTIMIZATIONS.md`) to the live benchmark server. All 10
KernelLake/PySpark comparisons still matched.

**Latency: PySpark still won every comparison, but by less** (2.5-3.9x,
down from 2-4x): e.g. Q6 cold went from KernelLake 22.7s -> 15.9s (PySpark
unchanged at ~5.2s, so the ratio narrowed from ~4.3x to ~3.0x); Q1 cold
went from KernelLake's earlier 28.0s standalone `metadata_inspection`
measurement down to 18.3s in the full Flight-SQL-round-trip comparison.
The metadata-inspection fix is real and measured (14.5x on its own metric,
`docs/GPU_OPTIMIZATIONS.md`), but wasn't the dominant cost at SF100 --
per the earlier breakdown, GPU compute (~9.6s) and the still-unmeasured
host-side/S3 discovery overhead together outweigh the ~5.1s this fix
recovered. Confirms this is a real partial win, not a full fix, and the
structural gap (single-instance g6.2xlarge host-side parallelism vs. a
4-instance/48-vCPU Spark cluster) documented in the pre-fix section above
still stands as the bigger remaining factor.

**Cost per completed query: KernelLake pulled further ahead**, 2.08x
cheaper ($0.147 vs. $0.307, up from 1.7x pre-fix) -- faster KernelLake
queries directly lower its own per-query cost share, while PySpark's
didn't change.

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
