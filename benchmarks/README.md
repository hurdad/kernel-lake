# Benchmarks

Three independent things live here, plus an (currently empty)
`CMakeLists.txt` for future in-process C++ microbenchmarks
(`KERNELLAKE_BUILD_BENCHMARKS`, added incrementally as their backing
operators become real -- see `docs/ROADMAP.md`).

## `tpch/`

The unofficial, TPC-H-*derived* query files (`queries/q01.sql` etc.) --
`{data}`/`{part_data}`/`{orders_data}`/`{customer_data}` placeholders
substituted by whichever caller runs them (the CLI's own `kernellake
benchmark tpch`, `tools/validate_tpch.py`, or `local/scripts/run_e2e.py`).
Not runnable on its own -- every other benchmark path here reads from it.

## `local/`

A self-contained `docker compose` stack (`kernellake-server` + OTel
Collector + Prometheus + Grafana + Jaeger + MinIO) for running KernelLake
end to end on one machine with real observability wired up -- no cloud
account, no Terraform. `docker compose up -d --build`, generate/upload
TPC-H data, then `scripts/run_e2e.py` runs the query suite against real
Flight SQL, cross-checks results against DuckDB, and confirms the metrics
(Prometheus) and traces (Jaeger) pipelines actually worked. See
`local/README.md`.

Reach for this first -- it verifies a real change (build, execution,
OTel export) against real infrastructure in a few minutes, with zero
cloud spend.

## `aws/`

Terraform-provisioned real EC2 infrastructure (GPU instance(s), Spark
cluster, Iceberg catalog, monitoring instance) for large-scale
benchmarking against real S3 data, real network conditions, and (via
`runner/scaling_test.py`) real multi-instance concurrency -- none of which
`local/`'s single-machine stack can represent. Costs real money per run
(see `aws/docs/COST_ESTIMATES.md`); see `aws/docs/RUNBOOK.md` for the
milestone-by-milestone command sequence, and **always confirm before
provisioning/tearing down** -- this is real, billed cloud infrastructure,
not a local dev loop.

Reach for this only once `local/` has already confirmed the change works,
and the question at hand genuinely needs EC2 scale, real S3, or real
multi-instance behavior to answer.
