# Local KernelLake stack

A self-contained `docker compose` stack for running KernelLake end to end
on one machine, with real observability wired up: `kernellake-server` +
OpenTelemetry Collector + Prometheus + Grafana + Jaeger + MinIO
(S3-compatible object storage standing in for real S3).

Distinct from `../aws/`: that directory provisions real EC2 infrastructure
via Terraform for large-scale/real-S3 benchmarking. This is the
zero-cloud-account, `docker compose up` version -- useful for verifying a
local change end to end (the actual `kernellake-server` binary, real
OTLP export, real Prometheus scraping, real Grafana dashboards) before
reaching for the AWS harness, or when a real EC2 GPU instance isn't worth
the cost/time for what you're checking.

## Prerequisites

- Docker with Compose v2 (`docker compose version`).
- For the default GPU-backed stack: NVIDIA Container Toolkit (`docker run
  --rm --gpus all <any cuda image> nvidia-smi` should work). For a
  GPU-less machine, see "CPU-only" below.
- Python 3 with `pyarrow`, `adbc-driver-flightsql`, `duckdb`, and
  `requests` installed, for `scripts/run_e2e.py`.

## Bring it up

```bash
cd benchmarks/local
docker compose up -d --build
```

`--build` matters: `kernellake-server`'s image is built from this repo's
own `docker/Dockerfile` (`target: runtime-gpu` by default), not pulled
from a registry -- so it always reflects your current working tree, not
whatever was last published to GHCR. First build compiles the whole
project inside the container (RAPIDS/libcudf fetch as prebuilt wheels, not
source-compiled -- see `cmake/ThirdPartyRapids.cmake` -- so this is a few
minutes, not hours); subsequent builds reuse Docker's layer cache and are
much faster unless `src/`/`include/`/`CMakeLists.txt` changed.

Check everything's healthy:

```bash
docker compose ps
```

## Generate and load data

```bash
./scripts/generate_and_upload_data.sh 1   # SF1 (scale factor); omit for the same default
```

Generates TPC-H-derived `lineitem`/`part`/`orders`/`customer` Parquet data
via the project's own `tools/generate_tpch.py` and uploads it to MinIO
(bucket `kernellake-bench`, created automatically by the `minio-init`
one-shot service) over its S3 API on `localhost:9000`. Re-run after
`docker compose down -v` (which drops the MinIO volume).

## Run the end-to-end test + benchmark

```bash
./scripts/run_e2e.py --scale-factor 1
```

For each of the six TPC-H-derived queries KernelLake currently supports
(`benchmarks/tpch/queries/`): runs it against `kernellake-server` over real
Flight SQL, cross-checks the result against DuckDB (reusing
`tools/duckdb_compare.py`/`tools/validate_tpch.py`'s own comparison logic,
pointed at the same MinIO data via DuckDB's `httpfs` extension), and
reports per-query pass/fail + wall time. Afterward, confirms the whole
observability pipeline actually worked: queries Prometheus for
`kernellake_query_duration_seconds_count` and (GPU stack only)
`kernellake_gpu_memory_peak_bytes`/`kernellake_gpu_memory_allocations_total`,
and queries Jaeger for at least one real trace from the `kernellake`
service -- fails the run if any of these is missing.

Useful flags: `--query 6` (just one query), `--skip-metrics-check` (if
Prometheus isn't part of your stack), `--skip-traces-check` (if Jaeger
isn't), `--no-gpu-metrics` (CPU-only stack).

## Look at it

- Grafana: <http://localhost:3000> (anonymous viewer access, no login) --
  "KernelLake" folder has the query-metrics
  (`grafana/dashboards/kernellake-query-metrics.json`), GPU-memory
  (`grafana/dashboards/kernellake-gpu-memory.json`), and NVMe-cache
  (`grafana/dashboards/kernellake-storage-cache.json`) dashboards. The
  cache dashboard only shows real data once `storage.cache.enabled` (on by
  default in this stack's own `config/kernellake-server.yaml`) and at
  least one `s3://...` query has run.
- Prometheus: <http://localhost:9090> -- try `kernellake_query_duration_seconds_count`,
  `kernellake_gpu_memory_peak_bytes`, or `kernellake_storage_cache_hits_total`
  in the query box directly (see `docs/OBSERVABILITY.md` §2.2.1 for the
  full OTel-name -> Prometheus-name table).
- Jaeger: <http://localhost:16686> -- pick service `kernellake` to see real
  query spans (`kernellake.query`), with the same attributes described in
  `docs/OBSERVABILITY.md` §2.1 (`kernellake.sql`, `kernellake.backend`,
  `kernellake.peak_gpu_memory_bytes`, etc.).
- MinIO console: <http://localhost:9001> (`minioadmin`/`minioadmin`).
- `kernellake-server` itself: `localhost:31337`, Flight SQL (see
  `scripts/run_e2e.py` for a real client example, or any Arrow Flight SQL
  client/driver).

## CPU-only

No GPU or NVIDIA Container Toolkit needed:

```bash
KERNELLAKE_TARGET=runtime-cpu docker compose up -d --build
# and drop/comment the kernellake-server `deploy.resources` block in
# docker-compose.yml first -- it unconditionally requests a GPU reservation
# regardless of KERNELLAKE_TARGET, which will fail to schedule without one.
./scripts/run_e2e.py --scale-factor 1 --no-gpu-metrics
```

## Tear down

```bash
docker compose down       # keeps the MinIO/Prometheus/Grafana/NVMe-cache volumes
docker compose down -v    # also drops them -- re-run generate_and_upload_data.sh next time
```

## What's deliberately not here

- **No allocation-overhead benchmark** or **standalone live-collector OTLP
  integration test** as a separate CI-style artifact -- `run_e2e.py`'s own
  metrics/traces checks *are* a real, live-collector integration test,
  just not packaged as a standalone pytest/CI job. See
  `docs/GPU_OPTIMIZATIONS.md`/`docs/OBSERVABILITY.md` for what's tracked
  as follow-up work.
- **Jaeger uses in-memory span storage** (`SPAN_STORAGE_TYPE=memory`, the
  default) -- traces are lost on `docker compose down`/container restart.
  Fine for local verification; swap in a real backend (Cassandra/
  Elasticsearch/Tempo) if traces need to survive a restart.
- **Logs have no backend either** -- `otel-collector-config.yaml`'s logs
  pipeline exports only to its own debug log (`docker compose logs
  otel-collector`), same as `benchmarks/aws/monitoring/otel-collector-
  config.yaml`. Add a Loki (or similar) service + exporter if that's
  wanted.
