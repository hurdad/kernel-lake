# KernelLake

**GPU-native analytics for the open lakehouse.**

KernelLake is an open-source GPU-native query engine for Apache Iceberg,
Delta Lake, and plain Parquet data lakes. It executes analytical SQL
directly against Parquet datasets, using Apache Arrow-compatible columnar
data as its in-memory representation and NVIDIA GPUs (via RAPIDS libcudf)
for accelerated execution. KernelLake is a compute and query layer, not a
storage database.

## Status

KernelLake is an early work in progress, but the spec's required initial
deliverable -- generating sample data and running the MVP query
end-to-end through real GPU execution -- works today and has been verified
on real GPU hardware. See [docs/ROADMAP.md](docs/ROADMAP.md) for exactly
what's done vs. not started, and why.

Concretely, this is real, reproducible output:

```
$ kernellake generate-data --output /tmp/kernellake-demo --rows 1000 --files 2 \
    --row-group-rows 200 --seed 42
wrote 1000 rows across 2 file(s) to /tmp/kernellake-demo

$ kernellake query --sql "SELECT region, SUM(amount) AS total \
    FROM read_parquet('/tmp/kernellake-demo/*.parquet') \
    WHERE event_date >= DATE '2026-01-01' GROUP BY region" --stats

query stats:
  rows_returned: 10
  files_considered: 2
  files_scanned: 2
  row_groups_considered: 6
  row_groups_scanned: 6
  peak_gpu_memory_bytes: 240073275
  elapsed_wall_seconds: 0.464069
region    total
--------  ------------------
region-8  19889.78518393226
region-6  19411.167328676154
...                            # (10 region rows total; truncated here)
```

That ran the full pipeline against real GPU hardware: SQL parsing,
binding/type-checking, the rule-based optimizer, file discovery, Parquet
metadata inspection and row-group pruning, GPU filtering, GPU grouped
aggregation (per-batch `cudf::groupby::groupby` folded into a running
partial result), and Arrow result conversion. `peak_gpu_memory_bytes` and `elapsed_wall_seconds` are measured,
not estimated -- KernelLake never fabricates a metric it can't measure (see
`QueryResult` in `include/kernellake/api/query_engine.hpp` for which fields
still report as unmeasured `std::nullopt`, and why).

Pruning is real too -- read each file's min/max statistics and skip a file
or row group only when a predicate is *proven* impossible to satisfy
(against the same generated dataset as above; `order_id` is assigned
sequentially, so it clusters cleanly by file and row group):

```
$ kernellake explain --sql "SELECT order_id FROM read_parquet('/tmp/kernellake-demo/*.parquet') \
    WHERE order_id < 50"

ArrowResult
    └── Filter
        predicate: (order_id < 50)
        └── ParquetScan
            files: 1/2
            row_groups: 1/3
            columns: [order_id]
```

(No `Projection` node: the optimizer's redundant-projection-removal rule
elides it here, since the scan is already narrowed to exactly the
`SELECT` list's own column -- see docs/ARCHITECTURE.md's optimizer
section. A `Projection` node reappears for a query that actually computes
or reorders columns, e.g. a `CASE` expression or `SELECT b, a`.)

`files: 1/2` and `row_groups: 1/3` mean KernelLake proved the second file
and two of the first file's three row groups couldn't contain any
`order_id < 50` rows, and skipped reading them entirely.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full pipeline, the
GPU operator set, the CPU execution backend (Apache Arrow Acero), and the
CPU/GPU build split. `kernellake query` runs on the GPU by default when
built with CUDA (`--backend gpu`, the default) and throws a clear
`ExecutionError` for that backend on a CPU-only build -- it never silently
substitutes a CPU implementation without being asked. Pass `--backend cpu`
(or set `engine.backend: cpu`) to run on the always-available Acero CPU
backend instead, in *either* build.

## Requirements

Tested on Ubuntu 24.04 and 26.04, x86_64 -- CI builds every preset inside a
plain `ubuntu:26.04` container (see `.github/workflows/ci.yml`), and this
project's own non-container dev sandbox has itself moved to 26.04 too (see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)'s "Ubuntu 26.04 baseline"
section for why -- Arrow Flight SQL and otel-cpp both need it).

```bash
# Core toolchain -- libxml2-dev/uuid-dev are for Arrow's bundled GCS/Azure
# filesystem support, libavro-dev/libcurl4-openssl-dev for Iceberg manifest
# reading and REST/OAuth2 calls, libgrpc++-dev/protobuf-compiler-grpc for
# the Delta Lake gRPC client -- all unconditional even for the base `dev`
# preset, not specific to any one optional feature (see CMakeLists.txt's
# own comments on each `find_package`/`pkg_check_modules` call).
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git pkg-config \
  python3 python3-pip \
  libgtest-dev libbenchmark-dev libspdlog-dev nlohmann-json3-dev libyaml-cpp-dev \
  libxml2-dev uuid-dev libavro-dev libcurl4-openssl-dev \
  libgrpc++-dev protobuf-compiler-grpc

# Apache Arrow / Parquet C++ (official Apache Arrow apt repo)
sudo apt-get install -y -V ca-certificates lsb-release wget
wget -O /tmp/arrow-apt-source.deb \
  "https://packages.apache.org/artifactory/arrow/ubuntu/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb"
sudo apt-get install -y -V /tmp/arrow-apt-source.deb
sudo apt-get update
sudo apt-get install -y libarrow-dev libparquet-dev libarrow-dataset-dev
```

GPU execution additionally needs the CUDA Toolkit (12+) and RAPIDS
libcudf/RMM. Only the CUDA Toolkit needs manual installation -- libcudf/RMM
are vendored automatically via CMake `FetchContent` from pinned RAPIDS PyPI
wheels the first time you configure the `gpu-dev` preset (no conda/mamba;
see "GPU dependency vendoring" in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)).
CMake >= 3.30.4 is required for that preset specifically -- only a real
constraint on Ubuntu 24.04, whose own apt `cmake` package is older than
that (26.04's, 4.2.3 at last check, already isn't).

Two more presets need their own extra packages, on top of the core
toolchain above: `server-dev` (the Arrow Flight SQL server) additionally
needs `libarrow-flight-dev`/`libarrow-flight-sql-dev` (same Apache Arrow
apt repo); `otel-dev` (OpenTelemetry export) needs `opentelemetry-cpp-dev`,
which -- like `libarrow-flight-sql-dev` -- has no Ubuntu 24.04 apt package
at all (26.04 only). The two can be combined with each other by passing
the extra flag directly rather than switching presets --
`cmake --preset otel-dev -DKERNELLAKE_BUILD_SERVER=ON` -- since there's no
dedicated combined preset yet (see `otel-dev`'s own `description` in
`CMakePresets.json`).

## Build and test

Five CMake presets (`CMakePresets.json`), each independently
build+test-able; `dev` is the one every other preset builds on
(`server-dev`/`otel-dev` both `inherits: dev`, `gpu-dev` shares its base
config):

```bash
# dev: CPU-only debug build -- SQL parsing through physical planning,
# pruning, generate-data. No CUDA needed; everything else below needs this
# one's own Requirements at minimum.
cmake --preset dev
cmake --build --preset dev
ctest --preset dev

# release: same CPU-only scope as dev, RelWithDebInfo instead of Debug.
cmake --preset release
cmake --build --preset release
ctest --preset release

# gpu-dev: adds real GPU query execution (needs CUDA Toolkit 12+, an
# NVIDIA GPU, and CMake >= 3.30.4 -- see Requirements above).
cmake --preset gpu-dev
cmake --build --preset gpu-dev
ctest --preset gpu-dev

# server-dev: CPU-only + kernellake-server (Arrow Flight SQL). Needs
# libarrow-flight-dev/libarrow-flight-sql-dev -- see Requirements above.
cmake --preset server-dev
cmake --build --preset server-dev
ctest --preset server-dev

# otel-dev: CPU-only + OpenTelemetry OTLP/gRPC export. Needs
# opentelemetry-cpp-dev -- see Requirements above.
cmake --preset otel-dev
cmake --build --preset otel-dev
ctest --preset otel-dev
```

## Usage

```bash
# Generate a deterministic synthetic dataset (works with either preset)
./build/dev/src/cli/kernellake generate-data --output /tmp/kernellake-sales \
  --rows 10000000 --files 16 --row-group-rows 250000 --seed 42

# Inspect a Parquet file's schema, row groups, and column statistics
./build/dev/src/cli/kernellake inspect-parquet --path /data/sales.parquet
./build/dev/src/cli/kernellake inspect-parquet --path /data/sales.parquet --format json

# See the plan KernelLake would run for a query
./build/dev/src/cli/kernellake explain \
  --sql "SELECT region, SUM(amount) FROM read_parquet('/data/sales/*.parquet') GROUP BY region"
./build/dev/src/cli/kernellake explain --logical --sql "..."   # pre-physical-planning view
./build/dev/src/cli/kernellake explain --format json --sql "..."

# Run a query on the GPU (requires a gpu-dev build; the dev build's binary
# throws a clear ExecutionError for the (default) gpu backend instead)
./build/gpu-dev/src/cli/kernellake query \
  --sql "SELECT region, SUM(amount) FROM read_parquet('/tmp/kernellake-sales/*.parquet') \
         WHERE event_date >= DATE '2026-01-01' GROUP BY region"
./build/gpu-dev/src/cli/kernellake query --file query.sql --format csv --output result.csv
./build/gpu-dev/src/cli/kernellake query --sql "..." --format jsonl
./build/gpu-dev/src/cli/kernellake query --sql "..." --format arrow --output result.arrow
./build/gpu-dev/src/cli/kernellake query --sql "..." --stats   # prints measured metrics to stderr

# Run the same query on the CPU (Apache Arrow Acero) instead -- works with
# either build, including the CPU-only dev build. Supports LIKE/CASE/JOIN
# (including N-way chains) too; see docs/ARCHITECTURE.md for this
# backend's current query-shape scope.
./build/dev/src/cli/kernellake query --backend cpu \
  --sql "SELECT region, SUM(amount) FROM read_parquet('/tmp/kernellake-sales/*.parquet') GROUP BY region"
```

`read_parquet(...)` reads plain Parquet files/globs; `read_iceberg('catalog.namespace.table')`
reads a real Apache Iceberg table via a REST catalog (manifest reading,
partition-spec-aware pruning, row-level deletes); `read_delta('table_uri')`
reads a real Delta Lake table. All three compose with joins, filters, and
aggregates the same way. Supported SQL grammar and everything
intentionally not yet supported are documented in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Correctness validation against DuckDB

`tools/validate_against_duckdb.py` runs a set of representative queries
through both `kernellake query --format arrow` and DuckDB reading the same
Parquet files, then compares the results (row order and floating-point
precision-insensitive). It requires a `gpu-dev` build plus the `duckdb` and
`pyarrow` Python packages:

```bash
pip install --user duckdb pyarrow   # or a virtualenv, if your system Python is externally managed
python3 tools/validate_against_duckdb.py \
  --kernellake build/gpu-dev/src/cli/kernellake --data '/tmp/kernellake-sales/*.parquet'
```

This has been run against a 500,000-row / 4-file generated dataset,
covering filtered projection, scalar aggregates (SUM/COUNT/AVG/MIN/MAX),
grouped aggregates (including a ~40,000-group `GROUP BY customer_id`), and
`COUNT` over a nullable column -- all matched DuckDB exactly.

## TPC-H-derived benchmarking (unofficial)

**Unofficial TPC-H-derived benchmark. Not a certified TPC result.** Q1, Q3,
Q6, Q12, Q14, and Q19 are supported -- both single-table scans (Q1/Q6) and
multi-table `INNER JOIN` chains (Q3's 3-way `customer`/`orders`/`lineitem`
join, Q12/Q14/Q19's 2-way joins), on both the CPU and GPU execution
backends. See [docs/TPCH.md](docs/TPCH.md) for the full generate -> query
-> validate -> benchmark workflow, including `tools/generate_tpch.py` (a
synthetic generator, not the official `dbgen`) and `kernellake benchmark
tpch`'s cold/warm timing modes.

`tools/benchmark_three_way.py` additionally cross-validates and times
KernelLake (via a persistent `kernellake-server` over Arrow Flight SQL),
PySpark, and DuckDB against the same generated dataset, including a
`--cost-per-hour` flag for a real cost-per-TB-processed comparison across
engines -- see `docs/ROADMAP.md` for real numbers from this.

## Docker

`docker/Dockerfile` is a single multi-stage file publishing two runtime
images: `runtime-cpu` (no CUDA/RAPIDS at all -- 456 MB) and `runtime-gpu`
(full CUDA/RAPIDS closure -- 2.17 GB). Both ship the same two binaries
(`kernellake`, `kernellake-server`); they differ only in whether the GPU
execution backend is compiled in. The `dev-cpu`/`dev-gpu` build stages
(full toolchain, repo built inside them -- the GPU one is 14.1 GB) are
intermediate only and are never published. Every stage builds on plain
`ubuntu:26.04`, with `runtime-gpu`/`dev-gpu` installing CUDA via apt's own
`nvidia-cuda-toolkit` (12.4.1), not Ubuntu 24.04 or NVIDIA's official
`nvidia/cuda` images -- see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)'s
"Ubuntu 26.04 baseline" section for why (Arrow Flight SQL and otel-cpp both
need it). This is independent of the Requirements/Build sections above,
which describe this project's own non-container development environment
-- neither one needs to match the other exactly (this project's own
sandbox happens to be Ubuntu 26.04 too, matching the container, but that's
not a requirement of the non-container path).

```bash
docker build --target runtime-cpu -f docker/Dockerfile -t kernellake/kernellake:runtime-cpu .
docker build --target runtime-gpu -f docker/Dockerfile -t kernellake/kernellake:runtime-gpu .
docker run --rm --gpus all -v /tmp/kernellake-sales:/data:ro \
  kernellake/kernellake:runtime-gpu query --backend gpu --sql "SELECT region, SUM(amount) FROM read_parquet('/data/*.parquet') GROUP BY region"
docker run --rm -v /tmp/kernellake-sales:/data:ro \
  kernellake/kernellake:runtime-cpu query --sql "SELECT region, SUM(amount) FROM read_parquet('/data/*.parquet') GROUP BY region"
```

`.github/workflows/ci.yml`'s `docker-publish` job builds and pushes
`runtime-cpu` to `ghcr.io/hurdad/kernel-lake-cpu:latest` and `runtime-gpu`
to `ghcr.io/hurdad/kernel-lake-gpu:latest` on every push to `main` -- but
only after `format-check`, `cpu-build-test`, and `tpch-tooling-smoke` all
succeed, so a broken build is never shipped as an image.
`kernel-lake-cpu:latest` is a multi-arch (`linux/amd64`+`linux/arm64`)
manifest -- every `runtime-cpu` dependency is a plain apt package with a
real arm64 build on Ubuntu 26.04, confirmed by a real
`docker buildx build --platform linux/arm64` (via QEMU user-mode emulation;
no arm64 hardware or runner was used to verify this). `kernel-lake-gpu` is
`linux/amd64` only for now -- CUDA/RAPIDS' own arm64 support needs real
arm64 GPU hardware (e.g. NVIDIA Grace/Jetson) to verify, which hasn't
happened yet; see `docs/ROADMAP.md`. `docker run --gpus all` against a real
GPU (RTX 5060 Ti) has been verified for real against this Ubuntu 26.04
baseline: the full `gpu-dev` test suite passes in the `dev-gpu` image (see
`docs/ROADMAP.md` for the exact, growing count as of each verified
milestone -- citing one fixed number here would just go stale), and a real
query against real generated data through the `runtime-gpu` image alone
produces correct GPU-executed results -- see `docs/ARCHITECTURE.md`.
`runtime-cpu` has been verified for real too: a real
`docker build --target runtime-cpu`,
`generate-data`, and `query --backend cpu` against real generated data all
produce correct results through the built image. CI's own `docker-publish`
job (building against the previous, pre-cpu/gpu-split Dockerfile) had
separately succeeded end to end on real GitHub Actions; re-confirming CI
still passes with this restructuring is still pending.

## Arrow Flight SQL server and observability

`kernellake-server` (built behind `KERNELLAKE_BUILD_SERVER`, the
`server-dev` preset) serves SQL queries over Arrow Flight SQL instead of
one-shot CLI invocations -- the same `QueryEngine` the CLI uses, so it
supports the identical SQL grammar and both execution backends. Query it
from Python with `adbc-driver-flightsql`, or deploy it to Kubernetes via
`charts/kernellake/` (see that chart's own `README.md`).

Built behind `KERNELLAKE_ENABLE_OTEL` (the `otel-dev` preset), KernelLake
also exports OpenTelemetry traces (one span per query), metrics (a
query-duration histogram; GPU-build-only process/device-level memory
gauges/counters; NVMe cache hit/miss/eviction/size gauges and counters,
`kernellake-server`-only), and every existing log line -- see
[docs/OBSERVABILITY.md](docs/OBSERVABILITY.md) for the full signal list,
config schema, and how to point it at a real collector.

## Project layout

```
include/kernellake/<module>/   public headers, one directory per module
src/<module>/                  implementation, mirrors include/
tests/unit/                    GoogleTest unit tests (CPU-only, every preset)
tests/gpu/                     GoogleTest GPU tests (gpu-dev preset only)
tools/                         Python tooling (DuckDB cross-validation, TPC-H generation)
benchmarks/tpch/queries/       Version-controlled TPC-H-derived SQL (q01/q03/q06/q12/q14/q19.sql)
benchmarks/local/              Single-machine docker-compose stack (kernellake-server + OTel
                                Collector + Prometheus + Grafana + Jaeger + MinIO), no cloud needed
benchmarks/aws/                Real Terraform-provisioned AWS benchmark harness (KernelLake vs.
                                PySpark/DuckDB over real S3 data) -- see its own README.md
docker/                        Dockerfile (runtime-cpu/runtime-gpu images) and related tooling
charts/kernellake/             Helm chart for deploying kernellake-server to Kubernetes
docs/                          ARCHITECTURE.md, ROADMAP.md, TPCH.md, OBSERVABILITY.md, GPU_OPTIMIZATIONS.md
config/kernellake.yaml         default engine configuration
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for what each module owns
and how they depend on each other.

## License

Apache License 2.0 (see `LICENSE` and `NOTICE`). See
`THIRD_PARTY_LICENSES.md` for every dependency actually linked into the
build and its license -- including two NVIDIA components (the CUDA
Toolkit and RAPIDS's vendored `nvcomp`) distributed under NVIDIA's own
proprietary SDK terms rather than an open-source license. TPC-H-derived
benchmarking in this project is clearly labeled as an unofficial,
uncertified derivative -- never presented as an official TPC-H result.
