# KernelLake

**GPU-native analytics for the open lakehouse.**

KernelLake is an open-source GPU-native query engine for Apache Iceberg and
Parquet data lakes. It executes analytical SQL directly against Parquet
datasets, using Apache Arrow-compatible columnar data as its in-memory
representation and NVIDIA GPUs (via RAPIDS libcudf) for accelerated
execution. KernelLake is a compute and query layer, not a storage database.

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
aggregation (`cudf::groupby::streaming_groupby`), and Arrow result
conversion. `peak_gpu_memory_bytes` and `elapsed_wall_seconds` are measured,
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
    └── Projection
        items: [order_id AS order_id]
        └── Filter
            predicate: (order_id < 50)
            └── ParquetScan
                files: 1/2
                row_groups: 1/3
                columns: [order_id]
```

`files: 1/2` and `row_groups: 1/3` mean KernelLake proved the second file
and two of the first file's three row groups couldn't contain any
`order_id < 50` rows, and skipped reading them entirely.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full pipeline, the
GPU operator set, and the CPU/GPU build split (`kernellake query` throws a
clear `ExecutionError` when built without CUDA -- it never silently falls
back to a CPU implementation).

## Requirements

Tested on Ubuntu 24.04, x86_64.

```bash
# Core toolchain
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git pkg-config \
  python3 python3-pip \
  libgtest-dev libbenchmark-dev libspdlog-dev nlohmann-json3-dev libyaml-cpp-dev

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
CMake >= 3.30.4 is required for that preset specifically (newer than Ubuntu
24.04's apt package).

## Build and test

```bash
# CPU-only: SQL parsing through physical planning, pruning, generate-data
cmake --preset dev
cmake --build --preset dev
ctest --preset dev

# GPU-enabled: adds real query execution (needs CUDA Toolkit 12+, an
# NVIDIA GPU, and CMake >= 3.30.4 -- see Requirements above)
cmake --preset gpu-dev
cmake --build --preset gpu-dev
ctest --preset gpu-dev
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

# Run a query for real (requires a gpu-dev build; the dev build's binary
# throws a clear ExecutionError explaining why instead)
./build/gpu-dev/src/cli/kernellake query \
  --sql "SELECT region, SUM(amount) FROM read_parquet('/tmp/kernellake-sales/*.parquet') \
         WHERE event_date >= DATE '2026-01-01' GROUP BY region"
./build/gpu-dev/src/cli/kernellake query --file query.sql --format csv --output result.csv
./build/gpu-dev/src/cli/kernellake query --sql "..." --format jsonl
./build/gpu-dev/src/cli/kernellake query --sql "..." --format arrow --output result.arrow
./build/gpu-dev/src/cli/kernellake query --sql "..." --stats   # prints measured metrics to stderr
```

Supported SQL grammar, the `read_parquet(...)` syntax, and everything
that's intentionally not yet supported are documented in
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

**Unofficial TPC-H-derived benchmark. Not a certified TPC result.** Q6 and
Q1 (both `lineitem`-only, no joins) are supported; see
[docs/TPCH.md](docs/TPCH.md) for the full generate -> query -> validate ->
benchmark workflow, including `tools/generate_tpch.py` (a synthetic
generator, not the official `dbgen`) and `kernellake benchmark tpch`'s
cold/warm timing modes.

## Docker

`docker/Dockerfile` is a single multi-stage file with two named targets:
`dev` (full CUDA devel image, repo built inside it) and `runtime` (only the
built binary plus its actual runtime dependency closure).

```bash
docker build --target dev     -f docker/Dockerfile -t kernellake/kernellake:dev .
docker build --target runtime -f docker/Dockerfile -t kernellake/kernellake:runtime .
docker run --rm --gpus all -v /tmp/kernellake-sales:/data:ro \
  kernellake/kernellake:runtime query --sql "SELECT region, SUM(amount) FROM read_parquet('/data/*.parquet') GROUP BY region"
```

`.github/workflows/docker-publish.yml` builds and pushes both targets to
`ghcr.io/<owner>/<repo>:dev` and `:runtime`/`:latest` on every push to
`main`. Neither the Dockerfile nor that workflow has been exercised by an
actual `docker build`/Actions run in this repository's own development
environment (Docker wasn't available there) -- see
[docs/ROADMAP.md](docs/ROADMAP.md).

## Project layout

```
include/kernellake/<module>/   public headers, one directory per module
src/<module>/                  implementation, mirrors include/
tests/unit/                    GoogleTest unit tests (CPU-only, both presets)
tests/gpu/                     GoogleTest GPU tests (gpu-dev preset only)
tools/                         Python tooling (DuckDB cross-validation, TPC-H generation)
benchmarks/tpch/queries/       Version-controlled TPC-H-derived SQL (q01.sql, q06.sql)
docs/                          ARCHITECTURE.md, ROADMAP.md, TPCH.md
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
