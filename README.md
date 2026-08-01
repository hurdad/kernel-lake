# KernelLake

**GPU-native analytics for the open lakehouse.**

KernelLake is an open-source GPU-native query engine for Apache Iceberg and
Parquet data lakes. It executes analytical SQL directly against Parquet
datasets, using Apache Arrow-compatible columnar data as its in-memory
representation and NVIDIA GPUs (via RAPIDS libcudf) for accelerated
execution. KernelLake is a compute and query layer, not a storage database.

## Status

KernelLake is an early work in progress. The full SQL-to-physical-plan
compiler is built and tested; **GPU execution is not implemented yet** (see
[docs/roadmap.md](docs/roadmap.md) for exactly what's done vs. blocked, and
why). Concretely, this is real, reproducible output against a small local
Parquet file (10 rows, 2 row groups, columns `id`/`amount`/`region`):

```
$ kernellake explain --sql "SELECT region, SUM(amount) AS total_amount \
    FROM read_parquet('/tmp/demo.parquet') WHERE region = 'west' GROUP BY region"

ArrowResult
    └── HashAggregate
        group_by: [region]
        aggregates: [SUM(amount) AS total_amount]
        └── Filter
            predicate: (region = 'west')
            └── ParquetScan
                files: 1/1
                row_groups: 1/2
                columns: [amount, region]
```

`row_groups: 1/2` is a real pruning decision: KernelLake read the file's
min/max statistics and proved the other row group couldn't contain any
`region = 'west'` rows, so it won't be scanned. `columns: [amount, region]`
is real projection pushdown too -- `id` was never referenced by this query,
so it's excluded. That plan reflects real SQL parsing, real
binding/type-checking, a real rule-based optimizer, real file discovery,
and real Parquet metadata inspection, all against an actual file on disk.
`kernellake query` (which would actually execute this plan on a GPU) isn't
implemented yet -- it needs libcudf/RMM, which aren't part of this build.
See [docs/architecture.md](docs/architecture.md) for the full pipeline and
the CPU/GPU build split.

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
libcudf/RMM, neither of which is required to build or test anything in this
repository today -- see [docs/roadmap.md](docs/roadmap.md) for the current
GPU-dependency status.

## Build and test

```bash
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

This builds the CPU-only `dev` preset (SQL parsing through physical
planning and pruning; no CUDA). All unit tests should pass without a GPU.

## Usage

```bash
# Inspect a Parquet file's schema, row groups, and column statistics
./build/dev/src/cli/kernellake inspect-parquet --path /data/sales.parquet
./build/dev/src/cli/kernellake inspect-parquet --path /data/sales.parquet --format json

# See the plan KernelLake would run for a query (no execution yet)
./build/dev/src/cli/kernellake explain \
  --sql "SELECT region, SUM(amount) FROM read_parquet('/data/sales/*.parquet') GROUP BY region"
./build/dev/src/cli/kernellake explain --logical --sql "..."   # pre-physical-planning view
./build/dev/src/cli/kernellake explain --format json --sql "..."
```

Supported SQL grammar, the `read_parquet(...)` syntax, and everything
that's intentionally not yet supported are documented in
[docs/architecture.md](docs/architecture.md).

## Project layout

```
include/kernellake/<module>/   public headers, one directory per module
src/<module>/                  implementation, mirrors include/
tests/unit/                    GoogleTest unit tests
docs/                          architecture.md, roadmap.md
config/kernellake.yaml         default engine configuration
```

See [docs/architecture.md](docs/architecture.md) for what each module owns
and how they depend on each other.

## License

Apache License 2.0 (see `LICENSE`). TPC-H-derived benchmarking in this
project, once implemented, will be clearly labeled as an unofficial,
uncertified derivative -- never presented as an official TPC-H result.
