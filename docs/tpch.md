# TPC-H-derived benchmarking

**Unofficial TPC-H-derived benchmark. Not a certified TPC result.** Nothing
in this document or its tooling may be published as an official TPC-H
result -- see the Licensing section of the spec and `LICENSE`/`NOTICE`.

## Scope

Only `lineitem`-only queries are supported today: **Q6** (scan, filter,
arithmetic expression, scalar aggregation) and **Q1** (grouped aggregation).
Q3/Q12/Q14 and the rest of the 22-query suite need hash joins, which
KernelLake does not implement yet (see `docs/roadmap.md`).

## 1. Generate data

`tools/generate_tpch.py` is a **synthetic** generator, not the official
TPC-H `dbgen` tool -- it produces a `lineitem` table with TPC-H's column
names and roughly TPC-H-shaped value distributions (see the script's
docstring for the full list of deviations, including DOUBLE instead of
DECIMAL, since KernelLake's GPU execution layer doesn't support Decimal
yet). It requires the `pyarrow` Python package.

```bash
python3 tools/generate_tpch.py \
  --scale-factor 1 --output /tmp/kernellake-tpch-sf1 \
  --format parquet --compression zstd --row-group-rows 1000000
```

Writes `lineitem-*.parquet` plus a `manifest.json` recording the generation
parameters. Real TPC-H SF1 has ~6,000,000 lineitem rows; this generator
targets the same row count at the same scale factor, split across
`--files` Parquet files.

## 2. Query

The queries live in version-controlled files, `benchmarks/tpch/queries/
q01.sql` and `q06.sql`, each with a header comment documenting its specific
deviations from canonical TPC-H syntax (`FROM lineitem` -> `FROM
read_parquet('{data}')`, no `INTERVAL` arithmetic, no `ORDER BY`). Substitute
`{data}` with your glob and run directly:

```bash
sql=$(grep -v '^--' benchmarks/tpch/queries/q06.sql | tr '\n' ' ' | \
  sed "s|{data}|/tmp/kernellake-tpch-sf1/*.parquet|")
./build/gpu-dev/src/cli/kernellake query --sql "$sql" --stats
```

## 3. Validate against DuckDB

`tools/validate_tpch.py` runs the same query file through both `kernellake
query --format arrow` and DuckDB against the same Parquet files, then
compares results (row order and floating-point precision insensitive).
Requires the `duckdb` and `pyarrow` Python packages and a `gpu-dev` build.

```bash
python3 tools/validate_tpch.py \
  --kernellake build/gpu-dev/src/cli/kernellake \
  --data '/tmp/kernellake-tpch-sf1/*.parquet' --scale-factor 1 --query all
```

This has been run at SF0.01 and SF0.1 (60,000 and 600,000 generated rows) --
both Q1 and Q6 matched DuckDB exactly.

## 4. Benchmark

`kernellake benchmark tpch` times a query over configurable warmup and
measured iterations, reporting each iteration plus median/mean/min/max/
standard deviation as JSON.

```bash
./build/gpu-dev/src/cli/kernellake benchmark tpch \
  --data '/tmp/kernellake-tpch-sf1/*.parquet' --scale-factor 1 \
  --query 6 --iterations 5 --mode cold --output tpch-q6-sf1.json
```

Three modes are named in the spec; two are implemented:

- **`cold`**: best-effort per-file `posix_fadvise(POSIX_FADV_DONTNEED)`
  before every iteration, hinting the kernel to evict that file's cached
  pages. This is a real technique that works without root, but it is a
  hint, not a guarantee -- the OS-wide page cache cannot be dropped without
  root, and the JSON report's `cache_clearing` field says so explicitly
  rather than implying a guaranteed-cold run.
- **`warm`**: no cache manipulation; measures whatever the warmup
  iterations left cached.
- **`execution-only`**: **not implemented.** It would need an
  operator-tree entry point that skips `ParquetScanOperator` entirely and
  feeds already-decoded `DeviceBatch`es straight to the rest of the
  pipeline, which doesn't exist yet. The CLI refuses this mode with a clear
  error rather than silently falling back to `cold` or `warm`.

The report's `result_validation_performed` is always `false` -- the
benchmark command does not call DuckDB in-process (no `libduckdb`
dependency has been added to the C++ build; see
`docs/architecture.md`). Run `tools/validate_tpch.py` separately before
trusting a benchmark number, per the spec's "do not treat benchmark timing
as valid when correctness validation fails."
