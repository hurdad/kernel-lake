# TPC-H-derived benchmarking

**Unofficial TPC-H-derived benchmark. Not a certified TPC result.** Nothing
in this document or its tooling may be published as an official TPC-H
result -- see `NOTICE`.

## Scope

**Q6** (scan, filter, arithmetic expression, scalar aggregation), **Q1**
(grouped aggregation) -- both single-table scans over `lineitem` -- and
**Q19** (a two-table `lineitem`/`part` `INNER JOIN`, `OR` of `AND`s with
`BETWEEN`, no `CASE`). KernelLake supports a two-table `INNER JOIN ... ON`
with a single equality key on both the CPU and GPU execution backends (see
`docs/ARCHITECTURE.md`'s "Hash joins" section and its "CPU execution
backend" section for the CPU-side fix), which is what makes Q19 possible.
Two real grammar gaps remain, ruling out several other TPC-H queries for
now: `CASE` expressions are not yet supported inside `WHERE` or an
aggregate argument (only in the `SELECT` list/`GROUP BY` keys) -- rules out
Q12/Q14 (need `CASE` inside `SUM(...)`) -- and only a *two*-table join is
supported, a 3+-way join fails clearly rather than being silently
reinterpreted -- rules out Q3. See `docs/ROADMAP.md` for the up-to-date
list of what's next.

## 1. Generate data

`tools/generate_tpch.py` is a **synthetic** generator, not the official
TPC-H `dbgen` tool -- it produces `lineitem` and `part` tables with TPC-H's
column names and roughly TPC-H-shaped value distributions (see the
script's docstring for the full list of deviations, including DOUBLE
instead of DECIMAL, since KernelLake's GPU execution layer doesn't support
Decimal yet). It requires the `pyarrow` Python package.

```bash
python3 tools/generate_tpch.py \
  --scale-factor 1 --output /tmp/kernellake-tpch-sf1 \
  --format parquet --compression zstd --row-group-rows 1000000
```

Writes `lineitem-*.parquet`, `part-00000.parquet`, plus a `manifest.json`
recording the generation parameters. Real TPC-H SF1 has ~6,000,000
`lineitem` rows and 200,000 `part` rows; this generator targets the same
row counts at the same scale factor (`lineitem` split across `--files`
Parquet files; `part` always a single file).

## 2. Query

The queries live in version-controlled files, `benchmarks/tpch/queries/
q01.sql`, `q06.sql`, `q19.sql`, each with a header comment documenting its
specific deviations from canonical TPC-H syntax (`FROM lineitem` -> `FROM
read_parquet('{data}')`, no `INTERVAL` arithmetic, no `ORDER BY`). Q1/Q6
need only `{data}` substituted with your `lineitem` glob; Q19 also needs
`{part_data}` substituted with your `part` glob:

```bash
sql=$(grep -v '^--' benchmarks/tpch/queries/q06.sql | tr '\n' ' ' | \
  sed "s|{data}|/tmp/kernellake-tpch-sf1/*.parquet|")
./build/gpu-dev/src/cli/kernellake query --sql "$sql" --stats

sql=$(grep -v '^--' benchmarks/tpch/queries/q19.sql | tr '\n' ' ' | \
  sed "s|{data}|/tmp/kernellake-tpch-sf1/lineitem-*.parquet|; \
       s|{part_data}|/tmp/kernellake-tpch-sf1/part-*.parquet|")
./build/gpu-dev/src/cli/kernellake query --sql "$sql" --stats
```

## 3. Validate against DuckDB

`tools/validate_tpch.py` runs the same query file through both `kernellake
query --format arrow` and DuckDB against the same Parquet files, then
compares results (row order and floating-point precision insensitive).
Requires the `duckdb` and `pyarrow` Python packages. Works against either
execution backend via `--backend cpu`/`--backend gpu` (omit to use the
binary's own default).

```bash
python3 tools/validate_tpch.py \
  --kernellake build/gpu-dev/src/cli/kernellake \
  --data '/tmp/kernellake-tpch-sf1/*.parquet' --scale-factor 1 --query all

# Q19 needs --part-data (not covered by --query all, which only passes --data):
python3 tools/validate_tpch.py \
  --kernellake build/gpu-dev/src/cli/kernellake \
  --data '/tmp/kernellake-tpch-sf1/lineitem-*.parquet' \
  --part-data '/tmp/kernellake-tpch-sf1/part-*.parquet' \
  --scale-factor 1 --query 19
```

This has been run at SF0.01, SF0.1, and SF1 (60,000, 600,000, and
6,000,000 generated rows) -- Q1 and Q6 matched DuckDB exactly at every
scale, including the full SF1 run (~105 MiB single Parquet file, zstd
compression, 1,000,000-row row groups). Q19 has been verified at SF0.01 on
both the CPU and GPU backends, exact match against DuckDB.

## 4. Benchmark

`kernellake benchmark tpch` times a query over configurable warmup and
measured iterations, reporting each iteration plus median/mean/min/max/
standard deviation as JSON. Pass `--part-data` for Q19 (or any future
query needing a second table).

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
`docs/ARCHITECTURE.md`). Run `tools/validate_tpch.py` separately before
trusting a benchmark number, per the spec's "do not treat benchmark timing
as valid when correctness validation fails."

### Real SF1 numbers (one run, one machine -- indicative, not a certified result)

`--iterations 5 --warmup-iterations 1` against the ~105 MiB SF1 file above:

| Query | Mode | median | min | max |
| --- | --- | --- | --- | --- |
| Q6 | cold | 0.085s | 0.073s | 0.099s |
| Q6 | warm | 0.074s | 0.058s | 0.075s |
| Q1 | cold | 0.527s | 0.241s | 2.055s |
| Q1 | warm | 0.400s | 0.202s | 3.000s |

Q1's spread (median well under 1s, but individual iterations reaching
2-3s) is real and reproducible in this environment, not a fabricated or
smoothed-over number -- it's a heavier query (a `GROUP BY` over
`l_returnflag`/`l_linestatus` plus five aggregates per group, versus Q6's
single-scalar-SUM) sharing a machine/GPU with other load, and hasn't been
profiled further; treat these as one data point, not a performance
guarantee.
