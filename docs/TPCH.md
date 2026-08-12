# TPC-H-derived benchmarking

**Unofficial TPC-H-derived benchmark. Not a certified TPC result.** Nothing
in this document or its tooling may be published as an official TPC-H
result -- see `NOTICE`.

## Scope

**Q6** (scan, filter, arithmetic expression, scalar aggregation), **Q1**
(grouped aggregation) -- both single-table scans over `lineitem` --
**Q19** (a two-table `lineitem`/`part` `INNER JOIN`, `OR` of `AND`s with
`BETWEEN`, no `CASE`), **Q12** (a two-table `orders`/`lineitem`
`INNER JOIN`, `CASE` inside a grouped aggregate argument), **Q14** (a
two-table `lineitem`/`part` `INNER JOIN`, `LIKE` inside a `CASE` inside a
*scalar* aggregate argument, two aggregates combined arithmetically into a
ratio), **Q3** (a three-table `customer`/`orders`/`lineitem`
`INNER JOIN` chain, a multi-key `ORDER BY` plus `LIMIT` after the join),
**Q10** (a four-table `customer`/`orders`/`lineitem`/`nation`
`INNER JOIN` chain, a 7-column `GROUP BY` spanning two non-adjacent join
sources, `ORDER BY` plus `LIMIT`), and **Q5** (a six-table `customer`/
`orders`/`lineitem`/`supplier`/`nation`/`region` `INNER JOIN` chain, plus
a `WHERE`-clause equality spanning two non-adjacent join sources --
`c_nationkey = s_nationkey` -- that can't be expressed as part of a
single-key `JOIN ... ON` step). KernelLake supports a chain of two or
more tables via `INNER JOIN ... ON`, each step a single equality key, on
both the CPU and GPU execution backends (see `docs/ARCHITECTURE.md`'s
"Hash joins" section, including its N-way-join generalization, and its
"CPU execution backend" section for the CPU-side fix); `CASE` inside an
aggregate argument (grouped *or* scalar), `LIKE` inside a `CASE` branch,
and a `SELECT` item that combines multiple aggregates arithmetically
(Q14's `100.00 * SUM(...) / SUM(...)`, rather than a single bare
aggregate call) all now work on both backends too (see
`docs/ARCHITECTURE.md`'s "LIKE/IN/CASE/CAST implementation notes" and the
entries just above it).

`CASE` inside `WHERE` on the GPU backend remains unsupported (a separate
`FilterOperator` gap; the CPU backend already supports it) but isn't
needed by any TPC-H query added so far. See `docs/ROADMAP.md` for the
up-to-date list of what's next.

## 1. Generate data

`tools/generate_tpch.py` is a **synthetic** generator, not the official
TPC-H `dbgen` tool -- it produces `lineitem`, `part`, `orders`, `customer`,
`nation`, `region`, and `supplier` tables with TPC-H's column names and
roughly TPC-H-shaped value distributions (`nation`/`region` are the
exception: they're small enough that the script hardcodes real `dbgen`
reference data instead of sampling -- see the script's docstring for the
full list of deviations, including DOUBLE instead of DECIMAL, since
KernelLake's GPU execution layer doesn't support Decimal yet). It
requires the `pyarrow` Python package.

```bash
python3 tools/generate_tpch.py \
  --scale-factor 1 --output /tmp/kernellake-tpch-sf1 \
  --format parquet --compression zstd --row-group-rows 1000000
```

Writes `lineitem-*.parquet`, `part-00000.parquet`, `orders-00000.parquet`,
`customer-00000.parquet`, `nation-00000.parquet`, `region-00000.parquet`,
`supplier-00000.parquet`, plus a `manifest.json` recording the generation
parameters. Real TPC-H SF1 has ~6,000,000 `lineitem` rows, 200,000 `part`
rows, ~1,500,000 `orders` rows, and 150,000 `customer` rows; this
generator targets the same `lineitem`/`part`/`orders` row counts at the
same scale factor (`lineitem` split across `--files` Parquet files;
`part`/`orders` always a single file each -- `orders` gets exactly one
row per distinct `l_orderkey` generated, per TPC-H's own 1:N
orders:lineitem relationship). `customer` is a fixed 150,000 rows
regardless of scale factor (matching TPC-H's own SF1 customer count),
also written as a single file. `nation` is a fixed 25 rows and `region` a
fixed 5 rows regardless of scale factor (real TPC-H's own counts, and
real `dbgen` reference data -- names and region keys -- rather than
sampled values), each a single file. `supplier` is a fixed 10,000 rows
regardless of scale factor (matching this generator's own fixed
`l_suppkey` range, see the script's docstring), also a single file.

## 2. Query

The queries live in version-controlled files, `benchmarks/tpch/queries/
q01.sql`, `q03.sql`, `q05.sql`, `q06.sql`, `q10.sql`, `q12.sql`, `q14.sql`,
`q19.sql`, each with a header comment documenting its specific deviations
from canonical TPC-H syntax (`FROM lineitem` -> `FROM
read_parquet('{data}')`, no `INTERVAL` arithmetic). Q1/Q6 need only
`{data}` substituted with your `lineitem` glob; Q19/Q14 also need
`{part_data}` substituted with your `part` glob; Q12 also needs
`{orders_data}` substituted with your `orders` glob; Q3 needs both
`{orders_data}` and `{customer_data}` substituted with your `orders` and
`customer` globs; Q10 needs `{orders_data}`, `{customer_data}`, and
`{nation_data}` substituted with your `orders`, `customer`, and `nation`
globs; Q5 needs `{orders_data}`, `{customer_data}`, `{supplier_data}`,
`{nation_data}`, and `{region_data}` substituted with your `orders`,
`customer`, `supplier`, `nation`, and `region` globs. `{data}` must always
be a `lineitem`-specific glob (e.g. `lineitem-*.parquet`), not a bare
`*.parquet` -- `generate_tpch.py` writes every table into the same output
directory, so a bare glob would pull in all of them at once and fail with
a schema mismatch:

```bash
sql=$(grep -v '^--' benchmarks/tpch/queries/q06.sql | tr '\n' ' ' | \
  sed "s|{data}|/tmp/kernellake-tpch-sf1/lineitem-*.parquet|")
./build/gpu-dev/src/cli/kernellake query --sql "$sql" --stats

sql=$(grep -v '^--' benchmarks/tpch/queries/q19.sql | tr '\n' ' ' | \
  sed "s|{data}|/tmp/kernellake-tpch-sf1/lineitem-*.parquet|; \
       s|{part_data}|/tmp/kernellake-tpch-sf1/part-*.parquet|")
./build/gpu-dev/src/cli/kernellake query --sql "$sql" --stats

sql=$(grep -v '^--' benchmarks/tpch/queries/q14.sql | tr '\n' ' ' | \
  sed "s|{data}|/tmp/kernellake-tpch-sf1/lineitem-*.parquet|; \
       s|{part_data}|/tmp/kernellake-tpch-sf1/part-*.parquet|")
./build/gpu-dev/src/cli/kernellake query --sql "$sql" --stats

sql=$(grep -v '^--' benchmarks/tpch/queries/q12.sql | tr '\n' ' ' | \
  sed "s|{data}|/tmp/kernellake-tpch-sf1/lineitem-*.parquet|; \
       s|{orders_data}|/tmp/kernellake-tpch-sf1/orders-*.parquet|")
./build/gpu-dev/src/cli/kernellake query --sql "$sql" --stats

sql=$(grep -v '^--' benchmarks/tpch/queries/q03.sql | tr '\n' ' ' | \
  sed "s|{data}|/tmp/kernellake-tpch-sf1/lineitem-*.parquet|; \
       s|{orders_data}|/tmp/kernellake-tpch-sf1/orders-*.parquet|; \
       s|{customer_data}|/tmp/kernellake-tpch-sf1/customer-*.parquet|")
./build/gpu-dev/src/cli/kernellake query --sql "$sql" --stats

sql=$(grep -v '^--' benchmarks/tpch/queries/q10.sql | tr '\n' ' ' | \
  sed "s|{data}|/tmp/kernellake-tpch-sf1/lineitem-*.parquet|; \
       s|{orders_data}|/tmp/kernellake-tpch-sf1/orders-*.parquet|; \
       s|{customer_data}|/tmp/kernellake-tpch-sf1/customer-*.parquet|; \
       s|{nation_data}|/tmp/kernellake-tpch-sf1/nation-*.parquet|")
./build/gpu-dev/src/cli/kernellake query --sql "$sql" --stats

sql=$(grep -v '^--' benchmarks/tpch/queries/q05.sql | tr '\n' ' ' | \
  sed "s|{data}|/tmp/kernellake-tpch-sf1/lineitem-*.parquet|; \
       s|{orders_data}|/tmp/kernellake-tpch-sf1/orders-*.parquet|; \
       s|{customer_data}|/tmp/kernellake-tpch-sf1/customer-*.parquet|; \
       s|{supplier_data}|/tmp/kernellake-tpch-sf1/supplier-*.parquet|; \
       s|{nation_data}|/tmp/kernellake-tpch-sf1/nation-*.parquet|; \
       s|{region_data}|/tmp/kernellake-tpch-sf1/region-*.parquet|")
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
  --data '/tmp/kernellake-tpch-sf1/lineitem-*.parquet' --scale-factor 1 --query 1
python3 tools/validate_tpch.py \
  --kernellake build/gpu-dev/src/cli/kernellake \
  --data '/tmp/kernellake-tpch-sf1/lineitem-*.parquet' --scale-factor 1 --query 6

# Q19/Q14 need --part-data, Q12 needs --orders-data, Q3 needs both
# --orders-data and --customer-data, Q10 needs --orders-data,
# --customer-data, and --nation-data, Q5 needs --orders-data,
# --customer-data, --supplier-data, --nation-data, and --region-data
# (none covered by --query all, which only passes --data):
python3 tools/validate_tpch.py \
  --kernellake build/gpu-dev/src/cli/kernellake \
  --data '/tmp/kernellake-tpch-sf1/lineitem-*.parquet' \
  --part-data '/tmp/kernellake-tpch-sf1/part-*.parquet' \
  --scale-factor 1 --query 19
python3 tools/validate_tpch.py \
  --kernellake build/gpu-dev/src/cli/kernellake \
  --data '/tmp/kernellake-tpch-sf1/lineitem-*.parquet' \
  --part-data '/tmp/kernellake-tpch-sf1/part-*.parquet' \
  --scale-factor 1 --query 14
python3 tools/validate_tpch.py \
  --kernellake build/gpu-dev/src/cli/kernellake \
  --data '/tmp/kernellake-tpch-sf1/lineitem-*.parquet' \
  --orders-data '/tmp/kernellake-tpch-sf1/orders-*.parquet' \
  --scale-factor 1 --query 12
python3 tools/validate_tpch.py \
  --kernellake build/gpu-dev/src/cli/kernellake \
  --data '/tmp/kernellake-tpch-sf1/lineitem-*.parquet' \
  --orders-data '/tmp/kernellake-tpch-sf1/orders-*.parquet' \
  --customer-data '/tmp/kernellake-tpch-sf1/customer-*.parquet' \
  --scale-factor 1 --query 3
python3 tools/validate_tpch.py \
  --kernellake build/gpu-dev/src/cli/kernellake \
  --data '/tmp/kernellake-tpch-sf1/lineitem-*.parquet' \
  --orders-data '/tmp/kernellake-tpch-sf1/orders-*.parquet' \
  --customer-data '/tmp/kernellake-tpch-sf1/customer-*.parquet' \
  --nation-data '/tmp/kernellake-tpch-sf1/nation-*.parquet' \
  --scale-factor 1 --query 10
python3 tools/validate_tpch.py \
  --kernellake build/gpu-dev/src/cli/kernellake \
  --data '/tmp/kernellake-tpch-sf1/lineitem-*.parquet' \
  --orders-data '/tmp/kernellake-tpch-sf1/orders-*.parquet' \
  --customer-data '/tmp/kernellake-tpch-sf1/customer-*.parquet' \
  --supplier-data '/tmp/kernellake-tpch-sf1/supplier-*.parquet' \
  --nation-data '/tmp/kernellake-tpch-sf1/nation-*.parquet' \
  --region-data '/tmp/kernellake-tpch-sf1/region-*.parquet' \
  --scale-factor 1 --query 5
```

This has been run at SF0.01, SF0.1, and SF1 (60,000, 600,000, and
6,000,000 generated rows) -- Q1 and Q6 matched DuckDB exactly at every
scale, including the full SF1 run (~105 MiB single Parquet file, zstd
compression, 1,000,000-row row groups). Q19, Q12, Q14, Q3, Q10, and Q5
have each been verified at SF0.01 on both the CPU and GPU backends, exact
match against DuckDB (Q3 including its 3-way join, `ORDER BY revenue
DESC, o_orderdate` multi-key sort, and `LIMIT 10`; Q10 including its
4-way join, 7-column `GROUP BY` spanning two non-adjacent join sources,
and `LIMIT 20`; Q5 including its 6-way join and the `c_nationkey =
s_nationkey` `WHERE`-clause predicate spanning two non-adjacent join
sources) -- Q19/Q12/Q14/Q3 also cross-validated against PySpark and
DuckDB via `tools/benchmark_three_way.py` (KernelLake via a persistent
`kernellake-server` over Arrow Flight SQL, PySpark, and DuckDB all agree
on every query, real GPU hardware, SF0.01/SF1/SF10 -- see
`docs/ROADMAP.md` for the full crossover numbers and a
`--cost-per-hour`-based cost-per-TB comparison); Q10/Q5 have not been
wired into `benchmark_three_way.py` yet (see `docs/ROADMAP.md`'s "Not yet
started").

## 4. Benchmark

`kernellake benchmark tpch` times a query over configurable warmup and
measured iterations, reporting each iteration plus median/mean/min/max/
standard deviation as JSON. Pass `--part-data` for Q19/Q14, `--orders-data`
for Q12, `--orders-data` and `--customer-data` for Q3, `--orders-data`,
`--customer-data`, and `--nation-data` for Q10, or `--orders-data`,
`--customer-data`, `--supplier-data`, `--nation-data`, and `--region-data`
for Q5 (or any combination for a future query needing those extra
tables).

```bash
./build/gpu-dev/src/cli/kernellake benchmark tpch \
  --data '/tmp/kernellake-tpch-sf1/lineitem-*.parquet' --scale-factor 1 \
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
