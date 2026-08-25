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
sources, `ORDER BY` plus `LIMIT`), **Q5** (a six-table `customer`/
`orders`/`lineitem`/`supplier`/`nation`/`region` `INNER JOIN` chain, plus
a `WHERE`-clause equality spanning two non-adjacent join sources --
`c_nationkey = s_nationkey` -- that can't be expressed as part of a
single-key `JOIN ... ON` step), **Q7** (a six-table `supplier`/
`lineitem`/`orders`/`customer`/`nation`/`nation` `INNER JOIN` chain --
`nation` joined *twice*, once per alias, to resolve both the supplier's
and the customer's own nation independently -- `OR` of `AND`s in `WHERE`,
`BETWEEN`, and a flattened-out-of-a-derived-table `EXTRACT`-as-`GROUP
BY`-key), and **Q9** (a six-table `part`/`supplier`/`lineitem`/
`partsupp`/`orders`/`nation` `INNER JOIN` chain, `LIKE`, and a genuinely
two-column join condition -- `partsupp` to `lineitem` on *both*
`ps_partkey = l_partkey` and `ps_suppkey = l_suppkey` together -- split
across a `JOIN ... ON` key and a `WHERE`-clause filter the same way Q5's
own cross-join-source predicate already is, since KernelLake has no
multi-key `JOIN ... ON` support), **Q11** (a three-table `partsupp`/
`supplier`/`nation` `INNER JOIN` chain, `GROUP BY` + `HAVING`, and a
non-correlated scalar subquery computing `HAVING`'s own threshold --
the first query in this project needing either `HAVING` or a subquery,
both now genuinely supported, not flattened away the way Q5/Q7/Q9's own
additions were; see `docs/ARCHITECTURE.md`'s "`HAVING` and scalar
subqueries" section for the full scope), and **Q18** (a three-table
`customer`/`orders`/`lineitem` `INNER JOIN` chain, plus a non-correlated
`IN (SELECT ...)` subquery in `WHERE` -- unlike Q11's HAVING subquery,
this one may return many rows, resolved into a literal list the same way
a literal `IN (...)` list already desugars; see `docs/ARCHITECTURE.md`'s
"`IN (SELECT ...)` subqueries" section), and **Q13** (a `customer`/`orders`
`LEFT OUTER JOIN` whose own `ON` clause combines the required equality key
with an extra predicate scoped to just the newly-joined side --
`o_comment NOT LIKE '%special%requests%'` -- plus a derived table,
`FROM (SELECT ...) AS c_orders`, whose own inner query's grouped `COUNT`
output the *outer* query groups by again -- the first query in this
project needing a `LEFT OUTER JOIN`, an `ON`-clause predicate beyond the
bare equality key, or a derived table, all three genuinely supported (not
flattened away); see `docs/ARCHITECTURE.md`'s "Hash joins" section for the
`LEFT OUTER JOIN`/`ON`-clause-predicate scope and its "Derived tables"
section for the `FROM`-subquery scope), and **Q4** (an `orders`-only outer
query whose `WHERE` clause combines an `o_orderdate` range with a
correlated `EXISTS (SELECT * FROM lineitem WHERE l_orderkey = o_orderkey
AND l_commitdate < l_receiptdate)` -- the first query in this project
needing a correlated subquery, rewritten internally into a `LEFT SEMI
JOIN` reusing the same ON-clause-auxiliary-predicate machinery Q13's
`LEFT OUTER JOIN` already exercises; see `docs/ARCHITECTURE.md`'s
"Correlated subqueries" section for the exact `EXISTS`/`NOT EXISTS` scope),
and **Q8** (an 8-way `part`/`lineitem`/`supplier`/`orders`/`customer`/
`nation`/`region`/`nation` `INNER JOIN` chain -- `nation` joined twice,
Q7's own pattern -- wrapped in a derived table kept exactly as canonical
TPC-H writes it (unlike Q7's own flattened deviation), the first query
combining a derived table with a real `JOIN` *inside* that derived
table's own `FROM`. Found and fixed a real bug along the way: an outer
`GROUP BY` over a derived table whose own inner query has no `GROUP BY`
of its own (a plain JOIN projection, Q8's shape -- unlike Q13's, whose
inner query is itself aggregated) crashed at execution time; see
`docs/ARCHITECTURE.md`'s "Derived tables" section for the fix), and
**Q15** (a `supplier`/`lineitem` `INNER JOIN` with a `HAVING` comparison
against a non-correlated scalar subquery whose own `FROM` is a derived
table with its own `GROUP BY` feeding an outer `MAX(...)` -- canonical
Q15's shared `revenue0` view, inlined twice since KernelLake has no
`CREATE VIEW`/`WITH` support. Found and fixed a real bug along the way:
`QueryEngine::run_subquery()` had no case for a subquery whose own
`FROM` is a derived table at all; also found and documented (not yet
fixed) a real GPU-backend limitation this query's own exact `=`
comparison exposes -- see `docs/ARCHITECTURE.md`'s "`HAVING` and scalar
subqueries" section for both. **CPU-backend only** -- see `q15.sql`'s
own header and the "GPU backend caveat" note below).
KernelLake supports a chain of two
or more tables via `INNER` or `LEFT OUTER JOIN ... ON`, each step a single
equality key (`LEFT OUTER` additionally allows an `ON`-clause predicate
scoped to just the newly-joined side), on both the CPU and GPU execution
backends (see `docs/ARCHITECTURE.md`'s "Hash joins" section, including its
N-way-join generalization, and its "CPU execution backend" section for the
CPU-side fix); `CASE` inside an
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
TPC-H `dbgen` tool -- it produces `lineitem`, `part`, `partsupp`,
`orders`, `customer`, `nation`, `region`, and `supplier` tables with
TPC-H's column names and roughly TPC-H-shaped value distributions
(`nation`/`region` are the exception: they're small enough that the
script hardcodes real `dbgen` reference data instead of sampling; `part`'s
own `p_name` uses a representative subset of real `dbgen`'s color-word
list rather than a purely synthetic string, specifically so Q9's `p_name
LIKE '%green%'` filter matches real rows -- see the script's docstring
for the full list of deviations, including DOUBLE instead of DECIMAL --
KernelLake's GPU execution layer does support DECIMAL now (see
`docs/ARCHITECTURE.md`'s "DECIMAL support" section), so this is a
generator choice, not an engine limitation). It requires the `pyarrow`
Python package.

`l_suppkey` is the one foreign key in this generator that is **not**
drawn independently at random: every part is stocked by exactly 4
suppliers (`PARTSUPP_SUPPLIERS_PER_PART`, matching real TPC-H's own fixed
count), and `generate_lineitem_batch()`'s own `l_suppkey` draw picks from
that same part's 4 suppliers -- otherwise Q9's join of `lineitem` to
`partsupp` on both `ps_partkey`/`ps_suppkey` together would essentially
never find a matching row. See `suppliers_for_part()`'s own doc comment.

```bash
python3 tools/generate_tpch.py \
  --scale-factor 1 --output /tmp/kernellake-tpch-sf1 \
  --format parquet --compression zstd --row-group-rows 1000000
```

Writes `lineitem-*.parquet`, `part-00000.parquet`, `partsupp-00000.parquet`,
`orders-00000.parquet`, `customer-00000.parquet`, `nation-00000.parquet`,
`region-00000.parquet`, `supplier-00000.parquet`, plus a `manifest.json`
recording the generation parameters. Real TPC-H SF1 has ~6,000,000
`lineitem` rows, 200,000 `part` rows, ~1,500,000 `orders` rows, and
150,000 `customer` rows; this
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
`partsupp` scales directly with `part` (exactly 4 rows per part, real
TPC-H's own fixed `PARTSUPP_SUPPLIERS_PER_PART` -- 800,000 rows at SF1),
also a single file.

## 2. Query

The queries live in version-controlled files, `benchmarks/tpch/queries/
q01.sql`, `q03.sql`, `q04.sql`, `q05.sql`, `q06.sql`, `q07.sql`, `q08.sql`,
`q09.sql`, `q10.sql`, `q11.sql`, `q12.sql`, `q13.sql`, `q14.sql`,
`q15.sql`, `q18.sql`, `q19.sql`, each with a header comment documenting
its specific deviations from canonical TPC-H syntax (`FROM lineitem` ->
`FROM read_parquet('{data}')`, no `INTERVAL` arithmetic). Q1/Q6 need only
`{data}` substituted with your `lineitem` glob; Q19/Q14 also need
`{part_data}` substituted with your `part` glob; Q12 also needs
`{orders_data}` substituted with your `orders` glob; Q4 needs the same
`{data}`/`{orders_data}` pair as Q12 (its correlated `EXISTS` subquery is
the one place `{data}` appears, its outer `FROM` the one place
`{orders_data}` does); Q3 needs both
`{orders_data}` and `{customer_data}` substituted with your `orders` and
`customer` globs; Q10 needs `{orders_data}`, `{customer_data}`, and
`{nation_data}` substituted with your `orders`, `customer`, and `nation`
globs; Q5 needs `{orders_data}`, `{customer_data}`, `{supplier_data}`,
`{nation_data}`, and `{region_data}` substituted with your `orders`,
`customer`, `supplier`, `nation`, and `region` globs; Q7 needs
`{orders_data}`, `{customer_data}`, `{supplier_data}`, and `{nation_data}`
substituted with your `orders`, `customer`, `supplier`, and `nation` globs
(`{nation_data}` appears *twice* in Q7's own text -- once per alias --
and both get the same substitution); Q8 needs `{part_data}`, `{data}`,
`{supplier_data}`, `{orders_data}`, `{customer_data}`, `{nation_data}`,
and `{region_data}` substituted with your `part`, `lineitem`, `supplier`,
`orders`, `customer`, `nation`, and `region` globs (`{nation_data}`
appears *twice*, same as Q7, both get the same substitution); Q9 needs
`{part_data}`,
`{supplier_data}`, `{partsupp_data}`, `{orders_data}`, and
`{nation_data}` substituted with your `part`, `supplier`, `partsupp`,
`orders`, and `nation` globs; Q11 needs `{partsupp_data}`,
`{supplier_data}`, and `{nation_data}` substituted with your `partsupp`,
`supplier`, and `nation` globs (each appears *twice* -- once in the outer
query, once in its own independently-scoped subquery -- and both get the
same substitution). Q11 doesn't reference `lineitem` at all, so `{data}`
never appears in its text -- `--data`/`{data}` is still unconditionally
required by this project's own tooling regardless of query, so it still
needs *some* real `lineitem` glob passed; it's simply never substituted
into Q11's query text. Q18 needs `{customer_data}`, `{orders_data}`, and
`{data}` substituted with your `customer`, `orders`, and `lineitem`
globs (`{data}` appears *twice* -- once in the outer join, once in its
own independently-scoped subquery -- and both get the same
substitution). Q13 needs `{customer_data}` and `{orders_data}`
substituted with your `customer` and `orders` globs -- like Q11, it
doesn't reference `lineitem` at all, so `{data}`/`--data` is still
required by this project's own tooling but never actually substituted
into Q13's query text. Q15 needs `{data}` and `{supplier_data}`
substituted with your `lineitem` and `supplier` globs (each appears
*twice* -- once in the outer `JOIN`, once again in the `HAVING`
subquery's own re-`JOIN` -- both get the same substitution).
`{data}` must always be a `lineitem`-specific
glob (e.g. `lineitem-*.parquet`), not a bare `*.parquet` --
`generate_tpch.py` writes every table into the same output directory, so
a bare glob would pull in all of them at once and fail with a schema
mismatch:

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

sql=$(grep -v '^--' benchmarks/tpch/queries/q04.sql | tr '\n' ' ' | \
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

sql=$(grep -v '^--' benchmarks/tpch/queries/q07.sql | tr '\n' ' ' | \
  sed "s|{data}|/tmp/kernellake-tpch-sf1/lineitem-*.parquet|; \
       s|{orders_data}|/tmp/kernellake-tpch-sf1/orders-*.parquet|; \
       s|{customer_data}|/tmp/kernellake-tpch-sf1/customer-*.parquet|; \
       s|{supplier_data}|/tmp/kernellake-tpch-sf1/supplier-*.parquet|; \
       s|{nation_data}|/tmp/kernellake-tpch-sf1/nation-*.parquet|")
./build/gpu-dev/src/cli/kernellake query --sql "$sql" --stats

# Q8's own {nation_data} appears twice -- once per alias, same as Q7's.
sql=$(grep -v '^--' benchmarks/tpch/queries/q08.sql | tr '\n' ' ' | \
  sed "s|{data}|/tmp/kernellake-tpch-sf1/lineitem-*.parquet|; \
       s|{part_data}|/tmp/kernellake-tpch-sf1/part-*.parquet|; \
       s|{supplier_data}|/tmp/kernellake-tpch-sf1/supplier-*.parquet|; \
       s|{orders_data}|/tmp/kernellake-tpch-sf1/orders-*.parquet|; \
       s|{customer_data}|/tmp/kernellake-tpch-sf1/customer-*.parquet|; \
       s|{nation_data}|/tmp/kernellake-tpch-sf1/nation-*.parquet|g; \
       s|{region_data}|/tmp/kernellake-tpch-sf1/region-*.parquet|")
./build/gpu-dev/src/cli/kernellake query --sql "$sql" --stats

sql=$(grep -v '^--' benchmarks/tpch/queries/q09.sql | tr '\n' ' ' | \
  sed "s|{data}|/tmp/kernellake-tpch-sf1/lineitem-*.parquet|; \
       s|{part_data}|/tmp/kernellake-tpch-sf1/part-*.parquet|; \
       s|{supplier_data}|/tmp/kernellake-tpch-sf1/supplier-*.parquet|; \
       s|{partsupp_data}|/tmp/kernellake-tpch-sf1/partsupp-*.parquet|; \
       s|{orders_data}|/tmp/kernellake-tpch-sf1/orders-*.parquet|; \
       s|{nation_data}|/tmp/kernellake-tpch-sf1/nation-*.parquet|")
./build/gpu-dev/src/cli/kernellake query --sql "$sql" --stats

# Q11 doesn't reference {data} at all -- only {partsupp_data}/
# {supplier_data}/{nation_data}, each substituted twice (outer query and
# subquery).
sql=$(grep -v '^--' benchmarks/tpch/queries/q11.sql | tr '\n' ' ' | \
  sed "s|{partsupp_data}|/tmp/kernellake-tpch-sf1/partsupp-*.parquet|g; \
       s|{supplier_data}|/tmp/kernellake-tpch-sf1/supplier-*.parquet|g; \
       s|{nation_data}|/tmp/kernellake-tpch-sf1/nation-*.parquet|g")
./build/gpu-dev/src/cli/kernellake query --sql "$sql" --stats

# Q18's own {data} appears twice -- the outer JOIN and the IN-subquery's
# own FROM.
sql=$(grep -v '^--' benchmarks/tpch/queries/q18.sql | tr '\n' ' ' | \
  sed "s|{data}|/tmp/kernellake-tpch-sf1/lineitem-*.parquet|g; \
       s|{customer_data}|/tmp/kernellake-tpch-sf1/customer-*.parquet|g; \
       s|{orders_data}|/tmp/kernellake-tpch-sf1/orders-*.parquet|g")
./build/gpu-dev/src/cli/kernellake query --sql "$sql" --stats

# Q13 doesn't reference {data} at all -- only {customer_data}/{orders_data}.
sql=$(grep -v '^--' benchmarks/tpch/queries/q13.sql | tr '\n' ' ' | \
  sed "s|{customer_data}|/tmp/kernellake-tpch-sf1/customer-*.parquet|; \
       s|{orders_data}|/tmp/kernellake-tpch-sf1/orders-*.parquet|")
./build/gpu-dev/src/cli/kernellake query --sql "$sql" --stats

# Q15's own {data}/{supplier_data} each appear twice; CPU-backend only
# (see q15.sql's own header) -- use --backend cpu here, not --backend gpu.
sql=$(grep -v '^--' benchmarks/tpch/queries/q15.sql | tr '\n' ' ' | \
  sed "s|{data}|/tmp/kernellake-tpch-sf1/lineitem-*.parquet|g; \
       s|{supplier_data}|/tmp/kernellake-tpch-sf1/supplier-*.parquet|g")
./build/gpu-dev/src/cli/kernellake query --sql "$sql" --backend cpu --stats
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

# Q19/Q14 need --part-data, Q12 and Q4 need --orders-data, Q3 needs both
# --orders-data and --customer-data, Q10 needs --orders-data,
# --customer-data, and --nation-data, Q5 needs --orders-data,
# --customer-data, --supplier-data, --nation-data, and --region-data,
# Q7 needs --orders-data, --customer-data, --supplier-data, and
# --nation-data, Q8 needs --part-data, --supplier-data, --orders-data,
# --customer-data, --nation-data, and --region-data, Q9 needs
# --part-data, --supplier-data, --partsupp-data, --orders-data, and
# --nation-data, Q11 needs
# --partsupp-data, --supplier-data, and --nation-data (--data is still
# required but unused by Q11's own query text), Q13 needs
# --customer-data and --orders-data (--data likewise required but
# unused), Q15 needs --supplier-data (CPU-backend only -- pass
# --backend cpu, see q15.sql's own header), Q18 needs --customer-data
# and --orders-data (none covered by --query all, which only passes
# --data):
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
  --scale-factor 1 --query 4
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
python3 tools/validate_tpch.py \
  --kernellake build/gpu-dev/src/cli/kernellake \
  --data '/tmp/kernellake-tpch-sf1/lineitem-*.parquet' \
  --orders-data '/tmp/kernellake-tpch-sf1/orders-*.parquet' \
  --customer-data '/tmp/kernellake-tpch-sf1/customer-*.parquet' \
  --supplier-data '/tmp/kernellake-tpch-sf1/supplier-*.parquet' \
  --nation-data '/tmp/kernellake-tpch-sf1/nation-*.parquet' \
  --scale-factor 1 --query 7
python3 tools/validate_tpch.py \
  --kernellake build/gpu-dev/src/cli/kernellake \
  --data '/tmp/kernellake-tpch-sf1/lineitem-*.parquet' \
  --part-data '/tmp/kernellake-tpch-sf1/part-*.parquet' \
  --supplier-data '/tmp/kernellake-tpch-sf1/supplier-*.parquet' \
  --orders-data '/tmp/kernellake-tpch-sf1/orders-*.parquet' \
  --customer-data '/tmp/kernellake-tpch-sf1/customer-*.parquet' \
  --nation-data '/tmp/kernellake-tpch-sf1/nation-*.parquet' \
  --region-data '/tmp/kernellake-tpch-sf1/region-*.parquet' \
  --scale-factor 1 --query 8
python3 tools/validate_tpch.py \
  --kernellake build/gpu-dev/src/cli/kernellake \
  --data '/tmp/kernellake-tpch-sf1/lineitem-*.parquet' \
  --part-data '/tmp/kernellake-tpch-sf1/part-*.parquet' \
  --supplier-data '/tmp/kernellake-tpch-sf1/supplier-*.parquet' \
  --partsupp-data '/tmp/kernellake-tpch-sf1/partsupp-*.parquet' \
  --orders-data '/tmp/kernellake-tpch-sf1/orders-*.parquet' \
  --nation-data '/tmp/kernellake-tpch-sf1/nation-*.parquet' \
  --scale-factor 1 --query 9
python3 tools/validate_tpch.py \
  --kernellake build/gpu-dev/src/cli/kernellake \
  --data '/tmp/kernellake-tpch-sf1/lineitem-*.parquet' \
  --partsupp-data '/tmp/kernellake-tpch-sf1/partsupp-*.parquet' \
  --supplier-data '/tmp/kernellake-tpch-sf1/supplier-*.parquet' \
  --nation-data '/tmp/kernellake-tpch-sf1/nation-*.parquet' \
  --scale-factor 1 --query 11
python3 tools/validate_tpch.py \
  --kernellake build/gpu-dev/src/cli/kernellake \
  --data '/tmp/kernellake-tpch-sf1/lineitem-*.parquet' \
  --customer-data '/tmp/kernellake-tpch-sf1/customer-*.parquet' \
  --orders-data '/tmp/kernellake-tpch-sf1/orders-*.parquet' \
  --scale-factor 1 --query 13
# CPU-backend only -- see q15.sql's own header for why.
python3 tools/validate_tpch.py \
  --kernellake build/gpu-dev/src/cli/kernellake \
  --data '/tmp/kernellake-tpch-sf1/lineitem-*.parquet' \
  --supplier-data '/tmp/kernellake-tpch-sf1/supplier-*.parquet' \
  --scale-factor 1 --query 15 --backend cpu
python3 tools/validate_tpch.py \
  --kernellake build/gpu-dev/src/cli/kernellake \
  --data '/tmp/kernellake-tpch-sf1/lineitem-*.parquet' \
  --customer-data '/tmp/kernellake-tpch-sf1/customer-*.parquet' \
  --orders-data '/tmp/kernellake-tpch-sf1/orders-*.parquet' \
  --scale-factor 1 --query 18
```

This has been run at SF0.01, SF0.1, and SF1 (60,000, 600,000, and
6,000,000 generated rows) -- Q1 and Q6 matched DuckDB exactly at every
scale, including the full SF1 run (~105 MiB single Parquet file, zstd
compression, 1,000,000-row row groups). Q19, Q12, Q14, Q3, Q10, Q5, Q7,
Q8, Q9, Q11, and Q18 have each been verified at SF0.01 on both the CPU and
GPU backends, exact match against DuckDB (Q3 including its 3-way join,
Q4 including its correlated `EXISTS` subquery -- verified not just at
SF0.01 but stress-tested at SF10 (a ~60M-row `lineitem` build side,
orders split across 4 files), 20/20 clean runs and an exact DuckDB match
on real GPU hardware, since the correlated-subquery-as-semi-join
operator's first (`cudf::filtered_join`-based) implementation had a
real, scale-dependent crash that only showed up there -- see the "Hash
joins" section's own note on why it's built on `cudf::hash_join` instead;
`ORDER BY revenue DESC, o_orderdate` multi-key sort, and `LIMIT 10`; Q10
including its 4-way join, 7-column `GROUP BY` spanning two non-adjacent
join sources, and `LIMIT 20`; Q5 including its 6-way join and the
`c_nationkey = s_nationkey` `WHERE`-clause predicate spanning two
non-adjacent join sources; Q7 including its 6-way join with `nation`
joined twice under different aliases; Q8 including its 8-way join (also
`nation` joined twice) wrapped in a real derived table with a JOIN
inside its own `FROM` -- the shape that surfaced the `LogicalAggregate`
column-remap bug fixed alongside this query, see `docs/ARCHITECTURE.md`'s
"Derived tables" section; Q9 including its 6-way join and
the `ps_suppkey = l_suppkey` `WHERE`-clause half of `partsupp`'s
otherwise-inexpressible two-column join condition -- 175 real matching
rows at SF0.01, not an empty/trivial result; Q11 including its 3-way join,
`GROUP BY` + `HAVING`, and a real non-correlated scalar subquery computing
`HAVING`'s own threshold; Q18 including its 3-way join and a real
non-correlated `IN (SELECT ...)` subquery in `WHERE` -- DuckDB needs no
code changes for `HAVING`/`IN`-subqueries, it already supports both
natively). Q15 has been verified at SF0.01 on the **CPU backend only**
(exact DuckDB match, 0/20 repeated runs mismatched) -- its own `HAVING`
comparison against a subquery whose `FROM` is itself a derived table
(`HAVING total_revenue = (SELECT MAX(total_revenue) FROM (SELECT ...
GROUP BY ...) AS r2)`) is unreliable on the GPU backend specifically (a
real, investigated, not-yet-fixed cross-backend floating-point
limitation -- confirmed 14/20 GPU-backend runs mismatched -- see
`docs/ARCHITECTURE.md`'s "`HAVING` and scalar subqueries" section) --
Q19/Q12/Q14/Q3 also cross-validated against PySpark and
DuckDB via `tools/benchmark_three_way.py` (KernelLake via a persistent
`kernellake-server` over Arrow Flight SQL, PySpark, and DuckDB all agree
on every query, real GPU hardware, SF0.01/SF1/SF10 -- see
`docs/ROADMAP.md` for the full crossover numbers and a
`--cost-per-hour`-based cost-per-TB comparison);
Q10/Q5/Q7/Q8/Q9/Q11/Q15/Q18 have not been wired into
`benchmark_three_way.py` yet (see `docs/ROADMAP.md`'s "Not yet started").

## 4. Benchmark

`kernellake benchmark tpch` times a query over configurable warmup and
measured iterations, reporting each iteration plus median/mean/min/max/
standard deviation as JSON. Pass `--part-data` for Q19/Q14, `--orders-data`
for Q12, `--orders-data` and `--customer-data` for Q3, `--orders-data`,
`--customer-data`, and `--nation-data` for Q10, `--orders-data`,
`--customer-data`, `--supplier-data`, `--nation-data`, and `--region-data`
for Q5, `--orders-data`, `--customer-data`, `--supplier-data`, and
`--nation-data` for Q7, `--part-data`, `--supplier-data`,
`--orders-data`, `--customer-data`, `--nation-data`, and `--region-data`
for Q8, `--part-data`, `--supplier-data`,
`--partsupp-data`, `--orders-data`, and `--nation-data` for Q9,
`--partsupp-data`, `--supplier-data`, and `--nation-data` for Q11
(`--data` is still required but unused by Q11's own query text),
`--customer-data` and `--orders-data` for Q13 (`--data` likewise
required but unused), `--supplier-data` for Q15 (CPU-backend only --
see `q15.sql`'s own header), or `--customer-data` and `--orders-data`
for Q18 (or any combination for a future query needing those extra
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
