-- TPC-H Q6 (unofficial, TPC-H-derived -- not a certified TPC result).
--
-- Deviation from canonical TPC-H Q6: `FROM lineitem` is replaced with
-- `FROM read_parquet('{data}')` (KernelLake has no catalog/table-name
-- resolution yet -- see docs/architecture.md's "read_parquet(...) adapter"
-- section), where {data} is substituted by the caller with a glob over the
-- lineitem Parquet files generate_tpch.py produced. No other semantic
-- change.
SELECT SUM(l_extendedprice * l_discount) AS revenue
FROM read_parquet('{data}')
WHERE l_shipdate >= DATE '1994-01-01'
  AND l_shipdate < DATE '1995-01-01'
  AND l_discount BETWEEN 0.05 AND 0.07
  AND l_quantity < 24
