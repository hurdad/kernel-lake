-- TPC-H Q1 (unofficial, TPC-H-derived -- not a certified TPC result).
--
-- Deviations from canonical TPC-H Q1:
--   1. `FROM lineitem` -> `FROM read_parquet('{data}')`, same reason as q06.sql.
--   2. `DATE '1998-12-01' - INTERVAL '90' DAY` is written as the literal
--      `DATE '1998-09-02'` -- KernelLake's expression grammar has no
--      INTERVAL arithmetic yet.
--   3. `ORDER BY l_returnflag, l_linestatus` is omitted -- KernelLake has
--      no physical sort operator yet (LogicalSort throws PlanningError;
--      see docs/architecture.md). Per the spec's own validation
--      methodology, an unordered query's results should be compared after
--      deterministic key ordering rather than requiring the query itself
--      to sort, so this is a syntax accommodation, not a semantic change.
SELECT l_returnflag,
       l_linestatus,
       SUM(l_quantity) AS sum_qty,
       SUM(l_extendedprice) AS sum_base_price,
       SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
       SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
       AVG(l_quantity) AS avg_qty,
       AVG(l_extendedprice) AS avg_price,
       AVG(l_discount) AS avg_disc,
       COUNT(*) AS count_order
FROM read_parquet('{data}')
WHERE l_shipdate <= DATE '1998-09-02'
GROUP BY l_returnflag, l_linestatus
