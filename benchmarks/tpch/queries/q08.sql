-- TPC-H Q8 (unofficial, TPC-H-derived -- not a certified TPC result).
--
-- Deviations from canonical TPC-H Q8: none semantic.
--   1. `FROM part, supplier, lineitem, orders, customer, nation n1,
--      nation n2, region WHERE p_partkey = l_partkey AND (...)` -> an
--      8-way `JOIN ... ON` chain (comma-style joins are not supported --
--      see docs/ARCHITECTURE.md's "read_parquet(...) adapter" section).
--      `nation` is joined twice, once per alias (`n1` for the customer's
--      own nation/region, `n2` for the supplier's), the same pattern
--      Q7 already uses. {data}/{part_data}/{supplier_data}/
--      {orders_data}/{customer_data}/{nation_data}/{region_data} are
--      substituted by the caller with globs over the lineitem/part/
--      supplier/orders/customer/nation/region Parquet files
--      generate_tpch.py produced -- {nation_data} is substituted into
--      *both* nation JOIN steps, both reading the same real table.
--   2. The derived table (`FROM (SELECT ...) AS all_nations`) is kept
--      exactly as canonical TPC-H writes it, unlike Q7's own deviation
--      (which flattened an equivalent derived table away, since KernelLake
--      had no derived-table support at the time). This is the first query
--      combining a derived table with a JOIN *inside* that derived
--      table's own FROM -- a shape a real bug was found and fixed under
--      while adding this query: an outer GROUP BY sitting on a derived
--      table whose own inner query is a plain (non-aggregate) JOIN
--      projection could resolve its own GROUP BY key against the wrong
--      (pre-projection) schema, an out-of-range column access at
--      execution time. See `docs/ARCHITECTURE.md`'s "Derived tables"
--      section and `physical_planner.cpp`'s `LogicalAggregate` case for
--      the fix.
--   3. `o_orderdate between date '1995-01-01' and date '1996-12-31'`
--      stays exactly as canonical TPC-H writes it (`BETWEEN`, already
--      supported).
SELECT o_year,
       SUM(CASE WHEN nation = 'BRAZIL' THEN volume ELSE 0 END) / SUM(volume) AS mkt_share
FROM (
    SELECT EXTRACT(YEAR FROM o.o_orderdate) AS o_year,
           l.l_extendedprice * (1 - l.l_discount) AS volume,
           n2.n_name AS nation
    FROM read_parquet('{part_data}') AS p
    JOIN read_parquet('{data}') AS l ON p.p_partkey = l.l_partkey
    JOIN read_parquet('{supplier_data}') AS s ON s.s_suppkey = l.l_suppkey
    JOIN read_parquet('{orders_data}') AS o ON l.l_orderkey = o.o_orderkey
    JOIN read_parquet('{customer_data}') AS c ON o.o_custkey = c.c_custkey
    JOIN read_parquet('{nation_data}') AS n1 ON c.c_nationkey = n1.n_nationkey
    JOIN read_parquet('{region_data}') AS r ON n1.n_regionkey = r.r_regionkey
    JOIN read_parquet('{nation_data}') AS n2 ON s.s_nationkey = n2.n_nationkey
    WHERE r.r_name = 'AMERICA'
      AND o.o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
      AND p.p_type = 'ECONOMY ANODIZED STEEL'
) AS all_nations
GROUP BY o_year
ORDER BY o_year
