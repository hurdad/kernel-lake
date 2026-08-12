-- TPC-H Q7 (unofficial, TPC-H-derived -- not a certified TPC result).
--
-- Deviations from canonical TPC-H Q7:
--   1. The canonical query wraps its 6-way join in a derived table
--      (`FROM (SELECT ... ) AS shipping GROUP BY ...`) purely to compute
--      `l_year`/`volume` once before grouping/aggregating over them --
--      KernelLake doesn't support subqueries yet, but nothing here
--      actually needs one: the derived table adds no filtering of its
--      own, so it flattens losslessly into one query, `EXTRACT`/the
--      arithmetic expression evaluated directly in the outer `SELECT`/
--      `GROUP BY` list -- the same "computed GROUP BY key" shape other
--      queries in this project already use.
--   2. `FROM supplier, lineitem, orders, customer, nation n1, nation n2
--      WHERE s_suppkey = l_suppkey AND o_orderkey = l_orderkey AND
--      c_custkey = o_custkey AND s_nationkey = n1.n_nationkey AND
--      c_nationkey = n2.n_nationkey AND (...)` -> a 6-way `JOIN ... ON`
--      chain (comma-style joins are not supported -- see
--      docs/ARCHITECTURE.md's "read_parquet(...) adapter" section).
--      `nation` is joined twice, once per alias (`n1`/`n2`) -- two
--      separate `read_parquet('{nation_data}')` sources, each its own
--      JOIN step; confirmed this resolves and executes correctly (no
--      special-casing needed: each alias is just another JOIN source,
--      the binder already disambiguates same-named columns from
--      different sides by position, not by name -- see
--      docs/ARCHITECTURE.md's "HashJoinNode" notes). {data}/
--      {orders_data}/{customer_data}/{supplier_data}/{nation_data} are
--      substituted by the caller with globs over the lineitem/orders/
--      customer/supplier/nation Parquet files generate_tpch.py produced
--      -- {nation_data} is substituted into *both* nation JOIN steps,
--      both reading the same real table.
--   3. `(n1.n_name = 'FRANCE' and n2.n_name = 'GERMANY') or (n1.n_name =
--      'GERMANY' and n2.n_name = 'FRANCE')` and `l_shipdate between date
--      '1995-01-01' and date '1996-12-31'` stay in `WHERE` exactly as
--      canonical TPC-H writes them (`OR` of `AND`s, `BETWEEN`, both
--      already supported -- Q19 already exercises the same `OR`-of-`AND`s
--      shape).
-- No other semantic change: the 3-column `GROUP BY` and multi-key
-- `ORDER BY supp_nation, cust_nation, l_year` (all ascending) work
-- exactly as canonical TPC-H writes them.
SELECT n1.n_name AS supp_nation,
       n2.n_name AS cust_nation,
       EXTRACT(YEAR FROM l_shipdate) AS l_year,
       SUM(l_extendedprice * (1 - l_discount)) AS revenue
FROM read_parquet('{supplier_data}') AS s
JOIN read_parquet('{data}') AS l ON s.s_suppkey = l.l_suppkey
JOIN read_parquet('{orders_data}') AS o ON o.o_orderkey = l.l_orderkey
JOIN read_parquet('{customer_data}') AS c ON c.c_custkey = o.o_custkey
JOIN read_parquet('{nation_data}') AS n1 ON s.s_nationkey = n1.n_nationkey
JOIN read_parquet('{nation_data}') AS n2 ON c.c_nationkey = n2.n_nationkey
WHERE ((n1.n_name = 'FRANCE' AND n2.n_name = 'GERMANY')
       OR (n1.n_name = 'GERMANY' AND n2.n_name = 'FRANCE'))
  AND l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
GROUP BY supp_nation, cust_nation, l_year
ORDER BY supp_nation, cust_nation, l_year
