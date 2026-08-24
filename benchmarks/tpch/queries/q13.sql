-- TPC-H Q13 (unofficial, TPC-H-derived -- not a certified TPC result).
--
-- The first query in this project needing a LEFT OUTER JOIN, a JOIN ON
-- clause combining the required equality key with an extra predicate
-- (`o_comment NOT LIKE '%special%requests%'`), and a derived table
-- (`FROM (SELECT ...) AS c_orders`) all together. See
-- docs/ARCHITECTURE.md's "Hash joins" section for the LEFT OUTER JOIN/
-- ON-clause-predicate scope (the extra predicate must reference only the
-- newly-joined side -- true here, it's purely a function of `orders`) and
-- its "Derived tables" section for the FROM-subquery scope (a single
-- derived table as this query's entire FROM clause, itself not joined).
--
-- Deviations from canonical TPC-H Q13: none semantic. {customer_data}/
-- {orders_data} are substituted by the caller with globs over the
-- customer/orders Parquet files generate_tpch.py produced. This query
-- never references lineitem at all, but the benchmark CLI's own file
-- discovery still requires a valid `--data` glob to be passed (any
-- existing Parquet glob works, e.g. point it at the same orders files via
-- `--data`) -- a pre-existing tool limitation, not a query-specific one.
SELECT c_count,
       COUNT(*) AS custdist
FROM (
    SELECT c.c_custkey AS c_custkey,
           COUNT(o.o_orderkey) AS c_count
    FROM read_parquet('{customer_data}') AS c
    LEFT JOIN read_parquet('{orders_data}') AS o
      ON c.c_custkey = o.o_custkey
      AND o.o_comment NOT LIKE '%special%requests%'
    GROUP BY c.c_custkey
) AS c_orders
GROUP BY c_count
ORDER BY custdist DESC, c_count DESC
