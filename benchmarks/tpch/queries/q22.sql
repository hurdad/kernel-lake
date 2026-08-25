-- TPC-H Q22 (unofficial, TPC-H-derived -- not a certified TPC result).
--
-- The first query needing SUBSTRING and a non-correlated scalar subquery
-- as a bare WHERE comparison operand (not inside IN, and not HAVING --
-- see docs/ARCHITECTURE.md's "COUNT(DISTINCT ...)"-adjacent
-- "`SUBSTRING`" section and its own note on resolve_subqueries() now also
-- running over WHERE, not just HAVING).
--
-- Deviations from canonical TPC-H Q22:
--   1. `SUBSTRING(c_phone FROM 1 FOR 2)` (SQL-92's own FROM/FOR syntax) ->
--      `SUBSTRING(c_phone, 1, 2)` (function-call form) -- hsql (this
--      project's SQL-92 parser) has no dedicated FROM/FOR substring
--      grammar rule; see src/sql/parser.cpp's own SUBSTRING handling.
--   2. `[I1]`..`[I7]` substitution parameters (country codes) resolved to
--      representative literal values (`13, 31, 23, 29, 30, 18, 17`) --
--      canonical TPC-H itself defines these as randomly-chosen-per-run
--      substitution parameters, not fixed literals, the same kind of
--      deviation every other parameterized query in this project's suite
--      already makes (see e.g. Q2/Q16's own header comments).
--   3. `create view` isn't used (no `CREATE VIEW`/`WITH` support) --
--      N/A here; canonical Q22 doesn't use one either.
--   4. The derived table's own inner `FROM customer` needs an explicit
--      alias (`AS c`) for `NOT EXISTS`'s correlation
--      (`o_custkey = c.c_custkey`) to resolve against -- canonical TPC-H
--      leaves it unaliased since real SQL lets an unqualified column
--      reference resolve to the enclosing scope implicitly; this
--      project's `EXISTS`/`NOT EXISTS` rewrite requires the correlated
--      side's own alias explicitly (see sql::rewrite_exists_subqueries()).
-- No other semantic change: the `IN (...)` literal list (twice, once in
-- the derived table's own filter and once in the correlated-average
-- subquery's identical filter), the derived table with its own `GROUP BY`-
-- free `SELECT`/`WHERE` feeding an outer `GROUP BY`/aggregate (the Q8/Q13/
-- Q15 pattern), and `ORDER BY cntrycode` all work exactly as canonical
-- TPC-H writes them.
SELECT cntrycode,
       COUNT(*) AS numcust,
       SUM(c_acctbal) AS totacctbal
FROM (
    SELECT SUBSTRING(c_phone, 1, 2) AS cntrycode, c_acctbal
    FROM read_parquet('{customer_data}') AS c
    WHERE SUBSTRING(c_phone, 1, 2) IN ('13', '31', '23', '29', '30', '18', '17')
      AND c_acctbal > (
          SELECT AVG(c_acctbal)
          FROM read_parquet('{customer_data}') AS c2
          WHERE c_acctbal > 0.00
            AND SUBSTRING(c_phone, 1, 2) IN ('13', '31', '23', '29', '30', '18', '17')
      )
      AND NOT EXISTS (
          SELECT * FROM read_parquet('{orders_data}') AS o
          WHERE o.o_custkey = c.c_custkey
      )
) AS custsale
GROUP BY cntrycode
ORDER BY cntrycode
