-- TPC-H Q19 (unofficial, TPC-H-derived -- not a certified TPC result).
--
-- Deviations from canonical TPC-H Q19:
--   1. `FROM lineitem, part` -> `FROM read_parquet('{data}') AS l JOIN
--      read_parquet('{part_data}') AS p ON l.l_partkey = p.p_partkey`
--      (comma-style joins are not supported -- see docs/ARCHITECTURE.md's
--      "read_parquet(...) adapter" section -- and KernelLake has no
--      catalog/table-name resolution yet). {data}/{part_data} are
--      substituted by the caller with globs over the lineitem/part
--      Parquet files generate_tpch.py produced.
--   2. The join predicate (`p_partkey = l_partkey`) is factored out of the
--      three `OR`-ed branches into the `JOIN ... ON` clause -- semantically
--      identical (it is common to every branch in the original), but
--      necessary since KernelLake's grammar only accepts a single
--      equality-key `ON` clause for a two-table `JOIN`, not an arbitrary
--      `WHERE`-clause join condition.
--   3. `p_container in (...)` is narrowed from 4 literals per brand down to
--      a single representative container per branch, since
--      `tools/generate_tpch.py`'s `CONTAINERS` list is a representative
--      subset of TPC-H's own container domain, not the full cross product
--      -- see that script's module docstring.
SELECT SUM(l_extendedprice * (1 - l_discount)) AS revenue
FROM read_parquet('{data}') AS l JOIN read_parquet('{part_data}') AS p ON l.l_partkey = p.p_partkey
WHERE (p_brand = 'Brand#12' AND p_container = 'SM CASE' AND l_quantity >= 1 AND l_quantity <= 11
       AND p_size BETWEEN 1 AND 5 AND l_shipmode = 'AIR' AND l_shipinstruct = 'DELIVER IN PERSON')
   OR (p_brand = 'Brand#23' AND p_container = 'MED BAG' AND l_quantity >= 10 AND l_quantity <= 20
       AND p_size BETWEEN 1 AND 10 AND l_shipmode = 'AIR' AND l_shipinstruct = 'DELIVER IN PERSON')
   OR (p_brand = 'Brand#34' AND p_container = 'LG CASE' AND l_quantity >= 20 AND l_quantity <= 30
       AND p_size BETWEEN 1 AND 15 AND l_shipmode = 'AIR' AND l_shipinstruct = 'DELIVER IN PERSON')
