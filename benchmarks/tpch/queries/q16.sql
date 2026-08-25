-- TPC-H Q16 (unofficial, TPC-H-derived -- not a certified TPC result).
--
-- The first query needing COUNT(DISTINCT ...) -- see
-- docs/ARCHITECTURE.md's "COUNT(DISTINCT ...)" section for the full
-- scope and the GPU backend's own restriction: it cannot be combined
-- with another aggregate in the same GROUP BY (the CPU/Acero backend has
-- no such restriction). Q16 itself only ever needs the one aggregate, so
-- this doesn't limit anything about the query below.
--
-- Deviations from canonical TPC-H Q16:
--   1. `FROM partsupp, part WHERE p_partkey = ps_partkey` -> a `JOIN ...
--      ON` (comma-style joins aren't supported). {partsupp_data}/
--      {part_data} are substituted by the caller with globs over the
--      partsupp/part Parquet files generate_tpch.py produced.
--   2. `[BRAND]`/`[TYPE]`/`[SIZE1]`..`[SIZE8]` substitution parameters
--      resolved to representative literal values (`Brand#45`, `MEDIUM
--      POLISHED%`, `49, 14, 23, 45, 19, 3, 36, 9`) -- canonical TPC-H
--      itself defines these as randomly-chosen-per-run substitution
--      parameters, not fixed literals, so any concrete choice is a
--      deviation from the spec's own parameter-generation procedure, the
--      same kind of fixed-literal simplification every other
--      parameterized query in this project's suite already makes.
-- No other semantic change: the NOT IN (SELECT ...) subquery (identical
-- shape to Q18's own NOT IN, just negated the other way -- see Q18's own
-- header for that machinery), the 3-column GROUP BY, and the
-- `ORDER BY supplier_cnt DESC, p_brand, p_type, p_size` multi-key sort
-- all work exactly as canonical TPC-H writes them.
SELECT p_brand,
       p_type,
       p_size,
       COUNT(DISTINCT ps_suppkey) AS supplier_cnt
FROM read_parquet('{partsupp_data}') AS ps
JOIN read_parquet('{part_data}') AS p ON p.p_partkey = ps.ps_partkey
WHERE p_brand <> 'Brand#45'
  AND p_type NOT LIKE 'MEDIUM POLISHED%'
  AND p_size IN (49, 14, 23, 45, 19, 3, 36, 9)
  AND ps_suppkey NOT IN (
    SELECT s_suppkey
    FROM read_parquet('{supplier_data}') AS s
    WHERE s_comment LIKE '%Customer%Complaints%'
  )
GROUP BY p_brand, p_type, p_size
ORDER BY supplier_cnt DESC, p_brand, p_type, p_size
