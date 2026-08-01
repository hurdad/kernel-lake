#!/usr/bin/env python3
"""Cross-validate KernelLake query results against DuckDB.

The spec's acceptance criteria require general-query results to match
DuckDB, used here as the correctness oracle. Each test query below is valid
SQL in both engines unchanged (read_parquet(...), DATE literals, GROUP BY,
and the aggregate functions KernelLake supports are all DuckDB-compatible),
so the same query string is run against both and the results compared.

Usage:
    python3 tools/validate_against_duckdb.py \
        --kernellake build/gpu-dev/src/cli/kernellake \
        --data '/tmp/kernellake-sales/*.parquet'

Requires the `duckdb` and `pyarrow` Python packages and a GPU-enabled
`kernellake` build (query execution has no CPU fallback -- see
docs/architecture.md). See validate_tpch.py for the TPC-H-derived variant
of this same check.
"""

import argparse
import sys

from duckdb_compare import normalize, rows_match, run_duckdb, run_kernellake

QUERIES = [
    "SELECT region, SUM(amount) AS total FROM read_parquet('{data}') "
    "WHERE event_date >= DATE '2026-01-01' GROUP BY region",
    "SELECT category, SUM(amount) AS total, COUNT(*) AS n FROM read_parquet('{data}') GROUP BY category",
    "SELECT SUM(amount) AS total, COUNT(*) AS n, AVG(amount) AS avg_amount, "
    "MIN(amount) AS min_amount, MAX(amount) AS max_amount FROM read_parquet('{data}')",
    "SELECT region FROM read_parquet('{data}') WHERE amount > 500.0",
    "SELECT region, COUNT(discount) AS non_null_discounts FROM read_parquet('{data}') GROUP BY region",
    "SELECT customer_id, SUM(amount) AS total FROM read_parquet('{data}') "
    "WHERE region = 'region-0' GROUP BY customer_id",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernellake", required=True, help="Path to the kernellake CLI binary")
    parser.add_argument("--data", required=True, help="Parquet glob passed to read_parquet(...)")
    args = parser.parse_args()

    failures = 0
    for template in QUERIES:
        sql = template.format(data=args.data)
        print(f"--- {sql}")
        try:
            kernellake_rows = normalize(run_kernellake(args.kernellake, sql))
            duckdb_rows = normalize(run_duckdb(sql))
        except Exception as exc:  # noqa: BLE001 -- report and continue to the next query
            print(f"    ERROR: {exc}")
            failures += 1
            continue

        if rows_match(kernellake_rows, duckdb_rows):
            print(f"    PASS ({len(kernellake_rows)} rows)")
        else:
            print(f"    FAIL: kernellake={len(kernellake_rows)} rows, duckdb={len(duckdb_rows)} rows")
            print(f"    kernellake sample: {kernellake_rows[:3]}")
            print(f"    duckdb sample:     {duckdb_rows[:3]}")
            failures += 1

    print()
    if failures:
        print(f"{failures}/{len(QUERIES)} queries FAILED to match DuckDB")
        return 1
    print(f"all {len(QUERIES)} queries matched DuckDB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
