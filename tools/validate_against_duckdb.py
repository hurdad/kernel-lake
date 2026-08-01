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
docs/architecture.md).
"""

import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.ipc as ipc

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


def run_kernellake(kernellake_bin: str, sql: str) -> pa.Table:
    with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as tmp:
        output_path = tmp.name
    try:
        result = subprocess.run(
            [kernellake_bin, "query", "--sql", sql, "--format", "arrow", "--output", output_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"kernellake query failed: {result.stderr}")
        with pa.memory_map(output_path, "rb") as source:
            return ipc.open_file(source).read_all()
    finally:
        Path(output_path).unlink(missing_ok=True)


def run_duckdb(sql: str) -> pa.Table:
    return duckdb.sql(sql).arrow().read_all()


def normalize(table: pa.Table) -> list:
    columns = sorted(table.column_names)
    table = table.select(columns)
    rows = table.to_pylist()
    return sorted(rows, key=lambda row: [str(row[c]) for c in columns])


def values_match(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is None and b is None
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6)
    return a == b


def rows_match(a: list, b: list) -> bool:
    if len(a) != len(b):
        return False
    for row_a, row_b in zip(a, b):
        if row_a.keys() != row_b.keys():
            return False
        if not all(values_match(row_a[k], row_b[k]) for k in row_a):
            return False
    return True


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
