"""Shared KernelLake-vs-DuckDB result comparison helpers.

Used by both validate_against_duckdb.py (general queries) and
validate_tpch.py (the TPC-H-derived query files in benchmarks/tpch/queries).
"""

import math
import subprocess
import tempfile
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.ipc as ipc


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


def values_match(a, b, rel_tol: float = 1e-9, abs_tol: float = 1e-6) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is None and b is None
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    return a == b


def rows_match(a: list, b: list, rel_tol: float = 1e-9, abs_tol: float = 1e-6) -> bool:
    if len(a) != len(b):
        return False
    for row_a, row_b in zip(a, b):
        if row_a.keys() != row_b.keys():
            return False
        if not all(values_match(row_a[k], row_b[k], rel_tol, abs_tol) for k in row_a):
            return False
    return True
