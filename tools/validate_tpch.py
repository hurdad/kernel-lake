#!/usr/bin/env python3
"""Validate KernelLake's TPC-H-derived queries (benchmarks/tpch/queries/) against DuckDB.

Unofficial TPC-H-derived benchmark. Not a certified TPC result.

Mirrors the spec's `kernellake validate tpch` CLI surface as a Python tool
rather than a C++ subcommand, matching the choice already made for
validate_against_duckdb.py: embedding libduckdb directly into the
KernelLake binary is a real option (see docs/ARCHITECTURE.md) but hasn't
been done, so DuckDB cross-validation lives here instead.

Usage:
    python3 tools/validate_tpch.py \
        --kernellake build/gpu-dev/src/cli/kernellake \
        --data '/tmp/kernellake-tpch-sf1/*.parquet' \
        --query 6
    python3 tools/validate_tpch.py --kernellake ... --data ... --query all
    # Q19 needs a second table:
    python3 tools/validate_tpch.py --kernellake ... --data '.../lineitem-*.parquet' \
        --part-data '.../part-*.parquet' --query 19
    # Q12 needs a second table too (orders, not part):
    python3 tools/validate_tpch.py --kernellake ... --data '.../lineitem-*.parquet' \
        --orders-data '.../orders-*.parquet' --query 12
    # Q3 needs two extra tables (orders and customer):
    python3 tools/validate_tpch.py --kernellake ... --data '.../lineitem-*.parquet' \
        --orders-data '.../orders-*.parquet' --customer-data '.../customer-*.parquet' --query 3
    # Q10 needs three extra tables (orders, customer, and nation):
    python3 tools/validate_tpch.py --kernellake ... --data '.../lineitem-*.parquet' \
        --orders-data '.../orders-*.parquet' --customer-data '.../customer-*.parquet' \
        --nation-data '.../nation-*.parquet' --query 10
"""

import argparse
import re
import sys
from pathlib import Path

from duckdb_compare import normalize, rows_match, run_duckdb, run_kernellake

QUERIES_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "tpch" / "queries"


def load_query(
    query_number: int,
    data_glob: str,
    part_data_glob: str | None,
    orders_data_glob: str | None,
    customer_data_glob: str | None,
    nation_data_glob: str | None,
) -> str:
    path = QUERIES_DIR / f"q{query_number:02d}.sql"
    if not path.exists():
        raise FileNotFoundError(f"no query file for Q{query_number}: {path}")
    text = path.read_text()
    text = re.sub(r"--[^\n]*\n", "\n", text)  # strip line comments
    if "{part_data}" in text and not part_data_glob:
        raise ValueError(f"Q{query_number} needs a second table -- pass --part-data")
    if "{orders_data}" in text and not orders_data_glob:
        raise ValueError(f"Q{query_number} needs a second table -- pass --orders-data")
    if "{customer_data}" in text and not customer_data_glob:
        raise ValueError(f"Q{query_number} needs a third table -- pass --customer-data")
    if "{nation_data}" in text and not nation_data_glob:
        raise ValueError(f"Q{query_number} needs a fourth table -- pass --nation-data")
    text = text.replace("{data}", data_glob)
    if part_data_glob:
        text = text.replace("{part_data}", part_data_glob)
    if orders_data_glob:
        text = text.replace("{orders_data}", orders_data_glob)
    if customer_data_glob:
        text = text.replace("{customer_data}", customer_data_glob)
    if nation_data_glob:
        text = text.replace("{nation_data}", nation_data_glob)
    return text.strip()


def validate_one(
    kernellake_bin: str,
    query_number: int,
    data_glob: str,
    part_data_glob: str | None,
    orders_data_glob: str | None,
    customer_data_glob: str | None,
    nation_data_glob: str | None,
    backend: str | None,
) -> bool:
    try:
        sql = load_query(
            query_number, data_glob, part_data_glob, orders_data_glob, customer_data_glob, nation_data_glob
        )
        print(f"--- Q{query_number}: {sql.splitlines()[0]}...")
        kernellake_rows = normalize(run_kernellake(kernellake_bin, sql, backend))
        duckdb_rows = normalize(run_duckdb(sql))
    except Exception as exc:  # noqa: BLE001 -- report and let the caller count it as a failure
        print(f"--- Q{query_number}: ERROR: {exc}")
        return False

    if rows_match(kernellake_rows, duckdb_rows):
        print(f"    PASS ({len(kernellake_rows)} rows)")
        return True

    print(f"    FAIL: kernellake={len(kernellake_rows)} rows, duckdb={len(duckdb_rows)} rows")
    print(f"    kernellake sample: {kernellake_rows[:3]}")
    print(f"    duckdb sample:     {duckdb_rows[:3]}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kernellake", required=True, help="Path to the kernellake CLI binary")
    parser.add_argument("--data", required=True, help="Parquet glob passed to read_parquet(...)")
    parser.add_argument("--part-data", default=None, help="Parquet glob for the 'part' table (Q19 needs this)")
    parser.add_argument("--orders-data", default=None, help="Parquet glob for the 'orders' table (Q12 needs this)")
    parser.add_argument(
        "--customer-data", default=None, help="Parquet glob for the 'customer' table (Q3 needs this)"
    )
    parser.add_argument(
        "--nation-data", default=None, help="Parquet glob for the 'nation' table (Q10 needs this)"
    )
    parser.add_argument("--scale-factor", type=float, default=None, help="Informational only, for the report")
    parser.add_argument("--query", required=True, help="Query number (e.g. 6) or 'all'")
    parser.add_argument("--baseline", default="duckdb", choices=["duckdb"])
    parser.add_argument("--backend", default=None, choices=["cpu", "gpu"],
                       help="kernellake --backend; omit to use the binary's own default")
    args = parser.parse_args()

    available = sorted(int(p.stem[1:]) for p in QUERIES_DIR.glob("q*.sql"))
    query_numbers = available if args.query == "all" else [int(args.query)]

    print("Unofficial TPC-H-derived benchmark. Not a certified TPC result.")
    if args.scale_factor is not None:
        print(f"scale_factor={args.scale_factor}")
    print()

    failures = 0
    for query_number in query_numbers:
        if not validate_one(
            args.kernellake,
            query_number,
            args.data,
            args.part_data,
            args.orders_data,
            args.customer_data,
            args.nation_data,
            args.backend,
        ):
            failures += 1

    print()
    if failures:
        print(f"{failures}/{len(query_numbers)} TPC-H queries FAILED to match {args.baseline}")
        return 1
    print(f"all {len(query_numbers)} TPC-H queries matched {args.baseline}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
