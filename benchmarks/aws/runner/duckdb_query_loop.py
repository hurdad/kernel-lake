#!/usr/bin/env python3
"""Standalone DuckDB TPC-H-derived query loop, meant to run directly on
terraform/duckdb_instance.tf's dedicated DuckDB host over SSH -- not
in-process on the orchestrator, unlike an earlier version of this harness
(see docs/RUNBOOK.md for why: no dedicated $/hour to attribute a query's
cost against otherwise). Reads the same real S3 data as
aws_benchmark_runner.py's KernelLake/Spark legs, via DuckDB's own
httpfs/aws extensions, and writes an independent results JSON that
reporting/generate_report.py merges in by query/mode alongside the
KernelLake/Spark results (see --duckdb-results there).

Deliberately self-contained -- duplicates a few small helpers from
tools/benchmark_three_way.py (load_query_text()/kernellake_sql()/
median_stats()) rather than importing them, since this runs on a bare
DuckDB host with no kernel-lake repo checkout: only this script plus a
sibling queries/ directory copied alongside it (see docs/RUNBOOK.md for
the exact scp invocation -- `scp -r benchmarks/tpch/queries
runner/duckdb_query_loop.py <duckdb-host>:~/`).

**No live cross-engine row validation against KernelLake here.** Unlike
aws_benchmark_runner.py's old in-process DuckDB path (which could compare
Arrow tables directly against the same-process KernelLake result via
duckdb_compare.py's rows_match()), this script runs on separate infra
with no access to KernelLake's live results. Each query/mode's row count
is recorded so a human can sanity-check it against the KernelLake/Spark
run's own row counts, but this is not the same guarantee as a real
row-level comparison -- a real gap opened up by giving DuckDB its own
dedicated box, not something silently dropped.

Usage (on the DuckDB host, after scp'ing this file + queries/ alongside it):
    python3 duckdb_query_loop.py --s3-bucket <bucket> --scale-factor 100 \\
        --query all --iterations 2 --modes cold,warm \\
        --output duckdb-results.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

QUERIES_DIR = Path(__file__).resolve().parent / "queries"

ALL_QUERIES = (1, 3, 6, 12, 14, 19)
QUERIES_WITH_SECOND_TABLE = {19: "part_data", 14: "part_data", 12: "orders_data", 3: "orders_data"}
QUERIES_WITH_THIRD_TABLE = {3: "customer_data"}


def load_query_text(query_number: int) -> str:
    path = QUERIES_DIR / f"q{query_number:02d}.sql"
    if not path.exists():
        raise FileNotFoundError(f"no query file for Q{query_number}: {path} -- scp queries/ alongside this script?")
    return re.sub(r"--[^\n]*\n", "\n", path.read_text())


def kernellake_sql(
    query_number: int,
    data_glob: str,
    part_data_glob: str | None = None,
    orders_data_glob: str | None = None,
    customer_data_glob: str | None = None,
) -> str:
    text = load_query_text(query_number)
    if "{part_data}" in text and not part_data_glob:
        raise ValueError(f"Q{query_number} needs a second table -- part_data glob missing")
    if "{orders_data}" in text and not orders_data_glob:
        raise ValueError(f"Q{query_number} needs a second table -- orders_data glob missing")
    if "{customer_data}" in text and not customer_data_glob:
        raise ValueError(f"Q{query_number} needs a third table -- customer_data glob missing")
    text = text.replace("{data}", data_glob)
    if part_data_glob:
        text = text.replace("{part_data}", part_data_glob)
    if orders_data_glob:
        text = text.replace("{orders_data}", orders_data_glob)
    if customer_data_glob:
        text = text.replace("{customer_data}", customer_data_glob)
    return text.strip()


def s3_data_glob(bucket: str, scale_factor: int, table: str, compression: str = "snappy") -> str:
    # Mirrors aws_benchmark_runner.py's s3_data_glob() (s3:// scheme only
    # -- no Spark s3a:// concern here, DuckDB isn't Spark).
    prefix_map = {
        "lineitem": "lineitem-*.parquet",
        "part": "part-00000.parquet",
        "orders": "orders-*.parquet",
        "customer": "customer-00000.parquet",
    }
    return f"s3://{bucket}/tpch-data/sf{scale_factor}-{compression}/{prefix_map[table]}"


def median_stats(samples: list) -> dict:
    return {
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.mean(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
    }


def new_duckdb_connection(region: str):
    """httpfs for s3:// reads, plus the aws extension's CREDENTIAL_CHAIN
    provider so it resolves this host's own IAM instance profile the same
    way boto3/the AWS SDK would -- consistent with how kernellake-server/
    Spark authenticate to S3 elsewhere in this harness."""
    import duckdb

    con = duckdb.connect()
    con.install_extension("httpfs")
    con.load_extension("httpfs")
    con.install_extension("aws")
    con.load_extension("aws")
    con.sql("CREATE OR REPLACE SECRET (TYPE S3, PROVIDER CREDENTIAL_CHAIN)")
    con.sql(f"SET s3_region = '{region}'")
    return con


def run_duckdb_query(con, query_number: int, globs: dict) -> tuple:
    sql = kernellake_sql(
        query_number, globs["data"], globs.get("part_data"), globs.get("orders_data"), globs.get("customer_data")
    )
    start = time.perf_counter()
    table = con.sql(sql).arrow().read_all()
    elapsed = time.perf_counter() - start
    return table, elapsed


def build_globs(bucket: str, scale_factor: int, query_number: int, compression: str) -> dict:
    globs = {"data": s3_data_glob(bucket, scale_factor, "lineitem", compression)}
    table_for_key = {"part_data": "part", "orders_data": "orders", "customer_data": "customer"}
    if query_number in QUERIES_WITH_SECOND_TABLE:
        key = QUERIES_WITH_SECOND_TABLE[query_number]
        globs[key] = s3_data_glob(bucket, scale_factor, table_for_key[key], compression)
    if query_number in QUERIES_WITH_THIRD_TABLE:
        key = QUERIES_WITH_THIRD_TABLE[query_number]
        globs[key] = s3_data_glob(bucket, scale_factor, table_for_key[key], compression)
    return globs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s3-bucket", required=True)
    parser.add_argument("--scale-factor", type=int, required=True)
    parser.add_argument("--compression", default="snappy")
    parser.add_argument("--query", default="all", help="Query number, or 'all' for every supported query")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--modes", default="cold,warm", help="Comma-separated: cold,warm (see module docstring)")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    queries = list(ALL_QUERIES) if args.query == "all" else [int(args.query)]
    modes = tuple(args.modes.split(","))

    con = new_duckdb_connection(args.region)
    results = []
    for query_number in queries:
        globs = build_globs(args.s3_bucket, args.scale_factor, query_number, args.compression)
        entry: dict = {"query": query_number, "modes": {}}
        for mode in modes:
            samples = []
            row_count = None
            for rep in range(args.iterations):
                if mode == "cold" and rep == 0:
                    con.close()
                    con = new_duckdb_connection(args.region)
                table, elapsed = run_duckdb_query(con, query_number, globs)
                samples.append(elapsed)
                row_count = table.num_rows
            stats = median_stats(samples)
            entry["modes"][mode] = {**stats, "row_count": row_count}
            print(f"Q{query_number} ({mode}): median {stats['median_seconds']:.3f}s, {row_count} rows", file=sys.stderr)
        results.append(entry)

    output = {
        "engine": "duckdb",
        "s3_bucket": args.s3_bucket,
        "scale_factor": args.scale_factor,
        "compression": args.compression,
        "queries": results,
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
