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


def compression_tag(compression: str, compression_level: int | None = None) -> str:
    # Mirrors aws_benchmark_runner.py's compression_tag() exactly -- see
    # its own docstring for why every choice (snappy included) gets a
    # fully-tagged prefix, and why zstd's un-leveled "zstd" tag differs
    # from an explicit-level "zstd-l<N>" one even at level 1.
    if compression_level is not None and compression != "zstd":
        raise ValueError("compression_level only applies to compression='zstd'")
    if compression == "zstd" and compression_level is None:
        return "zstd"
    if compression == "zstd":
        return f"zstd-l{compression_level}"
    return compression


def s3_data_glob(
    bucket: str, scale_factor: int, table: str, compression: str = "snappy",
    compression_level: int | None = None,
) -> str:
    # Mirrors aws_benchmark_runner.py's s3_data_glob() (s3:// scheme only
    # -- no Spark s3a:// concern here, DuckDB isn't Spark).
    prefix_map = {
        "lineitem": "lineitem-*.parquet",
        "part": "part-00000.parquet",
        "orders": "orders-*.parquet",
        "customer": "customer-00000.parquet",
    }
    tag = compression_tag(compression, compression_level)
    sf_dir = f"sf{scale_factor}-{tag}"
    return f"s3://{bucket}/tpch-data/{sf_dir}/{prefix_map[table]}"


def median_stats(samples: list) -> dict:
    return {
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.mean(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
    }


def new_duckdb_connection(region: str, enable_cache: bool = False):
    """httpfs for s3:// reads, plus the aws extension's CREDENTIAL_CHAIN
    provider so it resolves this host's own IAM instance profile the same
    way boto3/the AWS SDK would -- consistent with how kernellake-server/
    Spark authenticate to S3 elsewhere in this harness.

    enable_cache defaults False for this script's own worst-case/no-cache
    cold benchmarking (2026-08-15, see below) -- duckdb_scaling_test.py
    passes True instead: its concurrency test fires the *same* query
    repeatedly against one warm connection for the whole test duration, so
    disabling caching there would measure "does concurrent cold S3 access
    scale" (a real but different, and already-confounded-by-shared-
    bandwidth-contention, question -- see
    docs/CONCURRENCY_HARNESS_DESIGN.md) instead of the intended "does
    concurrent execution scale" question."""
    import duckdb

    con = duckdb.connect()
    con.install_extension("httpfs")
    con.load_extension("httpfs")
    con.install_extension("aws")
    con.load_extension("aws")
    con.sql("CREATE OR REPLACE SECRET (TYPE S3, PROVIDER CREDENTIAL_CHAIN)")
    con.sql(f"SET s3_region = '{region}'")
    if not enable_cache:
        # Worst-case/no-cache cold benchmarking (2026-08-15): duckdb.connect()
        # with no path is already an in-memory database with no local
        # disk-backed cache in the S3 read path (httpfs streams straight into
        # this connection's own buffer pool, never through a local file Linux's
        # page cache could intercept), so the fresh-connection-per-cold-rep
        # reset above is the real cache-clearing mechanism, not an OS-level
        # cache drop. These two SETs are belt-and-suspenders on top of that:
        # DuckDB's own object/HTTP-metadata caches default on and are otherwise
        # scoped to the connection's lifetime anyway (already cleared by the
        # reconnect), but disabling them explicitly removes any doubt that a
        # within-connection cache could still be doing something.
        con.sql("SET enable_object_cache = false")
        con.sql("SET enable_http_metadata_cache = false")
    return con


def run_duckdb_query(con, query_number: int, globs: dict) -> tuple:
    sql = kernellake_sql(
        query_number, globs["data"], globs.get("part_data"), globs.get("orders_data"), globs.get("customer_data")
    )
    start = time.perf_counter()
    table = con.sql(sql).arrow().read_all()
    elapsed = time.perf_counter() - start
    return table, elapsed


def build_globs(
    bucket: str, scale_factor: int, query_number: int, compression: str,
    compression_level: int | None = None,
) -> dict:
    globs = {"data": s3_data_glob(bucket, scale_factor, "lineitem", compression, compression_level)}
    table_for_key = {"part_data": "part", "orders_data": "orders", "customer_data": "customer"}
    if query_number in QUERIES_WITH_SECOND_TABLE:
        key = QUERIES_WITH_SECOND_TABLE[query_number]
        globs[key] = s3_data_glob(bucket, scale_factor, table_for_key[key], compression, compression_level)
    if query_number in QUERIES_WITH_THIRD_TABLE:
        key = QUERIES_WITH_THIRD_TABLE[query_number]
        globs[key] = s3_data_glob(bucket, scale_factor, table_for_key[key], compression, compression_level)
    return globs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s3-bucket", required=True)
    parser.add_argument("--scale-factor", type=int, required=True)
    parser.add_argument("--compression", default="snappy", choices=["none", "snappy", "zstd"])
    parser.add_argument("--compression-level", type=int, default=None,
                         help="zstd only -- reads tpch-data/sf<N>-zstd-l<N>/ instead of the "
                              "un-suffixed tpch-data/sf<N>-zstd/ (PyArrow's own default level).")
    parser.add_argument("--query", default="all", help="Query number, or 'all' for every supported query")
    # 1, not 2 (2026-08-15): now that every cold rep gets a fresh
    # connection (see new_duckdb_connection() call site below), a second
    # rep is a genuine but redundant second cold measurement, not a
    # median-vs-noise check worth its own real cost/time at SF1000+ scale
    # -- median_stats() of a single sample just reports that sample.
    parser.add_argument("--iterations", type=int, default=1)
    # Defaults to cold-only (2026-08-15): warm/cached numbers don't
    # represent real behavior once a table's working set exceeds whatever
    # cache is in front of it, which any real production scale eventually
    # does -- cold is the honest worst-case number to report. Pass
    # `--modes cold,warm` explicitly to still measure both.
    parser.add_argument("--modes", default="cold", help="Comma-separated: cold,warm (see module docstring)")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    queries = list(ALL_QUERIES) if args.query == "all" else [int(args.query)]
    modes = tuple(args.modes.split(","))

    con = new_duckdb_connection(args.region)
    results = []
    run_start_unix = time.time()
    for query_number in queries:
        globs = build_globs(args.s3_bucket, args.scale_factor, query_number, args.compression, args.compression_level)
        entry: dict = {"query": query_number, "modes": {}}
        for mode in modes:
            samples = []
            row_count = None
            for rep in range(args.iterations):
                # Every cold rep gets a fresh connection, not just rep 0 --
                # with rep 0 only, a second cold rep reused rep 0's now-
                # populated connection state, making it indistinguishable
                # from a warm rep. Confirmed for real on a SF1000 run
                # (2026-08-15): "cold" min_seconds landed within noise of
                # the corresponding warm median every time, and
                # statistics.median() of [true_cold, accidentally_warm]
                # silently averaged the two -- reported "cold" numbers were
                # diluted roughly 2x toward warm, not a genuine cold
                # measurement at all.
                if mode == "cold":
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
        "compression_level": args.compression_level,
        "run_start_unix": run_start_unix,
        "run_end_unix": time.time(),
        "queries": results,
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
