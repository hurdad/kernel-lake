#!/usr/bin/env python3
"""DuckDB's half of the concurrency headline test (see
docs/CONCURRENCY_HARNESS_DESIGN.md) -- mirrors scaling_test.py's shape
exactly (same worker/output pattern, same aggregate_results.py/
generate_report.py --scaling-results pipeline downstream) but against
DuckDB instead of KernelLake.

Meant to run directly on terraform/duckdb_instance.tf's dedicated DuckDB
host, alongside duckdb_query_loop.py -- imports new_duckdb_connection()/
run_duckdb_query()/build_globs() from it rather than duplicating them, so
scp both files (plus queries/) together, same as duckdb_query_loop.py's
own docstring already documents.

No shared server here (DuckDB has none in this project) -- each
concurrent "client" is its own independent duckdb.connect() instance, own
buffer pool, own thread pool. That last part matters: DuckDB defaults a
fresh connection's thread pool to every core on the host, so N concurrent
independent connections would oversubscribe the host N-to-1 by default,
measuring CPU thrashing rather than DuckDB's real concurrent-query
behavior. Each worker explicitly caps its own connection to
os.cpu_count() // concurrent_clients threads (floor 1) so N clients
split the host's cores fairly, the same way N kernellake-server replicas
in scaling_test.py each get their own dedicated instance rather than
oversubscribing one.

Usage:
    python3 duckdb_scaling_test.py \\
        --s3-bucket kernellake-bench-666052791151-ab12cd34 --scale-factor 100 \\
        --query 6 --concurrent-clients 8 --duration-seconds 120 \\
        --output scaling-duckdb.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from duckdb_query_loop import build_globs, new_duckdb_connection, run_duckdb_query  # noqa: E402


def worker(
    region: str, threads_per_connection: int, query_number: int, globs: dict,
    stop_event: threading.Event, latencies: list, lock: threading.Lock,
) -> None:
    # enable_cache=True (unlike duckdb_query_loop.py's own cold-benchmark
    # default) -- this worker fires the *same* query repeatedly against
    # one warm connection for the whole test duration, so disabling
    # caching would measure concurrent cold-S3-access scaling instead of
    # the intended concurrent-execution-scaling question. See
    # new_duckdb_connection()'s own comment and
    # docs/CONCURRENCY_HARNESS_DESIGN.md.
    con = new_duckdb_connection(region, enable_cache=True)
    con.sql(f"SET threads TO {threads_per_connection}")
    while not stop_event.is_set():
        try:
            _, elapsed = run_duckdb_query(con, query_number, globs)
        except Exception as e:  # noqa: BLE001 -- a single failed query shouldn't kill the whole concurrency test; recorded, not silently dropped.
            print(f"WARNING: query failed: {e}", file=sys.stderr)
            continue
        with lock:
            latencies.append(elapsed)


def run_scaling_test(
    region: str, query_number: int, globs: dict, concurrent_clients: int, duration_seconds: float,
) -> dict:
    latencies: list = []
    lock = threading.Lock()
    stop_event = threading.Event()

    # Floor 1, not 0 -- more concurrent clients than cores still means
    # every client gets to run (just more slowly), not a division-by-zero
    # SET threads call. Real oversubscription at high concurrent_clients
    # counts is a legitimate, worth-observing outcome of this test, not
    # something to avoid by silently clamping concurrency.
    threads_per_connection = max(1, (os.cpu_count() or 1) // concurrent_clients)

    with ThreadPoolExecutor(max_workers=concurrent_clients) as pool:
        futures = [
            pool.submit(worker, region, threads_per_connection, query_number, globs, stop_event, latencies, lock)
            for _ in range(concurrent_clients)
        ]
        start = time.monotonic()
        time.sleep(duration_seconds)
        stop_event.set()
        for f in as_completed(futures, timeout=30):
            f.result()  # surfaces any worker-thread exception that isn't the expected per-query one above
        actual_duration = time.monotonic() - start

    completed = len(latencies)
    return {
        "engine": "duckdb",
        "concurrent_clients": concurrent_clients,
        "threads_per_connection": threads_per_connection,
        "duration_seconds": actual_duration,
        "queries_completed": completed,
        "queries_per_hour": completed / actual_duration * 3600 if actual_duration > 0 else 0,
        "latency_median_seconds": statistics.median(latencies) if latencies else None,
        "latency_p95_seconds": (
            statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else (max(latencies) if latencies else None)
        ),
        "latency_p99_seconds": (
            statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else (max(latencies) if latencies else None)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s3-bucket", required=True)
    parser.add_argument("--scale-factor", type=int, required=True)
    parser.add_argument("--compression", default="snappy", choices=["none", "snappy", "zstd"])
    parser.add_argument("--compression-level", type=int, default=None)
    parser.add_argument("--query", type=int, default=6, help="A single-table query (1 or 6) keeps this test's SQL simple/uniform, same as scaling_test.py")
    parser.add_argument("--concurrent-clients", type=int, default=8)
    parser.add_argument("--duration-seconds", type=float, default=120.0)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    globs = build_globs(args.s3_bucket, args.scale_factor, args.query, args.compression, args.compression_level)

    print(f"=== DuckDB concurrency test: {args.concurrent_clients} concurrent clients, "
          f"{args.duration_seconds}s ===", file=sys.stderr)
    result = run_scaling_test(args.region, args.query, globs, args.concurrent_clients, args.duration_seconds)
    result["query"] = args.query
    result["scale_factor"] = args.scale_factor
    result["s3_bucket"] = args.s3_bucket

    Path(args.output).write_text(json.dumps(result, indent=2))
    if result["latency_median_seconds"] is not None:
        print(f"Wrote {args.output}: {result['queries_completed']} queries, "
              f"{result['queries_per_hour']:.0f}/hour, median latency "
              f"{result['latency_median_seconds']:.3f}s", file=sys.stderr)
    else:
        print(f"Wrote {args.output}: 0 queries completed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
