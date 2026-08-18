#!/usr/bin/env python3
"""M4's concurrency headline test: does KernelLake's per-query latency/cost
advantage hold, shrink, or grow as concurrent query load increases?

Drives concurrent query load across N independent kernellake-server
replicas (1/2/4/8, each its own g6.8xlarge -- see README.md's "What this
does not measure": these replicas never coordinate, so this measures
*concurrent-query throughput*, not a single query getting faster). For a
fixed test duration, `--concurrent-clients` worker threads repeatedly fire
queries, round-robined across the N replica hosts; aggregate completed-query
count and per-query latency distribution are the result. Combine with
cost_model.py's `compute_run_cost()`/`cost_per_completed_query()` (using
this run's own actual instance-hours) to get cost-per-completed-query *as a
function of concurrency* -- the actual headline number for this test, not
just aggregate queries/hour on its own.

Usage:
    python3 scaling_test.py \\
        --kernellake-hosts 10.0.1.10,10.0.1.11,10.0.1.12,10.0.1.13 \\
        --s3-bucket kernellake-bench-666052791151-ab12cd34 --scale-factor 100 \\
        --query 6 --concurrent-clients 8 --duration-seconds 120 \\
        --output scaling-4replicas.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

from aws_benchmark_runner import connect_kernellake, run_kernellake_query, s3_data_glob  # noqa: E402


def worker(host: str, port: int, query_number: int, globs: dict, stop_event: threading.Event, latencies: list, lock: threading.Lock) -> None:
    conn = connect_kernellake(host, port)
    cursor = conn.cursor()
    while not stop_event.is_set():
        try:
            _, elapsed = run_kernellake_query(cursor, query_number, globs)
        except Exception as e:  # noqa: BLE001 -- a single failed query shouldn't kill the whole concurrency test; recorded, not silently dropped.
            print(f"WARNING: query against {host} failed: {e}", file=sys.stderr)
            continue
        with lock:
            latencies.append(elapsed)


def run_scaling_test(
    hosts: list[str], port: int, query_number: int, globs: dict, concurrent_clients: int, duration_seconds: float,
    grace_seconds: float,
) -> dict:
    latencies: list = []
    lock = threading.Lock()
    stop_event = threading.Event()

    host_cycle = itertools.cycle(hosts)
    with ThreadPoolExecutor(max_workers=concurrent_clients) as pool:
        futures = [
            pool.submit(worker, next(host_cycle), port, query_number, globs, stop_event, latencies, lock)
            for _ in range(concurrent_clients)
        ]
        start = time.monotonic()
        time.sleep(duration_seconds)
        stop_event.set()
        # grace_seconds, not a hardcoded 30 -- a query already in flight
        # when stop_event fires still has to finish before its worker
        # thread notices and exits, and at real SF1000 scale a single cold
        # query can take 150-280s on its own (confirmed for real: 30s
        # wasn't enough, raised a bare TimeoutError with the very first
        # single-client run this was tried against). --grace-seconds needs
        # to comfortably exceed one query's own worst-case wall time, not
        # just be "a bit of slack".
        for f in as_completed(futures, timeout=grace_seconds):
            f.result()  # surfaces any worker-thread exception that isn't the expected per-query one above
        actual_duration = time.monotonic() - start

    completed = len(latencies)
    return {
        "replica_count": len(hosts),
        "concurrent_clients": concurrent_clients,
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
    parser.add_argument("--kernellake-hosts", required=True, help="Comma-separated list of replica IPs")
    parser.add_argument("--kernellake-port", type=int, default=31337)
    parser.add_argument("--s3-bucket", required=True)
    parser.add_argument("--scale-factor", type=int, required=True)
    parser.add_argument("--query", type=int, default=6, help="A single-table query (1 or 6) keeps this test's SQL simple/uniform across all replicas")
    parser.add_argument("--concurrent-clients", type=int, default=None, help="Default: 2x replica count, so demand always exceeds replica capacity")
    parser.add_argument("--duration-seconds", type=float, default=120.0)
    parser.add_argument("--grace-seconds", type=float, default=300.0,
                         help="How long to wait for an already-in-flight query to finish after duration-seconds "
                              "elapses -- must comfortably exceed one query's own worst-case wall time (SF1000 "
                              "queries commonly take 150-280s cold), not just be a small buffer")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    hosts = args.kernellake_hosts.split(",")
    concurrent_clients = args.concurrent_clients or (2 * len(hosts))
    # scheme="s3", not s3_data_glob()'s own default ("s3a", sized for
    # Spark's Hadoop connector) -- this script is KernelLake-only (see
    # module docstring), and read_parquet() rejects "s3a://" outright
    # (confirmed for real: a live adbc OperationalError, "unsupported URI
    # scheme 's3a'").
    globs = {"data": s3_data_glob(args.s3_bucket, args.scale_factor, "lineitem", scheme="s3")}

    print(f"=== Concurrency test: {len(hosts)} replica(s), {concurrent_clients} concurrent clients, "
          f"{args.duration_seconds}s ===", file=sys.stderr)
    result = run_scaling_test(
        hosts, args.kernellake_port, args.query, globs, concurrent_clients, args.duration_seconds, args.grace_seconds
    )
    result["hosts"] = hosts
    result["query"] = args.query
    result["scale_factor"] = args.scale_factor

    Path(args.output).write_text(json.dumps(result, indent=2))
    if result["latency_median_seconds"] is not None:
        print(f"Wrote {args.output}: {result['queries_completed']} queries, "
              f"{result['queries_per_hour']:.0f}/hour, median latency {result['latency_median_seconds']:.3f}s",
              file=sys.stderr)
    else:
        print(f"Wrote {args.output}: 0 queries completed", file=sys.stderr)
    print("Combine this with cost_model.py's compute_run_cost() (this run's own real instance-hours) "
          "for cost-per-completed-query at this concurrency level -- the actual headline number.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
