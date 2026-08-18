#!/usr/bin/env python3
"""PySpark's half of the concurrency headline test (see
docs/CONCURRENCY_HARNESS_DESIGN.md) -- mirrors scaling_test.py's shape
(same worker/output pattern, same --scaling-results pipeline downstream)
but against a real Spark Thrift Server instead of KernelLake.

Needs a Thrift Server already running on the Spark host, NOT
pyspark_query_loop.py's in-process local[*] driver -- that one has no
concept of multiple concurrent client sessions at all. Start one like
this (`--master local[*]`, same single-JVM-uses-all-cores mode
pyspark_query_loop.py already uses -- deliberately not a real
spark://<master>:7077 standalone cluster; see that script's own docstring
for the real Master/Worker-topology bugs this project hit and moved away
from):

    $SPARK_HOME/sbin/start-thriftserver.sh \\
        --master local[*] \\
        --conf spark.driver.memory=48g \\
        --conf spark.scheduler.mode=FAIR \\
        --conf spark.hadoop.fs.s3a.aws.credentials.provider=com.amazonaws.auth.InstanceProfileCredentialsProvider \\
        --conf spark.local.dir=/var/spark-tmp \\
        --packages org.apache.hadoop:hadoop-aws:3.3.4 \\
        --hiveconf hive.server2.thrift.port=10000

`spark.scheduler.mode=FAIR` is not optional for this test: Spark's
default FIFO scheduler serializes jobs within one SparkContext -- the
same real-work-serializes-behind-one-lock risk class as
GpuExecutionCoordinator's mutex on the KernelLake side, just not
documented anywhere as prominently on Spark's side. Running this test
under the default FIFO scheduler would understate Spark's real
concurrent capability and isn't a fair comparison.

Needs `pip install pyhive[hive] thrift thrift_sasl` on whichever host
runs this script (the orchestrator, not necessarily the Spark host
itself -- pyhive is a pure network client over the Thrift wire protocol,
same relationship aws_benchmark_runner.py's ADBC client has to
kernellake-server).

This script's own setup step registers each table this query needs as a
real (non-temporary) external table -- `CREATE TABLE ... USING parquet
OPTIONS (path '<glob>')`, which resolves glob patterns the same
underlying DataSource code path spark.read.parquet(glob) does -- once,
before spawning concurrent workers, so every worker's own Thrift session
sees the same catalog without needing pyspark_query_loop.py's
createOrReplaceTempView() (session-scoped, not visible across separate
Thrift client connections; a plain external table has no such scoping
problem and needs no `global_temp.` prefix on every query, unlike a
GLOBAL TEMPORARY VIEW). Reuses spark_sql()'s existing rewrite (bare
"lineitem"/"orders"/etc. table references) unchanged.

Usage:
    python3 spark_scaling_test.py \\
        --thrift-host <spark-host-ip> --s3-bucket kernellake-bench-666052791151-ab12cd34 \\
        --scale-factor 100 --query 6 --concurrent-clients 8 --duration-seconds 120 \\
        --output scaling-pyspark.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pyspark_query_loop import (  # noqa: E402
    QUERIES_WITH_SECOND_TABLE,
    QUERIES_WITH_THIRD_TABLE,
    s3_data_glob,
    spark_sql,
)


def new_thrift_connection(host: str, port: int):
    from pyhive import hive

    return hive.Connection(host=host, port=port, username="ubuntu")


def register_tables(
    host: str, port: int, bucket: str, scale_factor: int, query_number: int,
    compression: str, compression_level: int | None,
) -> None:
    tables_needed = ["lineitem"]
    if query_number in QUERIES_WITH_SECOND_TABLE:
        tables_needed.append(QUERIES_WITH_SECOND_TABLE[query_number])
    if query_number in QUERIES_WITH_THIRD_TABLE:
        tables_needed.append(QUERIES_WITH_THIRD_TABLE[query_number])

    conn = new_thrift_connection(host, port)
    cursor = conn.cursor()
    for table in tables_needed:
        glob = s3_data_glob(bucket, scale_factor, table, compression, compression_level)
        cursor.execute(f"DROP TABLE IF EXISTS {table}")
        cursor.execute(f"CREATE TABLE {table} USING parquet OPTIONS (path '{glob}')")
    cursor.close()
    conn.close()


def worker(
    host: str, port: int, query_number: int, stop_event: threading.Event, latencies: list, lock: threading.Lock,
) -> None:
    conn = new_thrift_connection(host, port)
    cursor = conn.cursor()
    sql = spark_sql(query_number)
    while not stop_event.is_set():
        try:
            start = time.perf_counter()
            cursor.execute(sql)
            cursor.fetchall()
            elapsed = time.perf_counter() - start
        except Exception as e:  # noqa: BLE001 -- a single failed query shouldn't kill the whole concurrency test; recorded, not silently dropped.
            print(f"WARNING: query failed: {e}", file=sys.stderr)
            continue
        with lock:
            latencies.append(elapsed)
    cursor.close()
    conn.close()


def run_scaling_test(
    host: str, port: int, query_number: int, concurrent_clients: int, duration_seconds: float,
) -> dict:
    latencies: list = []
    lock = threading.Lock()
    stop_event = threading.Event()

    with ThreadPoolExecutor(max_workers=concurrent_clients) as pool:
        futures = [
            pool.submit(worker, host, port, query_number, stop_event, latencies, lock)
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
        "engine": "pyspark",
        "mode": "local[*] thrift server, FAIR scheduling",
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
    parser.add_argument("--thrift-host", required=True, help="Spark host running start-thriftserver.sh (see module docstring)")
    parser.add_argument("--thrift-port", type=int, default=10000)
    parser.add_argument("--s3-bucket", required=True)
    parser.add_argument("--scale-factor", type=int, required=True)
    parser.add_argument("--compression", default="snappy", choices=["none", "snappy", "zstd"])
    parser.add_argument("--compression-level", type=int, default=None)
    parser.add_argument("--query", type=int, default=6, help="A single-table query (1 or 6) keeps this test's SQL simple/uniform, same as scaling_test.py")
    parser.add_argument("--concurrent-clients", type=int, default=8)
    parser.add_argument("--duration-seconds", type=float, default=120.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"=== Registering tables for Q{args.query} on {args.thrift_host}:{args.thrift_port} ===", file=sys.stderr)
    register_tables(
        args.thrift_host, args.thrift_port, args.s3_bucket, args.scale_factor, args.query,
        args.compression, args.compression_level,
    )

    print(f"=== PySpark concurrency test: {args.concurrent_clients} concurrent clients, "
          f"{args.duration_seconds}s ===", file=sys.stderr)
    result = run_scaling_test(args.thrift_host, args.thrift_port, args.query, args.concurrent_clients, args.duration_seconds)
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
