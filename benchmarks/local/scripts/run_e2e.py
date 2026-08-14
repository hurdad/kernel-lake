#!/usr/bin/env python3
"""End-to-end test + TPC-H benchmark runner for benchmarks/local/'s docker-compose stack.

Unofficial TPC-H-derived benchmark. Not a certified TPC result.

Connects to kernellake-server over real Flight SQL (matching how
benchmarks/aws/runner/aws_benchmark_runner.py talks to a remote server),
runs each supported TPC-H-derived query (benchmarks/tpch/queries/) against
the MinIO-backed data scripts/generate_and_upload_data.sh uploaded,
cross-checks results against DuckDB (reusing tools/duckdb_compare.py and
tools/validate_tpch.py's own load_query()), reports per-query timing, and
finally checks that this stack's own observability pipeline actually
worked end to end: both kernellake.query.duration_seconds and the GPU
memory metrics (docs/OBSERVABILITY.md) show up in Prometheus, and at
least one real trace shows up in Jaeger, after the queries ran.

Usage:
    ./scripts/generate_and_upload_data.sh 1     # once, uploads SF1 data to MinIO
    ./scripts/run_e2e.py --scale-factor 1
    ./scripts/run_e2e.py --scale-factor 1 --query 6   # just one query
    ./scripts/run_e2e.py --scale-factor 1 --skip-metrics-check  # if Prometheus isn't up
    ./scripts/run_e2e.py --scale-factor 1 --skip-traces-check   # if Jaeger isn't up
"""

import argparse
import socket
import sys
import time
from pathlib import Path

import adbc_driver_flightsql.dbapi as flightsql
import pyarrow as pa
import requests

TOOLS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from duckdb_compare import normalize, rows_match, run_duckdb  # noqa: E402
from validate_tpch import load_query  # noqa: E402


def run_kernellake_via_flightsql(cursor, sql: str) -> pa.Table:
    cursor.execute(sql)
    return cursor.fetch_arrow_table()


def wait_for_server(host: str, port: int, timeout_s: float = 60.0) -> None:
    # Plain TCP connect, not a real Flight SQL round trip: KernelLake's SQL
    # grammar has no bare `SELECT 1` (every query needs a
    # read_parquet(...)/read_iceberg(...)/read_delta(...) data source --
    # see QueryEngine's own error message), and this check shouldn't
    # depend on scripts/generate_and_upload_data.sh having already run
    # anyway. Mirrors docker-compose.yml's own kernellake-server healthcheck
    # (bash's /dev/tcp).
    print(f"=== Waiting for kernellake-server at {host}:{port} ===")
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                print("    server is up")
                return
        except OSError as exc:
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"kernellake-server never became reachable at {host}:{port}: {last_error}")


def configure_duckdb_for_minio(minio_endpoint: str) -> None:
    import duckdb

    duckdb.sql("INSTALL httpfs")
    duckdb.sql("LOAD httpfs")
    duckdb.sql(f"SET s3_endpoint='{minio_endpoint}'")
    duckdb.sql("SET s3_url_style='path'")
    duckdb.sql("SET s3_use_ssl=false")
    duckdb.sql("SET s3_access_key_id='minioadmin'")
    duckdb.sql("SET s3_secret_access_key='minioadmin'")


def run_query(cursor, query_number: int, data_glob: str, part_glob: str, orders_glob: str,
             customer_glob: str) -> tuple[bool, float, int]:
    # nation/supplier/region/partsupp: this script's own `available` query list
    # (Q1/Q3/Q6/Q12/Q14/Q19) never references those tables -- only Q5/Q7/Q9/Q10
    # do (see tools/generate_tpch.py's docstring) -- so None is always safe
    # here, matching load_query()'s own "only required if the query's SQL
    # actually needs it" contract (tools/validate_tpch.py).
    sql = load_query(query_number, data_glob, part_glob, orders_glob, customer_glob,
                     None, None, None, None)
    print(f"--- Q{query_number}: {sql.splitlines()[0]}...")

    start = time.time()
    try:
        kernellake_rows = normalize(run_kernellake_via_flightsql(cursor, sql))
    except Exception as exc:  # noqa: BLE001
        print(f"    FAIL: kernellake query error: {exc}")
        return False, 0.0, 0
    elapsed = time.time() - start

    try:
        duckdb_rows = normalize(run_duckdb(sql))
    except Exception as exc:  # noqa: BLE001
        print(f"    FAIL: duckdb query error: {exc}")
        return False, elapsed, len(kernellake_rows)

    if rows_match(kernellake_rows, duckdb_rows):
        print(f"    PASS ({len(kernellake_rows)} rows, {elapsed:.3f}s)")
        return True, elapsed, len(kernellake_rows)

    print(f"    FAIL: kernellake={len(kernellake_rows)} rows, duckdb={len(duckdb_rows)} rows")
    print(f"    kernellake sample: {kernellake_rows[:3]}")
    print(f"    duckdb sample:     {duckdb_rows[:3]}")
    return False, elapsed, len(kernellake_rows)


def check_metrics(prometheus_url: str, expect_gpu_metrics: bool) -> bool:
    print(f"=== Checking metrics arrived at Prometheus ({prometheus_url}) ===")
    # kernellake-server.yaml sets metrics.export_interval_ms=5000 -- give
    # the periodic exporter a couple of cycles to actually flush before
    # querying Prometheus (which itself scrapes on a 5s interval too, see
    # prometheus.yml).
    time.sleep(12)

    ok = True

    def query_prometheus(promql: str) -> float | None:
        response = requests.get(f"{prometheus_url}/api/v1/query", params={"query": promql}, timeout=10)
        response.raise_for_status()
        result = response.json()["data"]["result"]
        if not result:
            return None
        return float(result[0]["value"][1])

    duration_count = query_prometheus("kernellake_query_duration_seconds_count")
    if duration_count is None or duration_count <= 0:
        print(f"    FAIL: kernellake_query_duration_seconds_count not found or zero (got {duration_count})")
        ok = False
    else:
        print(f"    OK: kernellake_query_duration_seconds_count = {duration_count}")

    if expect_gpu_metrics:
        # kernellake.gpu.memory.peak (unit "By", ObservableGauge) -- verified
        # for real against this stack's own Prometheus (see
        # docs/OBSERVABILITY.md §2.2.1's naming-translation note): the
        # Prometheus OTel bridge renders it as kernellake_gpu_memory_
        # peak_bytes (unit suffix inserted, no _total -- it's a gauge, not
        # a counter). Checked instead of *_allocated_bytes (current usage):
        # peak only ever moves up, so it stays nonzero after the query that
        # set it completes, where current usage legitimately drops back to
        # whatever baseline it was before.
        peak = query_prometheus("kernellake_gpu_memory_peak_bytes")
        if peak is None or peak <= 0:
            print(f"    FAIL: kernellake_gpu_memory_peak_bytes not found or zero (got {peak})")
            ok = False
        else:
            print(f"    OK: kernellake_gpu_memory_peak_bytes = {peak}")

        allocations = query_prometheus("kernellake_gpu_memory_allocations_total")
        if allocations is None or allocations <= 0:
            print(f"    FAIL: kernellake_gpu_memory_allocations_total not found or zero (got {allocations})")
            ok = False
        else:
            print(f"    OK: kernellake_gpu_memory_allocations_total = {allocations}")

    return ok


def check_jaeger_traces(jaeger_url: str, service_name: str = "kernellake") -> bool:
    print(f"=== Checking traces arrived at Jaeger ({jaeger_url}) ===")
    ok = True

    services_response = requests.get(f"{jaeger_url}/api/services", timeout=10)
    services_response.raise_for_status()
    services = services_response.json().get("data") or []
    if service_name not in services:
        print(f"    FAIL: service '{service_name}' not in Jaeger's known services: {services}")
        return False
    print(f"    OK: service '{service_name}' is known to Jaeger")

    traces_response = requests.get(
        f"{jaeger_url}/api/traces", params={"service": service_name, "limit": 1}, timeout=10
    )
    traces_response.raise_for_status()
    traces = traces_response.json().get("data") or []
    if not traces:
        print(f"    FAIL: no traces found for service '{service_name}'")
        ok = False
    else:
        span_count = sum(len(t.get("spans", [])) for t in traces)
        print(f"    OK: found {len(traces)} trace(s), {span_count} span(s) in the most recent one")

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=31337)
    parser.add_argument("--minio-endpoint", default="localhost:9000")
    parser.add_argument("--bucket", default="kernellake-bench")
    parser.add_argument("--scale-factor", type=float, default=1)
    parser.add_argument("--query", default="all", help="Query number (e.g. 6) or 'all'")
    parser.add_argument("--prometheus-url", default="http://localhost:9090")
    parser.add_argument("--skip-metrics-check", action="store_true")
    parser.add_argument("--expect-gpu-metrics", action="store_true", default=True)
    parser.add_argument("--no-gpu-metrics", dest="expect_gpu_metrics", action="store_false",
                       help="Pass this if the stack is running KERNELLAKE_TARGET=runtime-cpu")
    parser.add_argument("--jaeger-url", default="http://localhost:16686")
    parser.add_argument("--skip-traces-check", action="store_true")
    args = parser.parse_args()

    sf_label = str(args.scale_factor).rstrip("0").rstrip(".") if "." in str(args.scale_factor) else str(
        int(args.scale_factor))
    prefix = f"tpch-sf{sf_label}"
    data_glob = f"s3://{args.bucket}/{prefix}/lineitem-*.parquet"
    part_glob = f"s3://{args.bucket}/{prefix}/part-*.parquet"
    orders_glob = f"s3://{args.bucket}/{prefix}/orders-*.parquet"
    customer_glob = f"s3://{args.bucket}/{prefix}/customer-*.parquet"

    wait_for_server(args.host, args.port)
    configure_duckdb_for_minio(args.minio_endpoint)

    conn = flightsql.connect(f"grpc://{args.host}:{args.port}")
    cursor = conn.cursor()

    available = [1, 3, 6, 12, 14, 19]
    query_numbers = available if args.query == "all" else [int(args.query)]

    print(f"\n=== Running {len(query_numbers)} TPC-H-derived quer{'ies' if len(query_numbers) != 1 else 'y'} "
         f"against s3://{args.bucket}/{prefix}/ ===")
    results = []
    for query_number in query_numbers:
        passed, elapsed, rows = run_query(cursor, query_number, data_glob, part_glob, orders_glob, customer_glob)
        results.append((query_number, passed, elapsed, rows))

    cursor.close()
    conn.close()

    metrics_ok = True
    if not args.skip_metrics_check:
        metrics_ok = check_metrics(args.prometheus_url, args.expect_gpu_metrics)

    traces_ok = True
    if not args.skip_traces_check:
        traces_ok = check_jaeger_traces(args.jaeger_url)

    print("\n=== Summary ===")
    failures = 0
    for query_number, passed, elapsed, rows in results:
        status = "PASS" if passed else "FAIL"
        print(f"  Q{query_number:<3} {status:5} {elapsed:7.3f}s  {rows} rows")
        if not passed:
            failures += 1
    print(f"  metrics check: {'PASS' if metrics_ok else 'FAIL' if not args.skip_metrics_check else 'SKIPPED'}")
    print(f"  traces check:  {'PASS' if traces_ok else 'FAIL' if not args.skip_traces_check else 'SKIPPED'}")

    if failures or not metrics_ok or not traces_ok:
        print(f"\n{failures} quer{'y' if failures == 1 else 'ies'} failed" +
             ("" if metrics_ok else "; metrics check failed") +
             ("" if traces_ok else "; traces check failed"))
        return 1

    print("\nAll queries passed, metrics and traces confirmed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
