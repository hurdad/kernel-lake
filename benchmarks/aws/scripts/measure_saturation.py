#!/usr/bin/env python3
"""GPU/CPU/network saturation during a completed benchmark run, sourced
from the Prometheus already running on the monitoring instance (see
monitoring/prometheus.yml.tftpl -- dcgm-exporter on the KernelLake host,
node_exporter on all three), not ad-hoc SSH polling during the run itself.

Each engine's own results JSON (aws_benchmark_runner.py /
pyspark_query_loop.py / duckdb_query_loop.py) now records
run_start_unix/run_end_unix (real wall-clock, not perf_counter) spanning
its whole query loop -- this script takes those windows and asks
Prometheus, after the fact, what GPU/CPU/network looked like during each
one. Answers "how saturated was the KernelLake host's GPU during its run,
compared to how saturated the Spark/DuckDB hosts' CPUs were during
theirs" -- not a claim about any single query, a whole-run average/peak.

Prometheus's port 9090 is internal-only (see terraform/networking.tf's
security group -- self-referencing internal ingress, nothing from the
public internet), so this opens a local SSH tunnel to the monitoring
instance rather than hitting it directly, the same jump-host pattern this
project already uses to reach kernellake-server's Flight SQL port.

Usage:
    python3 measure_saturation.py \\
        --monitoring-host <monitoring-public-ip> --ssh-key ~/.ssh/kernellake.pem \\
        --engine-result kernellake-cache-on=results-cache-on.json \\
        --engine-result kernellake-cache-off=results-cache-off.json \\
        --engine-result pyspark=pyspark-results.json \\
        --engine-result duckdb=duckdb-results.json \\
        --output saturation.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Which Prometheus target label (instance_role, set per-job in
# prometheus.yml.tftpl) each engine's host traffic should be attributed
# to -- matched by prefix against the --engine-result NAME so
# "kernellake-cache-on"/"kernellake-cache-off" both resolve to the one
# real KernelLake host.
ROLE_FOR_ENGINE_PREFIX = [
    ("kernellake", "kernellake-host"),
    ("pyspark", "spark-host"),
    ("spark", "spark-host"),
    ("duckdb", "duckdb-host"),
]


def role_for_engine(name: str) -> str:
    for prefix, role in ROLE_FOR_ENGINE_PREFIX:
        if name.startswith(prefix):
            return role
    raise ValueError(f"--engine-result name '{name}' doesn't start with a known engine prefix "
                      f"({', '.join(p for p, _ in ROLE_FOR_ENGINE_PREFIX)})")


def open_tunnel(monitoring_host: str, ssh_key: str, local_port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        ["ssh", "-N", "-L", f"{local_port}:localhost:9090",
         "-i", ssh_key, "-o", "StrictHostKeyChecking=no", "-o", "ExitOnForwardFailure=yes",
         f"ubuntu@{monitoring_host}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    import socket

    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            _, stderr = proc.communicate()
            raise RuntimeError(f"SSH tunnel to {monitoring_host} exited early: {stderr.decode(errors='replace')}")
        try:
            with socket.create_connection(("localhost", local_port), timeout=1.0):
                return proc
        except OSError:
            time.sleep(0.5)
    proc.terminate()
    raise RuntimeError(f"SSH tunnel to {monitoring_host}:9090 didn't come up within 20s")


def query_range(base_url: str, promql: str, start: float, end: float, step: str = "15s") -> list[tuple[float, float]]:
    import urllib.parse
    import urllib.request

    # A window shorter than one scrape interval (15s) returns zero
    # samples -- pad both ends so a fast query-loop run (e.g. a single
    # query, --query 6) still gets at least one real data point rather
    # than silently reporting "no data" for a run that plainly happened.
    start = start - 15.0
    end = end + 15.0
    qs = urllib.parse.urlencode({"query": promql, "start": start, "end": end, "step": step})
    with urllib.request.urlopen(f"{base_url}/api/v1/query_range?{qs}", timeout=15) as resp:
        payload = json.load(resp)
    if payload["status"] != "success":
        raise RuntimeError(f"Prometheus query failed: {payload}")
    results = payload["data"]["result"]
    if not results:
        return []
    # Single-series queries only here (avg(...)/sum(...) with no `by`) --
    # one result row is always expected; more than one means the PromQL
    # above stopped being a true aggregate, which would be a real bug in
    # this script, not something to silently sum away.
    if len(results) != 1:
        raise RuntimeError(f"expected exactly one series from an aggregate query, got {len(results)}: {promql}")
    return [(float(ts), float(val)) for ts, val in results[0]["values"]]


def summarize(samples: list[tuple[float, float]]) -> dict | None:
    if not samples:
        return None
    values = [v for _, v in samples]
    return {"avg": sum(values) / len(values), "max": max(values), "min": min(values), "samples": len(values)}


def measure_engine(base_url: str, name: str, run_start: float, run_end: float) -> dict:
    role = role_for_engine(name)
    metrics: dict = {
        "role": role,
        "run_start_unix": run_start,
        "run_end_unix": run_end,
        "duration_seconds": run_end - run_start,
    }

    cpu_busy_pct = (
        f'100 - (avg(rate(node_cpu_seconds_total{{mode="idle", instance_role="{role}"}}[30s])) * 100)'
    )
    metrics["cpu_busy_pct"] = summarize(query_range(base_url, cpu_busy_pct, run_start, run_end))

    net_recv_bps = f'sum(rate(node_network_receive_bytes_total{{device!="lo", instance_role="{role}"}}[30s]))'
    net_stats = summarize(query_range(base_url, net_recv_bps, run_start, run_end))
    if net_stats:
        net_stats["avg_mbps"] = net_stats["avg"] * 8 / 1_000_000
        net_stats["max_mbps"] = net_stats["max"] * 8 / 1_000_000
    metrics["network_receive_bytes_per_sec"] = net_stats

    if role == "kernellake-host":
        gpu_util_pct = f'avg(DCGM_FI_DEV_GPU_UTIL{{instance_role="{role}"}})'
        metrics["gpu_util_pct"] = summarize(query_range(base_url, gpu_util_pct, run_start, run_end))

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--monitoring-host", required=True)
    parser.add_argument("--ssh-key", required=True)
    parser.add_argument("--local-port", type=int, default=19090)
    parser.add_argument("--engine-result", action="append", required=True,
                         help="NAME=path/to/results.json, repeatable. NAME must start with one of: "
                              + ", ".join(p for p, _ in ROLE_FOR_ENGINE_PREFIX))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    engine_results = {}
    for spec in args.engine_result:
        name, _, path = spec.partition("=")
        if not path:
            print(f"--engine-result must be NAME=path, got: {spec}", file=sys.stderr)
            return 1
        data = json.loads(Path(path).read_text())
        if "run_start_unix" not in data or "run_end_unix" not in data:
            print(f"{path} has no run_start_unix/run_end_unix -- was it produced by an updated "
                  f"aws_benchmark_runner.py/pyspark_query_loop.py/duckdb_query_loop.py?", file=sys.stderr)
            return 1
        engine_results[name] = (data["run_start_unix"], data["run_end_unix"])

    print(f"Opening SSH tunnel to {args.monitoring_host}:9090...", file=sys.stderr)
    tunnel = open_tunnel(args.monitoring_host, args.ssh_key, args.local_port)
    base_url = f"http://localhost:{args.local_port}"
    try:
        output = {}
        for name, (run_start, run_end) in engine_results.items():
            print(f"=== {name} (role={role_for_engine(name)}, "
                  f"{run_end - run_start:.0f}s window) ===", file=sys.stderr)
            output[name] = measure_engine(base_url, name, run_start, run_end)
            print(json.dumps(output[name], indent=2), file=sys.stderr)
    finally:
        tunnel.terminate()
        tunnel.wait(timeout=5)

    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
