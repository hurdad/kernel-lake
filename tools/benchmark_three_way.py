#!/usr/bin/env python3
"""Three-way benchmark: KernelLake-CPU vs. KernelLake-GPU vs. PySpark.

Unofficial, TPC-H-*derived* benchmark (see tools/generate_tpch.py and
benchmarks/tpch/queries/ for the same caveats already documented there --
not the official dbgen/qgen tools, not a certified TPC-H result). Reuses
the exact same Parquet dataset (tools/generate_tpch.py) for all three
engines, and -- per this project's existing rule from tools/validate_tpch.py
(there: KernelLake vs. DuckDB; here: three engines instead of two) --
validates that all three engines agree on the result *before* trusting any
timing number. A query whose engines disagree is reported as a validation
failure and excluded from the timing table, never silently timed anyway.

Usage:
    python3 tools/benchmark_three_way.py \
        --kernellake build/gpu-dev/src/cli/kernellake \
        --data '/tmp/kernellake-tpch/*.parquet' \
        --query all --iterations 5

Requires: pyspark (+ a JVM on PATH), pyarrow. Not part of this project's
own CPU-only dev environment's dependency set -- see docker/Dockerfile's
`benchmark-gpu` stage, which adds both to a `dev-gpu`-based image
specifically to run this script in a reproducible container alongside a
real GPU (`docker run --gpus all`).
"""

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from duckdb_compare import normalize, rows_match, run_kernellake

QUERIES_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "tpch" / "queries"

ENGINES = ("kernellake-cpu", "kernellake-gpu", "pyspark")


def read_proc_field(path: str, field: str) -> str | None:
    try:
        with open(path) as f:
            for line in f:
                if line.startswith(field):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def total_ram_gib() -> float | None:
    mem_total_kb = read_proc_field("/proc/meminfo", "MemTotal")
    if mem_total_kb is None:
        return None
    try:
        return round(int(mem_total_kb.split()[0]) / (1024 * 1024), 2)
    except (ValueError, IndexError):
        return None


def gpu_info() -> dict | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    name, memory_total, driver_version = (part.strip() for part in result.stdout.strip().split(",")[:3])
    return {"name": name, "memory_total": memory_total, "driver_version": driver_version}


def collect_system_stats() -> dict:
    import pyspark

    return {
        "cpu_model": cpu_model(),
        "cpu_logical_cores": os.cpu_count(),
        "total_ram_gib": total_ram_gib(),
        "gpu": gpu_info(),
        "os": platform.platform(),
        "kernel_release": platform.release(),
        "python_version": platform.python_version(),
        "pyspark_version": pyspark.__version__,
    }


def load_query_text(query_number: int) -> str:
    path = QUERIES_DIR / f"q{query_number:02d}.sql"
    if not path.exists():
        raise FileNotFoundError(f"no query file for Q{query_number}: {path}")
    return re.sub(r"--[^\n]*\n", "\n", path.read_text())


def kernellake_sql(query_number: int, data_glob: str) -> str:
    return load_query_text(query_number).replace("{data}", data_glob).strip()


def spark_sql(query_number: int) -> str:
    # The query files' only KernelLake-specific syntax is the
    # read_parquet('{data}') table-valued-function FROM clause (see each
    # query file's own "Deviations" comment) -- Spark SQL has no such
    # function, so this is rewritten to a plain table reference against a
    # temp view the caller registers via spark.read.parquet(...) instead.
    # Everything else in these query files is already plain ANSI SQL both
    # engines understand identically.
    text = load_query_text(query_number)
    rewritten = re.sub(r"read_parquet\(\s*'\{data\}'\s*\)", "lineitem", text)
    if rewritten == text:
        raise ValueError(f"Q{query_number}: no read_parquet('{{data}}') found to rewrite for Spark SQL")
    return rewritten.strip()


def run_kernellake_backend(kernellake_bin: str, query_number: int, data_glob: str, backend: str):
    sql = kernellake_sql(query_number, data_glob)
    start = time.perf_counter()
    table = run_kernellake(kernellake_bin, sql, backend=backend)
    elapsed = time.perf_counter() - start
    return table, elapsed


def run_pyspark_query(spark, query_number: int):
    import pyarrow as pa

    sql = spark_sql(query_number)
    start = time.perf_counter()
    # Spark SQL is lazy -- toPandas() is what actually triggers execution
    # (via collect() internally), same as .collect() would but also gives a
    # ready-made bridge to pyarrow via pa.Table.from_pandas() below, and
    # handles a zero-row result correctly (unlike building a pa.Table
    # straight from an empty list of dicts, which can't infer a schema).
    pandas_df = spark.sql(sql).toPandas()
    elapsed = time.perf_counter() - start
    table = pa.Table.from_pandas(pandas_df, preserve_index=False)
    return table, elapsed


def median_stats(samples):
    return {
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.mean(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
    }


def benchmark_one_query(kernellake_bin: str, spark, query_number: int, data_glob: str, iterations: int) -> dict:
    print(f"=== Q{query_number} ===")

    # Correctness first: one untimed run per engine, cross-validated
    # pairwise before any timing is trusted.
    try:
        cpu_table, _ = run_kernellake_backend(kernellake_bin, query_number, data_glob, "cpu")
        gpu_table, _ = run_kernellake_backend(kernellake_bin, query_number, data_glob, "gpu")
        spark_table, _ = run_pyspark_query(spark, query_number)
    except Exception as exc:  # noqa: BLE001 -- reported as a validation failure, not a crash
        print(f"    ERROR during correctness check: {exc}")
        return {"query": query_number, "validated": False, "error": str(exc)}

    cpu_rows = normalize(cpu_table)
    gpu_rows = normalize(gpu_table)
    spark_rows = normalize(spark_table)

    pairs = [
        ("kernellake-cpu", "kernellake-gpu", cpu_rows, gpu_rows),
        ("kernellake-cpu", "pyspark", cpu_rows, spark_rows),
        ("kernellake-gpu", "pyspark", gpu_rows, spark_rows),
    ]
    mismatches = [(a, b) for a, b, ra, rb in pairs if not rows_match(ra, rb)]
    if mismatches:
        for a, b in mismatches:
            print(f"    FAIL: {a} and {b} disagree")
        print(f"    kernellake-cpu sample: {cpu_rows[:3]}")
        print(f"    kernellake-gpu sample: {gpu_rows[:3]}")
        print(f"    pyspark sample:        {spark_rows[:3]}")
        return {"query": query_number, "validated": False, "row_count": len(cpu_rows)}

    print(f"    PASS: all three engines agree ({len(cpu_rows)} rows)")

    # Now timed: iterations per engine, discarding the correctness-check
    # run above (it's untimed by construction, not reused for timing).
    timings = {engine: [] for engine in ENGINES}
    for i in range(iterations):
        _, t = run_kernellake_backend(kernellake_bin, query_number, data_glob, "cpu")
        timings["kernellake-cpu"].append(t)
        _, t = run_kernellake_backend(kernellake_bin, query_number, data_glob, "gpu")
        timings["kernellake-gpu"].append(t)
        _, t = run_pyspark_query(spark, query_number)
        timings["pyspark"].append(t)
        print(f"    iteration {i + 1}/{iterations} done")

    result = {"query": query_number, "validated": True, "row_count": len(cpu_rows), "iterations": iterations}
    for engine in ENGINES:
        result[engine] = median_stats(timings[engine])
    return result


def print_summary_table(results: list) -> None:
    print()
    print("=" * 78)
    print("UNOFFICIAL three-way benchmark -- NOT a certified TPC-H result.")
    print("KernelLake-CPU vs. KernelLake-GPU vs. PySpark (local[*]), same dataset.")
    print("=" * 78)
    header = f"{'query':<8}{'validated':<11}{'cpu (median s)':<16}{'gpu (median s)':<16}{'pyspark (median s)':<20}"
    print(header)
    print("-" * len(header))
    for result in results:
        if not result.get("validated"):
            print(f"Q{result['query']:<7}{'NO':<11}{'--':<16}{'--':<16}{'--':<20}")
            continue
        cpu_med = result["kernellake-cpu"]["median_seconds"]
        gpu_med = result["kernellake-gpu"]["median_seconds"]
        spark_med = result["pyspark"]["median_seconds"]
        print(f"Q{result['query']:<7}{'yes':<11}{cpu_med:<16.4f}{gpu_med:<16.4f}{spark_med:<20.4f}")
    print("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kernellake", required=True, help="Path to the kernellake CLI binary (gpu-dev build)")
    parser.add_argument("--data", required=True, help="Glob pattern over generate_tpch.py's Parquet output")
    parser.add_argument("--query", required=True, help="Query number (1, 6) or 'all'")
    parser.add_argument("--iterations", type=int, default=5, help="Timed iterations per engine per query")
    parser.add_argument(
        "--scale-factor", type=float, default=None, help="TPC-H scale factor of --data, recorded in --output only"
    )
    parser.add_argument("--output", default=None, help="Optional path to write the full report as JSON")
    args = parser.parse_args()

    if args.query == "all":
        query_numbers = [1, 6]
    else:
        query_numbers = [int(args.query)]

    system_stats = collect_system_stats()
    print("=== System ===")
    for key, value in system_stats.items():
        print(f"    {key}: {value}")

    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName("kernellake-benchmark-three-way")
        .master("local[*]")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    spark.read.parquet(args.data).createOrReplaceTempView("lineitem")

    results = []
    try:
        for query_number in query_numbers:
            results.append(benchmark_one_query(args.kernellake, spark, query_number, args.data, args.iterations))
    finally:
        spark.stop()

    print_summary_table(results)

    if args.output:
        report = {
            "unofficial": True,
            "disclaimer": "Unofficial TPC-H-derived benchmark. Not a certified TPC result.",
            "scale_factor": args.scale_factor,
            "data": args.data,
            "system": system_stats,
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote report to {args.output}")

    return 0 if all(r.get("validated") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
