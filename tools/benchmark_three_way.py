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
        --data '/tmp/kernellake-tpch/lineitem-*.parquet' \
        --query all --iterations 5

    # Q19 needs a second table (--query all only runs it if --part-data is
    # given -- see the --query all handling in main()):
    python3 tools/benchmark_three_way.py \
        --kernellake build/gpu-dev/src/cli/kernellake \
        --data '/tmp/kernellake-tpch/lineitem-*.parquet' \
        --part-data '/tmp/kernellake-tpch/part-*.parquet' \
        --query 19 --iterations 5

Requires: pyspark (+ a JVM on PATH), pyarrow. Not part of this project's
own CPU-only dev environment's dependency set -- see docker/Dockerfile's
`benchmark-gpu` stage, which adds both to a `dev-gpu`-based image
specifically to run this script in a reproducible container alongside a
real GPU (`docker run --gpus all`).
"""

import argparse
import glob
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
MODES = ("cold", "warm")


def evict_from_page_cache(path: str) -> None:
    # Best-effort cache eviction for a specific file, usable without root:
    # POSIX_FADV_DONTNEED asks the kernel to drop that file's cached pages.
    # Mirrors src/cli/benchmark_tpch_command.cpp's own
    # evict_from_page_cache() exactly (same rationale: a hint, not a
    # guarantee, and "cold" is approximate as a result) -- this Python
    # script is a separate orchestrator process from all three engines, but
    # it's the OS page cache for the underlying file that "cold" actually
    # means here, not anything engine-internal, so evicting it once here
    # (before each engine's own read) applies uniformly regardless of which
    # engine reads the file next.
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except (OSError, AttributeError):
        pass
    finally:
        os.close(fd)


def evict_data_files(data_glob: str) -> None:
    for path in glob.glob(data_glob):
        evict_from_page_cache(path)


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


def kernellake_sql(
    query_number: int, data_glob: str, part_data_glob: str | None = None, orders_data_glob: str | None = None
) -> str:
    text = load_query_text(query_number)
    if "{part_data}" in text and not part_data_glob:
        raise ValueError(f"Q{query_number} needs a second table -- pass --part-data")
    if "{orders_data}" in text and not orders_data_glob:
        raise ValueError(f"Q{query_number} needs a second table -- pass --orders-data")
    text = text.replace("{data}", data_glob)
    if part_data_glob:
        text = text.replace("{part_data}", part_data_glob)
    if orders_data_glob:
        text = text.replace("{orders_data}", orders_data_glob)
    return text.strip()


def spark_sql(query_number: int) -> str:
    # The query files' only KernelLake-specific syntax is the
    # read_parquet('{data}')/read_parquet('{part_data}')/
    # read_parquet('{orders_data}') table-valued-function FROM clauses (see
    # each query file's own "Deviations" comment) -- Spark SQL has no such
    # function, so these are rewritten to plain table references against
    # temp views the caller registers via spark.read.parquet(...) instead
    # ("lineitem"/"part"/"orders"). Everything else in these query files is
    # already plain ANSI SQL both engines understand identically.
    # {part_data}/{orders_data} are only present for queries needing that
    # second table (e.g. Q19/Q12); their regexes simply find no match
    # otherwise.
    text = load_query_text(query_number)
    rewritten = re.sub(r"read_parquet\(\s*'\{data\}'\s*\)", "lineitem", text)
    if rewritten == text:
        raise ValueError(f"Q{query_number}: no read_parquet('{{data}}') found to rewrite for Spark SQL")
    rewritten = re.sub(r"read_parquet\(\s*'\{part_data\}'\s*\)", "part", rewritten)
    rewritten = re.sub(r"read_parquet\(\s*'\{orders_data\}'\s*\)", "orders", rewritten)
    return rewritten.strip()


def run_kernellake_backend(
    kernellake_bin: str,
    query_number: int,
    data_glob: str,
    backend: str,
    part_data_glob: str | None = None,
    orders_data_glob: str | None = None,
    cold: bool = False,
):
    if cold:
        evict_data_files(data_glob)
        if part_data_glob:
            evict_data_files(part_data_glob)
        if orders_data_glob:
            evict_data_files(orders_data_glob)
    sql = kernellake_sql(query_number, data_glob, part_data_glob, orders_data_glob)
    start = time.perf_counter()
    table = run_kernellake(kernellake_bin, sql, backend=backend)
    elapsed = time.perf_counter() - start
    return table, elapsed


def run_pyspark_query(
    spark,
    query_number: int,
    data_glob: str,
    part_data_glob: str | None = None,
    orders_data_glob: str | None = None,
    cold: bool = False,
):
    import pyarrow as pa

    if cold:
        evict_data_files(data_glob)
        if part_data_glob:
            evict_data_files(part_data_glob)
        if orders_data_glob:
            evict_data_files(orders_data_glob)
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


def benchmark_one_query(
    kernellake_bin: str,
    spark,
    query_number: int,
    data_glob: str,
    part_data_glob: str | None,
    orders_data_glob: str | None,
    iterations: int,
    modes: tuple,
) -> dict:
    print(f"=== Q{query_number} ===")

    # Correctness first: one untimed, warm run per engine, cross-validated
    # pairwise before any timing is trusted. Cache state doesn't affect
    # correctness, so this never evicts.
    try:
        cpu_table, _ = run_kernellake_backend(
            kernellake_bin, query_number, data_glob, "cpu", part_data_glob, orders_data_glob
        )
        gpu_table, _ = run_kernellake_backend(
            kernellake_bin, query_number, data_glob, "gpu", part_data_glob, orders_data_glob
        )
        spark_table, _ = run_pyspark_query(spark, query_number, data_glob, part_data_glob, orders_data_glob)
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

    result = {"query": query_number, "validated": True, "row_count": len(cpu_rows), "iterations": iterations}
    for mode in modes:
        cold = mode == "cold"
        # Cache is evicted again before *each* engine's own read within an
        # iteration, not just once per iteration -- otherwise one engine's
        # read would silently re-warm the cache for the next engine in the
        # same iteration, making its own "cold" measurement actually warm.
        timings = {engine: [] for engine in ENGINES}
        for i in range(iterations):
            _, t = run_kernellake_backend(
                kernellake_bin, query_number, data_glob, "cpu", part_data_glob, orders_data_glob, cold=cold
            )
            timings["kernellake-cpu"].append(t)
            _, t = run_kernellake_backend(
                kernellake_bin, query_number, data_glob, "gpu", part_data_glob, orders_data_glob, cold=cold
            )
            timings["kernellake-gpu"].append(t)
            _, t = run_pyspark_query(
                spark, query_number, data_glob, part_data_glob, orders_data_glob, cold=cold
            )
            timings["pyspark"].append(t)
            print(f"    [{mode}] iteration {i + 1}/{iterations} done")
        result[mode] = {engine: median_stats(timings[engine]) for engine in ENGINES}
    return result


def print_summary_table(results: list, modes: tuple) -> None:
    print()
    print("=" * 90)
    print("UNOFFICIAL three-way benchmark -- NOT a certified TPC-H result.")
    print("KernelLake-CPU vs. KernelLake-GPU vs. PySpark (local[*]), same dataset.")
    print("=" * 90)
    header = (
        f"{'query':<7}{'mode':<7}{'validated':<11}{'cpu (median s)':<16}"
        f"{'gpu (median s)':<16}{'pyspark (median s)':<20}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        if not result.get("validated"):
            for mode in modes:
                print(f"Q{result['query']:<6}{mode:<7}{'NO':<11}{'--':<16}{'--':<16}{'--':<20}")
            continue
        for mode in modes:
            cpu_med = result[mode]["kernellake-cpu"]["median_seconds"]
            gpu_med = result[mode]["kernellake-gpu"]["median_seconds"]
            spark_med = result[mode]["pyspark"]["median_seconds"]
            print(f"Q{result['query']:<6}{mode:<7}{'yes':<11}{cpu_med:<16.4f}{gpu_med:<16.4f}{spark_med:<20.4f}")
    print("=" * 90)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kernellake", required=True, help="Path to the kernellake CLI binary (gpu-dev build)")
    parser.add_argument("--data", required=True, help="Glob pattern over generate_tpch.py's lineitem Parquet output")
    parser.add_argument(
        "--part-data", default=None, help="Glob pattern over generate_tpch.py's part Parquet output (Q19 needs this)"
    )
    parser.add_argument(
        "--orders-data",
        default=None,
        help="Glob pattern over generate_tpch.py's orders Parquet output (Q12 needs this)",
    )
    parser.add_argument("--query", required=True, help="Query number (1, 6, 12, 19) or 'all'")
    parser.add_argument("--iterations", type=int, default=5, help="Timed iterations per engine per query per mode")
    parser.add_argument(
        "--mode",
        choices=("cold", "warm", "both"),
        default="both",
        help="cold: evict the OS page cache (POSIX_FADV_DONTNEED, no root needed -- same approach as "
        "src/cli/benchmark_tpch_command.cpp's own --mode cold) before every single engine read. "
        "warm: no eviction, relies on whatever's already cached. both (default): run and report both.",
    )
    parser.add_argument(
        "--scale-factor", type=float, default=None, help="TPC-H scale factor of --data, recorded in --output only"
    )
    parser.add_argument("--output", default=None, help="Optional path to write the full report as JSON")
    parser.add_argument(
        "--spark-driver-memory",
        default="4g",
        help="spark.driver.memory for the local[*] SparkSession (default 4g; raise this for larger "
        "--data scale factors -- a real SF10/60M-row run hit a genuine Java heap OutOfMemoryError "
        "at the previous unconfigured default)",
    )
    args = parser.parse_args()

    if args.query == "all":
        # Q19 needs --part-data, Q12 needs --orders-data; each is silently
        # omitted from "all" without its flag (not attempted-and-failed,
        # since that's a missing argument, not a cross-engine disagreement)
        # -- pass --query 19/12 explicitly to see that error.
        query_numbers = [1, 6] + ([19] if args.part_data else []) + ([12] if args.orders_data else [])
    else:
        query_numbers = [int(args.query)]

    modes = MODES if args.mode == "both" else (args.mode,)

    system_stats = collect_system_stats()
    print("=== System ===")
    for key, value in system_stats.items():
        print(f"    {key}: {value}")

    from pyspark.sql import SparkSession

    # spark.driver.memory: local[*] mode runs the driver and every executor
    # thread inside one JVM process (no separate executor JVMs), so this is
    # the one setting that actually bounds all of it. Left at PySpark's own
    # small default, a real run against a genuinely large dataset (SF10,
    # 60M rows, local[*] spreading the scan across every CPU core at once)
    # hit a real "java.lang.OutOfMemoryError: Java heap space" failure --
    # confirmed by an actual run, not a hypothetical concern. args.driver_memory
    # is left tunable rather than hardcoded, since the right value scales
    # with both the dataset size and the host's available RAM.
    spark = (
        SparkSession.builder.appName("kernellake-benchmark-three-way")
        .master("local[*]")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.memory", args.spark_driver_memory)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    spark.read.parquet(args.data).createOrReplaceTempView("lineitem")
    if args.part_data:
        spark.read.parquet(args.part_data).createOrReplaceTempView("part")
    if args.orders_data:
        spark.read.parquet(args.orders_data).createOrReplaceTempView("orders")

    results = []
    try:
        for query_number in query_numbers:
            results.append(
                benchmark_one_query(
                    args.kernellake,
                    spark,
                    query_number,
                    args.data,
                    args.part_data,
                    args.orders_data,
                    args.iterations,
                    modes,
                )
            )
    finally:
        spark.stop()

    print_summary_table(results, modes)

    if args.output:
        report = {
            "unofficial": True,
            "disclaimer": "Unofficial TPC-H-derived benchmark. Not a certified TPC result.",
            "scale_factor": args.scale_factor,
            "data": args.data,
            "part_data": args.part_data,
            "orders_data": args.orders_data,
            "modes": list(modes),
            "system": system_stats,
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote report to {args.output}")

    return 0 if all(r.get("validated") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
