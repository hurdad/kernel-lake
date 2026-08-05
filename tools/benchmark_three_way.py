#!/usr/bin/env python3
"""Three-way benchmark: KernelLake-GPU (server) vs. PySpark vs. DuckDB.

Unofficial, TPC-H-*derived* benchmark (see tools/generate_tpch.py and
benchmarks/tpch/queries/ for the same caveats already documented there --
not the official dbgen/qgen tools, not a certified TPC-H result). Reuses
the exact same Parquet dataset (tools/generate_tpch.py) for all engines
(DuckDB and KernelLake even share the identical substituted SQL string,
since DuckDB natively supports the same `read_parquet(...) AS alias JOIN`
syntax; PySpark needs its own placeholder-to-temp-view rewrite instead),
and -- per this project's existing rule from tools/validate_tpch.py
(there: KernelLake vs. DuckDB; here: N engines instead of two) --
validates that every enabled engine agrees on the result *before* trusting
any timing number. A query whose engines disagree is reported as a
validation failure and excluded from the timing table, never silently
timed anyway.

KernelLake is measured by hitting a long-lived `kernellake-server` process
over Arrow Flight SQL (via adbc-driver-flightsql), not by relaunching the
`kernellake` CLI as a fresh subprocess for every query. An earlier version
of this script did exactly that (one subprocess per query, `--backend
cpu`/`gpu`), and it turned out to measure mostly fixed overhead rather
than real execution time: process launch, dynamic-linker loading of every
shared library (Arrow/Parquet/CUDA/cudf/RMM), and -- specific to the GPU
backend -- a fresh CUDA context + RMM memory pool init on *every single
query*. `kernellake-server` constructs that same `RmmEnvironment` exactly
**once**, at server startup (`src/server/flight_sql_server.cpp`'s
constructor), and reuses it for every request after that -- confirmed for
real to be 6-11x faster than the CLI-subprocess measurement at the same
scale factor (see docs/ROADMAP.md). The CPU backend and the CLI-subprocess
GPU measurement were dropped from this script for that reason, not because
either backend itself was removed from KernelLake.

Usage:
    python3 tools/benchmark_three_way.py \
        --kernellake-server build/gpu-dev/src/server/kernellake-server \
        --data '/tmp/kernellake-tpch/lineitem-*.parquet' \
        --query all --iterations 5

    # Q19 needs a second table (--query all only runs it if --part-data is
    # given -- see the --query all handling in main()):
    python3 tools/benchmark_three_way.py \
        --kernellake-server build/gpu-dev/src/server/kernellake-server \
        --data '/tmp/kernellake-tpch/lineitem-*.parquet' \
        --part-data '/tmp/kernellake-tpch/part-*.parquet' \
        --query 19 --iterations 5

Requires: pyspark (+ a JVM on PATH), pyarrow, duckdb, adbc-driver-flightsql.
Not part of this project's own CPU-only dev environment's dependency set --
see docker/Dockerfile's `benchmark-gpu` stage, which adds all of them to a
`dev-gpu`-based image specifically to run this script in a reproducible
container alongside a real GPU (`docker run --gpus all`).
"""

import argparse
import glob
import json
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

from duckdb_compare import normalize, rows_match, run_duckdb

QUERIES_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "tpch" / "queries"

ENGINES = ("kernellake-gpu-server", "pyspark", "duckdb")
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
    query_number: int,
    data_glob: str,
    part_data_glob: str | None = None,
    orders_data_glob: str | None = None,
    customer_data_glob: str | None = None,
) -> str:
    text = load_query_text(query_number)
    if "{part_data}" in text and not part_data_glob:
        raise ValueError(f"Q{query_number} needs a second table -- pass --part-data")
    if "{orders_data}" in text and not orders_data_glob:
        raise ValueError(f"Q{query_number} needs a second table -- pass --orders-data")
    if "{customer_data}" in text and not customer_data_glob:
        raise ValueError(f"Q{query_number} needs a third table -- pass --customer-data")
    text = text.replace("{data}", data_glob)
    if part_data_glob:
        text = text.replace("{part_data}", part_data_glob)
    if orders_data_glob:
        text = text.replace("{orders_data}", orders_data_glob)
    if customer_data_glob:
        text = text.replace("{customer_data}", customer_data_glob)
    return text.strip()


def spark_sql(query_number: int) -> str:
    # The query files' only KernelLake-specific syntax is the
    # read_parquet('{data}')/read_parquet('{part_data}')/
    # read_parquet('{orders_data}')/read_parquet('{customer_data}')
    # table-valued-function FROM clauses (see each query file's own
    # "Deviations" comment) -- Spark SQL has no such function, so these are
    # rewritten to plain table references against temp views the caller
    # registers via spark.read.parquet(...) instead
    # ("lineitem"/"part"/"orders"/"customer"). Everything else in these
    # query files is already plain ANSI SQL both engines understand
    # identically. {part_data}/{orders_data}/{customer_data} are only
    # present for queries needing that extra table (e.g. Q19/Q12/Q3); their
    # regexes simply find no match otherwise.
    text = load_query_text(query_number)
    rewritten = re.sub(r"read_parquet\(\s*'\{data\}'\s*\)", "lineitem", text)
    if rewritten == text:
        raise ValueError(f"Q{query_number}: no read_parquet('{{data}}') found to rewrite for Spark SQL")
    rewritten = re.sub(r"read_parquet\(\s*'\{part_data\}'\s*\)", "part", rewritten)
    rewritten = re.sub(r"read_parquet\(\s*'\{orders_data\}'\s*\)", "orders", rewritten)
    rewritten = re.sub(r"read_parquet\(\s*'\{customer_data\}'\s*\)", "customer", rewritten)
    return rewritten.strip()


def run_pyspark_query(
    spark,
    query_number: int,
    data_glob: str,
    part_data_glob: str | None = None,
    orders_data_glob: str | None = None,
    customer_data_glob: str | None = None,
    cold: bool = False,
):
    import pyarrow as pa

    if cold:
        evict_data_files(data_glob)
        if part_data_glob:
            evict_data_files(part_data_glob)
        if orders_data_glob:
            evict_data_files(orders_data_glob)
        if customer_data_glob:
            evict_data_files(customer_data_glob)
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


def run_duckdb_query(
    query_number: int,
    data_glob: str,
    part_data_glob: str | None = None,
    orders_data_glob: str | None = None,
    customer_data_glob: str | None = None,
    cold: bool = False,
):
    # Unlike Spark, DuckDB accepts read_parquet('path') AS alias JOIN ...
    # natively -- the exact same substituted SQL kernellake_sql() already
    # builds for KernelLake itself, no separate rewrite needed (the same
    # reason tools/validate_tpch.py reuses one substituted SQL string for
    # both run_kernellake() and run_duckdb()).
    if cold:
        evict_data_files(data_glob)
        if part_data_glob:
            evict_data_files(part_data_glob)
        if orders_data_glob:
            evict_data_files(orders_data_glob)
        if customer_data_glob:
            evict_data_files(customer_data_glob)
    sql = kernellake_sql(query_number, data_glob, part_data_glob, orders_data_glob, customer_data_glob)
    start = time.perf_counter()
    table = run_duckdb(sql)
    elapsed = time.perf_counter() - start
    return table, elapsed


def write_server_config(port: int, base_config_path: str = "config/kernellake.yaml") -> str:
    # A literal-string replace of the exact default ("port: 31337"), not a
    # generic `port: \d+` regex -- config/kernellake.yaml also has an
    # unrelated `hdfs.connection_config.port: 0` earlier in the file, which
    # a generic regex's first match would hit instead. engine.backend
    # already defaults to "gpu" (include/kernellake/common/config.hpp), so
    # no override needed there.
    text = Path(base_config_path).read_text()
    needle = "port: 31337"
    if needle not in text:
        raise RuntimeError(f"{base_config_path}: expected to find '{needle}' under server: to override")
    text = text.replace(needle, f"port: {port}", 1)
    out_path = Path(f"/tmp/kernellake-server-benchmark-{port}.yaml")
    out_path.write_text(text)
    return str(out_path)


def start_kernellake_server(server_bin: str, port: int, startup_timeout_seconds: float = 30.0) -> subprocess.Popen:
    config_path = write_server_config(port)
    proc = subprocess.Popen(
        [server_bin, "--config", config_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    # kernellake-server blocks in Serve() once Init() succeeds, with no
    # separate "ready" signal easy to consume from a Popen pipe
    # concurrently -- polling a raw TCP connect on its own listening port is
    # simpler and backend-agnostic (works whether GpuExecutionCoordinator's
    # RmmEnvironment construction, the slow part on the "gpu" backend, has
    # finished or not, since Init()/Serve() only run after that).
    deadline = time.monotonic() + startup_timeout_seconds
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            output = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"kernellake-server exited early (code {proc.returncode}):\n{output}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return proc
        except OSError:
            time.sleep(0.2)
    proc.terminate()
    raise RuntimeError(f"kernellake-server did not start listening on port {port} within {startup_timeout_seconds}s")


def run_kernellake_server_query(
    cursor,
    query_number: int,
    data_glob: str,
    part_data_glob: str | None = None,
    orders_data_glob: str | None = None,
    customer_data_glob: str | None = None,
    cold: bool = False,
):
    # Same substituted SQL as the CLI-subprocess and DuckDB paths -- the
    # server runs it through the exact same QueryEngine. Cache eviction
    # still applies here: the server process itself stays warm (that's the
    # whole point), but the underlying Parquet files' OS-page-cache state
    # is independent of the server process's lifetime, so "cold" still
    # means what it means for every other engine.
    if cold:
        evict_data_files(data_glob)
        if part_data_glob:
            evict_data_files(part_data_glob)
        if orders_data_glob:
            evict_data_files(orders_data_glob)
        if customer_data_glob:
            evict_data_files(customer_data_glob)
    sql = kernellake_sql(query_number, data_glob, part_data_glob, orders_data_glob, customer_data_glob)
    start = time.perf_counter()
    cursor.execute(sql)
    table = cursor.fetch_arrow_table()
    elapsed = time.perf_counter() - start
    return table, elapsed


def median_stats(samples):
    return {
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.mean(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
    }


def benchmark_one_query(
    spark,
    server_cursor,
    query_number: int,
    data_glob: str,
    part_data_glob: str | None,
    orders_data_glob: str | None,
    customer_data_glob: str | None,
    iterations: int,
    modes: tuple,
    backends: tuple,
) -> dict:
    print(f"=== Q{query_number} ===")

    # Correctness first: one untimed, warm run per *enabled* engine (see
    # --backends -- a real reason to skip one exists, e.g. at a large
    # enough scale factor -- see docs/ROADMAP.md), cross-validated
    # pairwise before any timing is trusted. Cache state doesn't affect
    # correctness, so this never evicts.
    try:
        tables = {}
        if "kernellake-gpu-server" in backends:
            tables["kernellake-gpu-server"], _ = run_kernellake_server_query(
                server_cursor, query_number, data_glob, part_data_glob, orders_data_glob, customer_data_glob
            )
        if "pyspark" in backends:
            tables["pyspark"], _ = run_pyspark_query(
                spark, query_number, data_glob, part_data_glob, orders_data_glob, customer_data_glob
            )
        if "duckdb" in backends:
            tables["duckdb"], _ = run_duckdb_query(
                query_number, data_glob, part_data_glob, orders_data_glob, customer_data_glob
            )
    except Exception as exc:  # noqa: BLE001 -- reported as a validation failure, not a crash
        print(f"    ERROR during correctness check: {exc}")
        return {"query": query_number, "validated": False, "error": str(exc)}

    rows_by_engine = {engine: normalize(table) for engine, table in tables.items()}

    pairs = [
        (a, b, rows_by_engine[a], rows_by_engine[b])
        for i, a in enumerate(backends)
        for b in backends[i + 1 :]
    ]
    mismatches = [(a, b) for a, b, ra, rb in pairs if not rows_match(ra, rb)]
    if mismatches:
        for a, b in mismatches:
            print(f"    FAIL: {a} and {b} disagree")
        for engine, rows in rows_by_engine.items():
            print(f"    {engine} sample: {rows[:3]}")
        return {"query": query_number, "validated": False, "row_count": len(next(iter(rows_by_engine.values())))}

    row_count = len(next(iter(rows_by_engine.values())))
    print(f"    PASS: {', '.join(backends)} agree ({row_count} rows)")

    result = {"query": query_number, "validated": True, "row_count": row_count, "iterations": iterations}
    for mode in modes:
        cold = mode == "cold"
        # Cache is evicted again before *each* engine's own read within an
        # iteration, not just once per iteration -- otherwise one engine's
        # read would silently re-warm the cache for the next engine in the
        # same iteration, making its own "cold" measurement actually warm.
        timings = {engine: [] for engine in backends}
        for i in range(iterations):
            if "kernellake-gpu-server" in backends:
                _, t = run_kernellake_server_query(
                    server_cursor,
                    query_number,
                    data_glob,
                    part_data_glob,
                    orders_data_glob,
                    customer_data_glob,
                    cold=cold,
                )
                timings["kernellake-gpu-server"].append(t)
            if "pyspark" in backends:
                _, t = run_pyspark_query(
                    spark, query_number, data_glob, part_data_glob, orders_data_glob, customer_data_glob, cold=cold
                )
                timings["pyspark"].append(t)
            if "duckdb" in backends:
                _, t = run_duckdb_query(
                    query_number, data_glob, part_data_glob, orders_data_glob, customer_data_glob, cold=cold
                )
                timings["duckdb"].append(t)
            print(f"    [{mode}] iteration {i + 1}/{iterations} done")
        result[mode] = {engine: median_stats(timings[engine]) for engine in backends}
    return result


def print_summary_table(results: list, modes: tuple, backends: tuple) -> None:
    print()
    print("=" * 90)
    print("UNOFFICIAL benchmark -- NOT a certified TPC-H result.")
    print("KernelLake-GPU (server) vs. PySpark (local[*]) vs. DuckDB.")
    if len(backends) < len(ENGINES):
        skipped = [e for e in ENGINES if e not in backends]
        print(f"NOTE: {', '.join(skipped)} skipped for this run (--backends) -- see docs/ROADMAP.md.")
    print("=" * 90)
    columns = {
        "kernellake-gpu-server": "gpu-server (median s)",
        "pyspark": "pyspark (median s)",
        "duckdb": "duckdb (median s)",
    }
    header = f"{'query':<7}{'mode':<7}{'validated':<11}" + "".join(
        f"{columns[engine]:<20}" for engine in backends
    )
    print(header)
    print("-" * len(header))
    for result in results:
        if not result.get("validated"):
            for mode in modes:
                print(f"Q{result['query']:<6}{mode:<7}{'NO':<11}" + "".join(f"{'--':<20}" for _ in backends))
            continue
        for mode in modes:
            medians = "".join(f"{result[mode][engine]['median_seconds']:<20.4f}" for engine in backends)
            print(f"Q{result['query']:<6}{mode:<7}{'yes':<11}{medians}")
    print("=" * 90)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Glob pattern over generate_tpch.py's lineitem Parquet output")
    parser.add_argument(
        "--part-data", default=None, help="Glob pattern over generate_tpch.py's part Parquet output (Q19 needs this)"
    )
    parser.add_argument(
        "--orders-data",
        default=None,
        help="Glob pattern over generate_tpch.py's orders Parquet output (Q12 needs this)",
    )
    parser.add_argument(
        "--customer-data",
        default=None,
        help="Glob pattern over generate_tpch.py's customer Parquet output (Q3 needs this)",
    )
    parser.add_argument("--query", required=True, help="Query number (1, 3, 6, 12, 19) or 'all'")
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
    parser.add_argument(
        "--kernellake-server",
        default=None,
        help="Path to the kernellake-server binary (gpu-dev build, at "
        "build/gpu-dev/src/server/kernellake-server). Required by --backends ...,gpu-server -- spawns "
        "one long-lived server process (paying CUDA context/RMM pool init once, at startup, not per "
        "query) and hits it over Arrow Flight SQL via adbc-driver-flightsql for every query instead.",
    )
    parser.add_argument(
        "--server-port", type=int, default=31337, help="Port for the --kernellake-server process to listen on"
    )
    parser.add_argument(
        "--backends",
        default="gpu-server,pyspark,duckdb",
        help="Comma-separated subset of gpu-server,pyspark,duckdb to run and cross-validate (default: "
        "all three). At least two backends are required, since this script's whole point is validating "
        "that engines agree before trusting any timing.",
    )
    args = parser.parse_args()

    backends = tuple(b.strip() for b in args.backends.split(","))
    backend_name_by_flag = {
        "gpu-server": "kernellake-gpu-server",
        "pyspark": "pyspark",
        "duckdb": "duckdb",
    }
    unknown = [b for b in backends if b not in backend_name_by_flag]
    if unknown:
        parser.error(f"--backends: unknown engine(s) {unknown} -- choose from gpu-server, pyspark, duckdb")
    backends = tuple(backend_name_by_flag[b] for b in backends)
    if len(backends) < 2:
        parser.error("--backends: need at least two engines to cross-validate against each other")
    if "kernellake-gpu-server" in backends and not args.kernellake_server:
        parser.error("--backends gpu-server requires --kernellake-server <path>")

    if args.query == "all":
        # Q19 needs --part-data, Q12 needs --orders-data, Q3 needs both
        # --orders-data and --customer-data; each is silently omitted from
        # "all" without its flag(s) (not attempted-and-failed, since that's
        # a missing argument, not a cross-engine disagreement) -- pass
        # --query 19/12/3 explicitly to see that error.
        query_numbers = (
            [1, 6]
            + ([19] if args.part_data else [])
            + ([12] if args.orders_data else [])
            + ([3] if (args.orders_data and args.customer_data) else [])
        )
    else:
        query_numbers = [int(args.query)]

    modes = MODES if args.mode == "both" else (args.mode,)

    system_stats = collect_system_stats()
    print("=== System ===")
    for key, value in system_stats.items():
        print(f"    {key}: {value}")

    spark = None
    if "pyspark" in backends:
        from pyspark.sql import SparkSession

        # spark.driver.memory: local[*] mode runs the driver and every
        # executor thread inside one JVM process (no separate executor
        # JVMs), so this is the one setting that actually bounds all of it.
        # Left at PySpark's own small default, a real run against a
        # genuinely large dataset (SF10, 60M rows, local[*] spreading the
        # scan across every CPU core at once) hit a real
        # "java.lang.OutOfMemoryError: Java heap space" failure -- confirmed
        # by an actual run, not a hypothetical concern. args.driver_memory
        # is left tunable rather than hardcoded, since the right value
        # scales with both the dataset size and the host's available RAM.
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
        if args.customer_data:
            spark.read.parquet(args.customer_data).createOrReplaceTempView("customer")

    server_proc = None
    server_conn = None
    server_cursor = None
    if "kernellake-gpu-server" in backends:
        print(f"=== Starting kernellake-server on port {args.server_port} ===")
        server_proc = start_kernellake_server(args.kernellake_server, args.server_port)
        import adbc_driver_flightsql.dbapi as flightsql

        server_conn = flightsql.connect(f"grpc://127.0.0.1:{args.server_port}")
        server_cursor = server_conn.cursor()

    results = []
    try:
        for query_number in query_numbers:
            results.append(
                benchmark_one_query(
                    spark,
                    server_cursor,
                    query_number,
                    args.data,
                    args.part_data,
                    args.orders_data,
                    args.customer_data,
                    args.iterations,
                    modes,
                    backends,
                )
            )
    finally:
        if spark is not None:
            spark.stop()
        if server_cursor is not None:
            server_cursor.close()
        if server_conn is not None:
            server_conn.close()
        if server_proc is not None:
            server_proc.terminate()
            server_proc.wait(timeout=10)

    print_summary_table(results, modes, backends)

    if args.output:
        report = {
            "unofficial": True,
            "disclaimer": "Unofficial TPC-H-derived benchmark. Not a certified TPC result.",
            "scale_factor": args.scale_factor,
            "data": args.data,
            "part_data": args.part_data,
            "orders_data": args.orders_data,
            "customer_data": args.customer_data,
            "kernellake_server": args.kernellake_server,
            "modes": list(modes),
            "backends": list(backends),
            "system": system_stats,
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote report to {args.output}")

    return 0 if all(r.get("validated") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
