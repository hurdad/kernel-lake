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

    # Q2 needs five extra tables (part/partsupp/supplier/nation/region) --
    # same --*-data flags tools/validate_tpch.py already uses:
    python3 tools/benchmark_three_way.py \
        --kernellake-server build/gpu-dev/src/server/kernellake-server \
        --data '/tmp/kernellake-tpch/lineitem-*.parquet' \
        --part-data '/tmp/kernellake-tpch/part-*.parquet' \
        --partsupp-data '/tmp/kernellake-tpch/partsupp-*.parquet' \
        --supplier-data '/tmp/kernellake-tpch/supplier-*.parquet' \
        --nation-data '/tmp/kernellake-tpch/nation-*.parquet' \
        --region-data '/tmp/kernellake-tpch/region-*.parquet' \
        --query 2 --iterations 5

Requires: pyspark (+ a JVM on PATH), pyarrow, duckdb, adbc-driver-flightsql.
Not part of this project's own CPU-only dev environment's dependency set --
see docker/Dockerfile's `benchmark-gpu` stage, which adds all of them to a
`gpu-release`-based image specifically to run this script in a reproducible
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


def bytes_for_glob(data_glob: str) -> int:
    return sum(os.path.getsize(path) for path in glob.glob(data_glob))


def bytes_processed_for_query(
    query_number: int,
    data_glob: str,
    part_data_glob: str | None,
    orders_data_glob: str | None,
    customer_data_glob: str | None,
    nation_data_glob: str | None = None,
    supplier_data_glob: str | None = None,
    region_data_glob: str | None = None,
    partsupp_data_glob: str | None = None,
) -> int:
    # Real on-disk (compressed Parquet) bytes for whichever tables this
    # specific query's SQL actually references -- not every --data/
    # --part-data/--orders-data/--customer-data/--nation-data/
    # --supplier-data/--region-data/--partsupp-data glob passed on the
    # command line, since e.g. Q1/Q6 only ever touch {data} regardless of
    # what else was supplied for other queries in the same --query all
    # run. This is what "cost per TB processed" (see --cost-per-hour)
    # divides against: a real, measured input size, not a query's row
    # count or output size.
    text = load_query_text(query_number)
    total = bytes_for_glob(data_glob)
    if part_data_glob and "{part_data}" in text:
        total += bytes_for_glob(part_data_glob)
    if orders_data_glob and "{orders_data}" in text:
        total += bytes_for_glob(orders_data_glob)
    if customer_data_glob and "{customer_data}" in text:
        total += bytes_for_glob(customer_data_glob)
    if nation_data_glob and "{nation_data}" in text:
        total += bytes_for_glob(nation_data_glob)
    if supplier_data_glob and "{supplier_data}" in text:
        total += bytes_for_glob(supplier_data_glob)
    if region_data_glob and "{region_data}" in text:
        total += bytes_for_glob(region_data_glob)
    if partsupp_data_glob and "{partsupp_data}" in text:
        total += bytes_for_glob(partsupp_data_glob)
    return total


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


# One entry per non-{data} placeholder this whole toolchain understands,
# in the order each first appeared -- reused by both kernellake_sql() and
# spark_sql() below so the two never drift apart on which tables exist.
# Mirrors tools/validate_tpch.py's own load_query()/argparse surface
# exactly (that tool needed this full table set for Q5/Q7/Q9/Q10/Q11
# before this file did -- see its own module docstring for real per-query
# --*-data examples).
EXTRA_TABLE_PLACEHOLDERS = ("part_data", "orders_data", "customer_data", "nation_data", "supplier_data",
                           "region_data", "partsupp_data")


def kernellake_sql(
    query_number: int,
    data_glob: str,
    part_data_glob: str | None = None,
    orders_data_glob: str | None = None,
    customer_data_glob: str | None = None,
    nation_data_glob: str | None = None,
    supplier_data_glob: str | None = None,
    region_data_glob: str | None = None,
    partsupp_data_glob: str | None = None,
) -> str:
    text = load_query_text(query_number)
    globs = {
        "part_data": part_data_glob, "orders_data": orders_data_glob, "customer_data": customer_data_glob,
        "nation_data": nation_data_glob, "supplier_data": supplier_data_glob, "region_data": region_data_glob,
        "partsupp_data": partsupp_data_glob,
    }
    for placeholder in EXTRA_TABLE_PLACEHOLDERS:
        if f"{{{placeholder}}}" in text and not globs[placeholder]:
            raise ValueError(f"Q{query_number} needs a table -- pass --{placeholder.replace('_', '-')}")
    # {data}/lineitem is the one placeholder every *other* query needs but
    # Q11/Q13 don't reference at all (both are entirely lineitem-free --
    # see benchmarks/tpch/queries/q11.sql's/q13.sql's own header comments)
    # -- .replace() is a safe no-op when the placeholder isn't present, so
    # this line never needs its own conditional the way the optional
    # tables above do.
    text = text.replace("{data}", data_glob)
    for placeholder, glob in globs.items():
        if glob:
            text = text.replace(f"{{{placeholder}}}", glob)
    return text.strip()


def spark_sql(query_number: int) -> str:
    # The query files' only KernelLake-specific syntax is the
    # read_parquet('{data}')/read_parquet('{part_data}')/... table-valued-
    # function FROM clauses (see each query file's own "Deviations"
    # comment) -- Spark SQL has no such function, so these are rewritten to
    # plain table references against temp views the caller registers via
    # spark.read.parquet(...) instead ("lineitem"/"part"/"orders"/
    # "customer"/"nation"/"supplier"/"region"/"partsupp"). Everything else
    # in these query files is already plain ANSI SQL both engines
    # understand identically. Every placeholder here is optional -- only
    # the queries that actually reference a given one (e.g. {part_data}
    # for Q19/Q12/Q3, {nation_data} for Q7/Q9/Q10/Q11) have a match; Q11/
    # Q13 reference no {data}/lineitem at all (see kernellake_sql()'s own
    # comment), so unlike the substituted-tables loop below there is no
    # single "this one must always match" placeholder to assert on --
    # instead, after every substitution, assert no `{..._data}` placeholder
    # survives unrewritten (a real typo/missing-table bug would leave one
    # behind; a query file syntax error unrelated to these placeholders
    # would not, and isn't this function's job to catch).
    text = load_query_text(query_number)
    rewritten = re.sub(r"read_parquet\(\s*'\{data\}'\s*\)", "lineitem", text)
    for placeholder, table in (
        ("part_data", "part"), ("orders_data", "orders"), ("customer_data", "customer"),
        ("nation_data", "nation"), ("supplier_data", "supplier"), ("region_data", "region"),
        ("partsupp_data", "partsupp"),
    ):
        rewritten = re.sub(r"read_parquet\(\s*'\{" + placeholder + r"\}'\s*\)", table, rewritten)
    leftover = re.search(r"\{\w+_data\}", rewritten)
    if leftover:
        raise ValueError(f"Q{query_number}: unrewritten placeholder {leftover.group()!r} in Spark SQL "
                         "-- a table this query needs has no read_parquet(...) rewrite wired up above")
    return rewritten.strip()


def run_pyspark_query(
    spark,
    query_number: int,
    data_glob: str,
    part_data_glob: str | None = None,
    orders_data_glob: str | None = None,
    customer_data_glob: str | None = None,
    nation_data_glob: str | None = None,
    supplier_data_glob: str | None = None,
    region_data_glob: str | None = None,
    partsupp_data_glob: str | None = None,
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
        if nation_data_glob:
            evict_data_files(nation_data_glob)
        if supplier_data_glob:
            evict_data_files(supplier_data_glob)
        if region_data_glob:
            evict_data_files(region_data_glob)
        if partsupp_data_glob:
            evict_data_files(partsupp_data_glob)
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
    nation_data_glob: str | None = None,
    supplier_data_glob: str | None = None,
    region_data_glob: str | None = None,
    partsupp_data_glob: str | None = None,
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
        if nation_data_glob:
            evict_data_files(nation_data_glob)
        if supplier_data_glob:
            evict_data_files(supplier_data_glob)
        if region_data_glob:
            evict_data_files(region_data_glob)
        if partsupp_data_glob:
            evict_data_files(partsupp_data_glob)
    sql = kernellake_sql(
        query_number,
        data_glob,
        part_data_glob,
        orders_data_glob,
        customer_data_glob,
        nation_data_glob,
        supplier_data_glob,
        region_data_glob,
        partsupp_data_glob,
    )
    start = time.perf_counter()
    table = run_duckdb(sql)
    elapsed = time.perf_counter() - start
    return table, elapsed


def write_server_config(port: int, base_config_path: str = "config/kernellake-server.yaml") -> str:
    # A literal-string replace of the exact default ("port: 31337"), not a
    # generic `port: \d+` regex -- config/kernellake-server.yaml also has an
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
    nation_data_glob: str | None = None,
    supplier_data_glob: str | None = None,
    region_data_glob: str | None = None,
    partsupp_data_glob: str | None = None,
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
        if nation_data_glob:
            evict_data_files(nation_data_glob)
        if supplier_data_glob:
            evict_data_files(supplier_data_glob)
        if region_data_glob:
            evict_data_files(region_data_glob)
        if partsupp_data_glob:
            evict_data_files(partsupp_data_glob)
    sql = kernellake_sql(
        query_number,
        data_glob,
        part_data_glob,
        orders_data_glob,
        customer_data_glob,
        nation_data_glob,
        supplier_data_glob,
        region_data_glob,
        partsupp_data_glob,
    )
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


def cost_per_tb_dollars(cost_per_hour: float, median_seconds: float, bytes_processed: int) -> float | None:
    if bytes_processed <= 0:
        return None
    tb_processed = bytes_processed / 1e12
    hours = median_seconds / 3600
    return cost_per_hour * hours / tb_processed


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
    cost_per_hour: dict | None = None,
    nation_data_glob: str | None = None,
    supplier_data_glob: str | None = None,
    region_data_glob: str | None = None,
    partsupp_data_glob: str | None = None,
) -> dict:
    print(f"=== Q{query_number} ===")

    extra_globs = dict(
        nation_data_glob=nation_data_glob,
        supplier_data_glob=supplier_data_glob,
        region_data_glob=region_data_glob,
        partsupp_data_glob=partsupp_data_glob,
    )

    # Correctness first: one untimed, warm run per *enabled* engine (see
    # --backends -- a real reason to skip one exists, e.g. at a large
    # enough scale factor -- see docs/ROADMAP.md), cross-validated
    # pairwise before any timing is trusted. Cache state doesn't affect
    # correctness, so this never evicts.
    try:
        tables = {}
        if "kernellake-gpu-server" in backends:
            tables["kernellake-gpu-server"], _ = run_kernellake_server_query(
                server_cursor,
                query_number,
                data_glob,
                part_data_glob,
                orders_data_glob,
                customer_data_glob,
                **extra_globs,
            )
        if "pyspark" in backends:
            tables["pyspark"], _ = run_pyspark_query(
                spark, query_number, data_glob, part_data_glob, orders_data_glob, customer_data_glob, **extra_globs
            )
        if "duckdb" in backends:
            tables["duckdb"], _ = run_duckdb_query(
                query_number, data_glob, part_data_glob, orders_data_glob, customer_data_glob, **extra_globs
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

    bytes_processed = bytes_processed_for_query(
        query_number, data_glob, part_data_glob, orders_data_glob, customer_data_glob, **extra_globs
    )
    result = {
        "query": query_number,
        "validated": True,
        "row_count": row_count,
        "iterations": iterations,
        "bytes_processed": bytes_processed,
    }
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
                    **extra_globs,
                )
                timings["kernellake-gpu-server"].append(t)
            if "pyspark" in backends:
                _, t = run_pyspark_query(
                    spark,
                    query_number,
                    data_glob,
                    part_data_glob,
                    orders_data_glob,
                    customer_data_glob,
                    cold=cold,
                    **extra_globs,
                )
                timings["pyspark"].append(t)
            if "duckdb" in backends:
                _, t = run_duckdb_query(
                    query_number,
                    data_glob,
                    part_data_glob,
                    orders_data_glob,
                    customer_data_glob,
                    cold=cold,
                    **extra_globs,
                )
                timings["duckdb"].append(t)
            print(f"    [{mode}] iteration {i + 1}/{iterations} done")
        result[mode] = {engine: median_stats(timings[engine]) for engine in backends}
        if cost_per_hour:
            for engine in backends:
                if engine in cost_per_hour:
                    result[mode][engine]["cost_per_tb_dollars"] = cost_per_tb_dollars(
                        cost_per_hour[engine], result[mode][engine]["median_seconds"], bytes_processed
                    )
    return result


def print_summary_table(results: list, modes: tuple, backends: tuple, cost_per_hour: dict | None = None) -> None:
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

    if not cost_per_hour:
        return
    cost_columns = [e for e in backends if e in cost_per_hour]
    if not cost_columns:
        return
    print()
    print("=" * 90)
    print("Cost per TB processed (real on-disk Parquet bytes, $/hour you supplied via --cost-per-hour).")
    print("=" * 90)
    cost_labels = {e: columns[e].replace("median s", "$/TB") for e in cost_columns}
    cost_header = f"{'query':<7}{'mode':<7}" + "".join(f"{cost_labels[e]:<20}" for e in cost_columns)
    print(cost_header)
    print("-" * len(cost_header))
    for result in results:
        if not result.get("validated"):
            continue
        for mode in modes:
            cells = []
            for engine in cost_columns:
                value = result[mode][engine].get("cost_per_tb_dollars")
                cells.append(f"{value:<20.4f}" if value is not None else f"{'n/a':<20}")
            print(f"Q{result['query']:<6}{mode:<7}" + "".join(cells))
    print("=" * 90)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Glob pattern over generate_tpch.py's lineitem Parquet output")
    parser.add_argument(
        "--part-data",
        default=None,
        help="Glob pattern over generate_tpch.py's part Parquet output (Q19/Q17/Q16/Q20/Q2/Q8/Q9/Q14 need this)",
    )
    parser.add_argument(
        "--orders-data",
        default=None,
        help="Glob pattern over generate_tpch.py's orders Parquet output "
        "(Q12/Q21/Q22/Q5/Q7/Q8/Q9/Q10/Q18/Q4/Q13 need this)",
    )
    parser.add_argument(
        "--customer-data",
        default=None,
        help="Glob pattern over generate_tpch.py's customer Parquet output "
        "(Q3/Q22/Q5/Q7/Q8/Q10/Q18/Q13 need this)",
    )
    parser.add_argument(
        "--nation-data",
        default=None,
        help="Glob pattern over generate_tpch.py's nation Parquet output "
        "(Q21/Q2/Q20/Q5/Q7/Q8/Q9/Q10/Q11 need this)",
    )
    parser.add_argument(
        "--supplier-data",
        default=None,
        help="Glob pattern over generate_tpch.py's supplier Parquet output "
        "(Q16/Q21/Q2/Q20/Q5/Q7/Q8/Q9/Q11/Q15 need this)",
    )
    parser.add_argument(
        "--region-data",
        default=None,
        help="Glob pattern over generate_tpch.py's region Parquet output (Q2/Q5/Q8 need this)",
    )
    parser.add_argument(
        "--partsupp-data",
        default=None,
        help="Glob pattern over generate_tpch.py's partsupp Parquet output (Q16/Q2/Q20/Q9/Q11 need this)",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Query number (1-22) or 'all'",
    )
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
        "--cpu-cores",
        type=int,
        default=None,
        help="Cap PySpark (local[N] instead of local[*]) and DuckDB (SET threads TO N) to this many "
        "cores. Only affects the two CPU-bound engines -- KernelLake-GPU's own CPU-side work "
        "(query planning, Arrow/Flight SQL marshalling) is unaffected, since its real bottleneck is "
        "the GPU, not host cores. Default: no cap, each engine uses every core it can see.",
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
    parser.add_argument(
        "--cost-per-hour",
        default=None,
        help="Comma-separated engine=dollars-per-hour pairs (e.g. "
        "'gpu-server=3.06,pyspark=0.85,duckdb=0.10'), the $/hour of whatever hardware that engine "
        "actually runs on in your deployment -- there is no built-in default, since real cost varies by "
        "cloud region, on-demand vs. reserved pricing, and on-prem amortization, and a fabricated "
        "default would misrepresent an actual dollar figure. When given, each result gains a "
        "cost_per_tb_dollars figure (rate * median_seconds/3600, divided by real on-disk Parquet bytes "
        "that specific query reads / 1e12) -- a real cost-efficiency comparison, not just wall-clock "
        "time. An engine with no rate given here is simply left without a cost figure.",
    )
    args = parser.parse_args()

    cost_per_hour = None
    if args.cost_per_hour:
        cost_per_hour = {}
        for pair in args.cost_per_hour.split(","):
            engine, _, rate = pair.partition("=")
            if not rate:
                parser.error(f"--cost-per-hour: '{pair}' is not engine=dollars-per-hour")
            try:
                cost_per_hour[engine.strip()] = float(rate)
            except ValueError:
                parser.error(f"--cost-per-hour: '{rate}' is not a number")
        engine_name_by_flag = {"gpu-server": "kernellake-gpu-server", "pyspark": "pyspark", "duckdb": "duckdb"}
        unknown_cost_engines = [e for e in cost_per_hour if e not in engine_name_by_flag]
        if unknown_cost_engines:
            parser.error(f"--cost-per-hour: unknown engine(s) {unknown_cost_engines} -- choose from gpu-server, "
                        "pyspark, duckdb")
        cost_per_hour = {engine_name_by_flag[e]: rate for e, rate in cost_per_hour.items()}

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
        # Q1/Q6 need only --data. Every other supported query needs --data
        # plus some subset of the extra tables below -- each is silently
        # omitted from "all" without its required flag(s) (not attempted-
        # and-failed, since that's a missing argument, not a cross-engine
        # disagreement) -- pass --query <N> explicitly to see that error.
        # Table requirements mirror tools/validate_tpch.py's own per-query
        # --*-data documentation exactly (see that script's own module
        # docstring). Q15 is included despite its own documented
        # GPU-backend unreliability (`kernellake-gpu-server` always runs
        # the outer query on GPU; Q15's own HAVING subquery always runs on
        # CPU regardless -- see docs/ARCHITECTURE.md's "HAVING and scalar
        # subqueries" section) -- this benchmark's whole point is
        # validating cross-engine agreement before trusting a timing, so
        # an occasional Q15 validation failure here is real signal about a
        # real, already-documented gap, not a bug to paper over by
        # excluding it.
        required_extra_globs = {
            2: ("part_data", "partsupp_data", "supplier_data", "nation_data", "region_data"),
            3: ("orders_data", "customer_data"),
            4: ("orders_data",),
            5: ("orders_data", "customer_data", "supplier_data", "nation_data", "region_data"),
            7: ("orders_data", "customer_data", "supplier_data", "nation_data"),
            8: ("part_data", "supplier_data", "orders_data", "customer_data", "nation_data", "region_data"),
            9: ("part_data", "supplier_data", "partsupp_data", "orders_data", "nation_data"),
            10: ("orders_data", "customer_data", "nation_data"),
            11: ("partsupp_data", "supplier_data", "nation_data"),
            12: ("orders_data",),
            13: ("customer_data", "orders_data"),
            14: ("part_data",),
            15: ("supplier_data",),
            16: ("partsupp_data", "part_data", "supplier_data"),
            17: ("part_data",),
            18: ("customer_data", "orders_data"),
            19: ("part_data",),
            20: ("supplier_data", "nation_data", "partsupp_data", "part_data"),
            21: ("supplier_data", "orders_data", "nation_data"),
            22: ("customer_data", "orders_data"),
        }
        query_numbers = [1, 6] + sorted(
            query_number
            for query_number, needed_globs in required_extra_globs.items()
            if all(getattr(args, glob_name) for glob_name in needed_globs)
        )
    else:
        query_numbers = [int(args.query)]

    modes = MODES if args.mode == "both" else (args.mode,)

    system_stats = collect_system_stats()
    print("=== System ===")
    for key, value in system_stats.items():
        print(f"    {key}: {value}")

    # Declared before the try block, not just before use: if either
    # SparkSession creation/temp-view registration or kernellake-server
    # startup below raises (e.g. a bad --kernellake-server path, the GPU
    # already busy, a slow GPU init exceeding the startup timeout, or a
    # bad --data glob Spark can't read), the *other* one may have already
    # started successfully -- both must still get torn down by the same
    # finally block rather than leaking a running JVM or server process
    # because the exception happened before a narrower try/finally around
    # just the query loop would have caught it.
    spark = None
    server_proc = None
    server_conn = None
    server_cursor = None
    results = []
    try:
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
            spark_master = f"local[{args.cpu_cores}]" if args.cpu_cores else "local[*]"
            spark = (
                SparkSession.builder.appName("kernellake-benchmark-three-way")
                .master(spark_master)
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
            if args.nation_data:
                spark.read.parquet(args.nation_data).createOrReplaceTempView("nation")
            if args.supplier_data:
                spark.read.parquet(args.supplier_data).createOrReplaceTempView("supplier")
            if args.region_data:
                spark.read.parquet(args.region_data).createOrReplaceTempView("region")
            if args.partsupp_data:
                spark.read.parquet(args.partsupp_data).createOrReplaceTempView("partsupp")

        if "duckdb" in backends and args.cpu_cores:
            # run_duckdb() (tools/duckdb_compare.py) runs every query
            # against DuckDB's own global default connection via
            # duckdb.sql(...), not a connection this script holds a handle
            # to -- so the thread cap has to be set on that same global
            # connection, once, before any query runs, rather than passed
            # per-call.
            import duckdb

            duckdb.sql(f"SET threads TO {args.cpu_cores}")

        if "kernellake-gpu-server" in backends:
            print(f"=== Starting kernellake-server on port {args.server_port} ===")
            server_proc = start_kernellake_server(args.kernellake_server, args.server_port)
            import adbc_driver_flightsql.dbapi as flightsql

            server_conn = flightsql.connect(f"grpc://127.0.0.1:{args.server_port}")
            server_cursor = server_conn.cursor()

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
                    cost_per_hour,
                    nation_data_glob=args.nation_data,
                    supplier_data_glob=args.supplier_data,
                    region_data_glob=args.region_data,
                    partsupp_data_glob=args.partsupp_data,
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

    print_summary_table(results, modes, backends, cost_per_hour)

    if args.output:
        report = {
            "unofficial": True,
            "disclaimer": "Unofficial TPC-H-derived benchmark. Not a certified TPC result.",
            "scale_factor": args.scale_factor,
            "data": args.data,
            "part_data": args.part_data,
            "orders_data": args.orders_data,
            "customer_data": args.customer_data,
            "nation_data": args.nation_data,
            "supplier_data": args.supplier_data,
            "region_data": args.region_data,
            "partsupp_data": args.partsupp_data,
            "kernellake_server": args.kernellake_server,
            "cost_per_hour": cost_per_hour,
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
