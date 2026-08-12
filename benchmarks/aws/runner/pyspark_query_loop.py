#!/usr/bin/env python3
"""Standalone PySpark TPC-H-derived query loop, meant to run directly on
terraform/spark_cluster.tf's dedicated single-node Spark host over SSH --
not in-process on the orchestrator, and not against a real multi-node
standalone cluster. Mirrors runner/duckdb_query_loop.py's own shape and
rationale exactly (see docs/RUNBOOK.md): every engine gets its own
dedicated, cost-tracked box with a real $/hour to attribute a query's cost
against, rather than sharing whichever machine happens to run the
orchestrator.

Always runs Spark in local[*] mode (SparkSession.builder.master("local[*]")
-- one JVM, using all of this host's own cores as "executors"), never a
real spark://<master>:7077 standalone cluster: see
new_spark_session()'s own comment for why this sidesteps a whole class of
standalone-mode-specific problems (Master/Worker daemon JMX-port
collisions, SPARK_LOCAL_DIRS silently overriding driver-side config) this
project's own history hit for real running the old multi-node topology.
--executor-memory's default assumes the standard spark_instance_type
(m7i.4xlarge, 16 vCPU/64GB RAM) -- resize it if you override that
variable to something smaller. Despite the name (kept for CLI back-
compat), it's applied to spark.driver.memory, the only JVM-heap knob
that actually does anything in local[*] mode -- see new_spark_session()'s
own comment for the real OOM this fixes.

Deliberately self-contained -- duplicates a few small helpers from
tools/benchmark_three_way.py (load_query_text()/spark_sql()/
median_stats()) rather than importing them, since this runs on a bare
Spark host with no kernel-lake repo checkout: only this script plus a
sibling queries/ directory copied alongside it (see docs/RUNBOOK.md for
the exact scp invocation -- `scp -r benchmarks/tpch/queries
runner/pyspark_query_loop.py <spark-host>:~/`).

**No live cross-engine row validation against KernelLake here**, same gap
duckdb_query_loop.py has and for the same reason: this runs on separate
infra with no access to KernelLake's live results. Each query/mode's row
count is recorded so a human can sanity-check it against the KernelLake/
DuckDB run's own row counts, not a real row-level comparison.

Usage (on the Spark host, after scp'ing this file + queries/ alongside it):
    python3 pyspark_query_loop.py --s3-bucket <bucket> --scale-factor 100 \\
        --query all --iterations 2 --modes cold,warm \\
        --output pyspark-results.json
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
QUERIES_WITH_SECOND_TABLE = {19: "part", 14: "part", 12: "orders", 3: "orders"}
QUERIES_WITH_THIRD_TABLE = {3: "customer"}

# pyspark's pip wheel bundles Hadoop 3.3.4's client but not the cloud
# connector JAR that provides org.apache.hadoop.fs.s3a.S3AFileSystem --
# reading any s3a:// path with no other config fails with a real
# ClassNotFoundException, confirmed on a live run before this was added
# (see aws_benchmark_runner.py's identical constant/comment, which this
# mirrors). hadoop-aws pulls in a matching aws-java-sdk-bundle
# transitively via spark.jars.packages' normal Maven dependency
# resolution, no separate pin needed.
HADOOP_AWS_PACKAGE = "org.apache.hadoop:hadoop-aws:3.3.4"


def load_query_text(query_number: int) -> str:
    path = QUERIES_DIR / f"q{query_number:02d}.sql"
    if not path.exists():
        raise FileNotFoundError(f"no query file for Q{query_number}: {path} -- scp queries/ alongside this script?")
    return re.sub(r"--[^\n]*\n", "\n", path.read_text())


def spark_sql(query_number: int) -> str:
    # The query files' only KernelLake-specific syntax is the
    # read_parquet('{data}')/read_parquet('{part_data}')/
    # read_parquet('{orders_data}')/read_parquet('{customer_data}')
    # table-valued-function FROM clauses -- Spark SQL has no such
    # function, so these are rewritten to plain table references against
    # temp views register_spark_views() below creates ("lineitem"/"part"/
    # "orders"/"customer"). Everything else in these query files is
    # already plain ANSI SQL both engines understand identically.
    text = load_query_text(query_number)
    rewritten = re.sub(r"read_parquet\(\s*'\{data\}'\s*\)", "lineitem", text)
    if rewritten == text:
        raise ValueError(f"Q{query_number}: no read_parquet('{{data}}') found to rewrite for Spark SQL")
    rewritten = re.sub(r"read_parquet\(\s*'\{part_data\}'\s*\)", "part", rewritten)
    rewritten = re.sub(r"read_parquet\(\s*'\{orders_data\}'\s*\)", "orders", rewritten)
    rewritten = re.sub(r"read_parquet\(\s*'\{customer_data\}'\s*\)", "customer", rewritten)
    return rewritten.strip()


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
    # s3a://, not s3:// -- Spark's DataFrameReader needs the Hadoop S3A
    # connector's own scheme (registered by the hadoop-aws jar). Mirrors
    # aws_benchmark_runner.py's identical s3_data_glob() exactly.
    prefix_map = {
        "lineitem": "lineitem-*.parquet",
        "part": "part-00000.parquet",
        "orders": "orders-*.parquet",
        "customer": "customer-00000.parquet",
    }
    tag = compression_tag(compression, compression_level)
    sf_dir = f"sf{scale_factor}-{tag}"
    return f"s3a://{bucket}/tpch-data/{sf_dir}/{prefix_map[table]}"


def median_stats(samples: list) -> dict:
    return {
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.mean(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
    }


def new_spark_session(executor_memory: str = "48g"):
    from pyspark.sql import SparkSession

    # local[*]: ONE JVM total, using all of this host's own cores as
    # "executor" threads inside it -- there is no separate executor
    # process for spark.executor.memory to size. Confirmed for real: with
    # spark.executor.memory=48g and spark.driver.memory=6g both set, the
    # actual JVM's -Xmx (checked via `ps aux | grep java`) was 6g, not
    # 48g -- spark.executor.memory is silently a no-op in local[*] mode. A
    # SparkOutOfMemoryError on Q3's join at real SF100 data is what
    # surfaced this (6g isn't enough for a 600M-row lineitem join),
    # despite the host having 64GB free. So: driver_memory (not
    # executor_memory) is the one real knob here, and it needs the large,
    # host-sized value -- executor_memory is accepted for CLI-flag back-
    # compat (--executor-memory is what RUNBOOK.md/callers already pass
    # and think of as "size this for the host") but is applied to
    # spark.driver.memory below, not spark.executor.memory, which this
    # deliberately leaves unset rather than configuring something with no
    # effect.
    builder = (
        SparkSession.builder.appName("kernellake-aws-benchmark-pyspark-local")
        .master("local[*]")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.InstanceProfileCredentialsProvider")
        .config("spark.driver.memory", executor_memory)
        # Spark's default shuffle-spill dir is java.io.tmpdir (/tmp), which
        # can be RAM-backed tmpfs with only a few GB total on some AMIs --
        # point shuffle spill at the real EBS-backed disk explicitly.
        .config("spark.local.dir", "/var/spark-tmp")
        .config("spark.jars.packages", HADOOP_AWS_PACKAGE)
    )
    return builder.getOrCreate()


def register_spark_views(
    spark, bucket: str, scale_factor: int, compression: str = "snappy",
    compression_level: int | None = None,
) -> None:
    for table in ("lineitem", "part", "orders", "customer"):
        # s3_data_glob() already returns a directly Spark-readable glob for
        # every table (including lineitem's multi-file
        # "lineitem-*.parquet") -- Spark's own reader natively supports
        # glob patterns in a path, no rewriting needed.
        spark.read.parquet(
            s3_data_glob(bucket, scale_factor, table, compression, compression_level)
        ).createOrReplaceTempView(table)


def run_pyspark_query(spark, query_number: int) -> tuple:
    import pyarrow as pa

    sql = spark_sql(query_number)
    start = time.perf_counter()
    # Spark SQL is lazy -- toPandas() is what actually triggers execution
    # (via collect() internally), and gives a ready-made bridge to pyarrow.
    pandas_df = spark.sql(sql).toPandas()
    elapsed = time.perf_counter() - start
    table = pa.Table.from_pandas(pandas_df, preserve_index=False)
    return table, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s3-bucket", required=True)
    parser.add_argument("--scale-factor", type=int, required=True)
    parser.add_argument("--compression", default="snappy", choices=["none", "snappy", "zstd"])
    parser.add_argument("--compression-level", type=int, default=None,
                         help="zstd only -- reads tpch-data/sf<N>-zstd-l<N>/ instead of the "
                              "un-suffixed tpch-data/sf<N>-zstd/ (PyArrow's own default level).")
    parser.add_argument("--query", default="all", help="Query number, or 'all' for every supported query")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--modes", default="cold,warm", help="Comma-separated: cold,warm (see module docstring)")
    parser.add_argument("--executor-memory", default="48g", help="Sized for the default m7i.4xlarge host -- lower if spark_instance_type is smaller")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    queries = list(ALL_QUERIES) if args.query == "all" else [int(args.query)]
    modes = tuple(args.modes.split(","))

    spark = new_spark_session(args.executor_memory)
    register_spark_views(spark, args.s3_bucket, args.scale_factor, args.compression, args.compression_level)

    results = []
    run_start_unix = time.time()
    for query_number in queries:
        entry: dict = {"query": query_number, "modes": {}}
        for mode in modes:
            samples = []
            row_count = None
            for rep in range(args.iterations):
                if mode == "cold" and rep == 0:
                    # A fresh SparkSession (not just fresh views) for a
                    # genuine "cold" rep -- same convention as
                    # duckdb_query_loop.py's fresh-connection-per-cold-rep
                    # and aws_benchmark_runner.py's kernellake-server
                    # restart: measures JVM/session warm-up, not just a
                    # repeated query against an already-warm one.
                    spark.stop()
                    spark = new_spark_session(args.executor_memory)
                    register_spark_views(spark, args.s3_bucket, args.scale_factor, args.compression, args.compression_level)
                table, elapsed = run_pyspark_query(spark, query_number)
                samples.append(elapsed)
                row_count = table.num_rows
            stats = median_stats(samples)
            entry["modes"][mode] = {**stats, "row_count": row_count}
            print(f"Q{query_number} ({mode}): median {stats['median_seconds']:.3f}s, {row_count} rows", file=sys.stderr)
        results.append(entry)

    spark.stop()

    output = {
        "engine": "pyspark",
        "mode": "local[*]",
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
