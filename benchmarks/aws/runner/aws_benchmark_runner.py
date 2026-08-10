#!/usr/bin/env python3
"""AWS benchmark orchestrator: KernelLake (remote kernellake-server over
Flight SQL) vs. PySpark (remote standalone cluster) vs. optionally DuckDB
(in-process on this orchestrator, reading the same real S3 data), across
scale factors.

Extends tools/benchmark_three_way.py's shape rather than reimplementing it:
reuses kernellake_sql()/spark_sql()/load_query_text() (query-text
substitution), median_stats() (per-query aggregation), and
duckdb_compare.py's normalize()/rows_match() (cross-engine validation
*before* trusting any timing number -- a query whose two engines disagree
is reported as a validation failure and excluded from the timing table,
never silently timed anyway, same rule benchmark_three_way.py already
follows).

DuckDB is opt-in (--duckdb) and, unlike KernelLake/PySpark, isn't scored
in any cost table: it runs in-process on whichever host runs this script
(the Spark master instance in the standard topology), not on infra
dedicated to it, so there's no real per-instance $/hour to attribute a
query's cost against -- attributing one would imply a dedicated-infra cost
that doesn't exist. It's included purely as a third, single-node,
CPU-only latency/correctness reference point.

**"cold" vs "warm" mean something different here than in
benchmark_three_way.py**, and that difference is deliberate, not an
oversight: that script's cold mode evicts a *local file's* OS page cache
(`posix_fadvise(POSIX_FADV_DONTNEED)`) -- meaningless for data read over
the network from S3, where there's no local page cache for the data itself
to evict. Here:
  - **cold**: the first query issued against a freshly (re)started
    kernellake-server / freshly created SparkSession -- measures
    connection setup, JVM/process warm-up, and any first-query metadata/
    footer caching effects.
  - **warm**: a repeated query against an already-running server/session
    that has already served at least one prior query.
This is a real, different thing from disk-cache cold/warm, and is labeled
as such in every report this produces -- never presented as the same
concept under the same name without qualification.

Usage:
    python3 aws_benchmark_runner.py \\
        --kernellake-host 10.0.1.23 --spark-master-host 10.0.1.45 \\
        --s3-bucket kernellake-bench-666052791151-ab12cd34 --scale-factor 100 \\
        --query all --iterations 2 --output results-sf100.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

from benchmark_three_way import kernellake_sql, spark_sql, load_query_text, median_stats  # noqa: E402
from duckdb_compare import normalize, rows_match  # noqa: E402

# Q3 needs three tables (lineitem/{data}, orders/{orders_data},
# customer/{customer_data} -- confirmed against its actual query text,
# benchmarks/tpch/queries/q03.sql), not just customer -- missing the
# orders_data entry here caused a real ValueError ("Q3 needs a second
# table -- pass --orders-data") on a live run before this fix.
QUERIES_WITH_SECOND_TABLE = {19: "part_data", 14: "part_data", 12: "orders_data", 3: "orders_data"}
QUERIES_WITH_THIRD_TABLE = {3: "customer_data"}
ALL_QUERIES = (1, 3, 6, 12, 14, 19)

# The other 16 of TPC-H's 22 queries -- listed explicitly (never silently
# omitted) so every report this produces states plainly why they're
# missing, per README.md's "What this does not measure".
UNSUPPORTED_QUERIES = {
    q: "needs SQL features KernelLake doesn't support yet (correlated subqueries, EXISTS, "
    "DECIMAL, HAVING, or non-equi/multi-way joins beyond a simple INNER JOIN chain)"
    for q in (2, 4, 5, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 20, 21, 22)
}


def s3_data_glob(bucket: str, scale_factor: int, table: str, compression: str = "snappy", scheme: str = "s3a") -> str:
    # scheme differs by engine, not just style: Spark's DataFrameReader
    # needs the Hadoop S3A connector's own "s3a://" scheme (registered by
    # the hadoop-aws jar), while KernelLake's read_parquet() rejects that
    # exact scheme -- confirmed via a real adbc OperationalError
    # ("unsupported URI scheme 's3a'... expected... 's3://'...") -- and
    # needs plain "s3://" instead. This one function builds paths for
    # both engines, so the scheme must be a parameter, not hardcoded.
    # orders is multi-file too (generate_tpch.py batches it via
    # ORDERS_BATCH_ROWS the same way lineitem is batched via --files, e.g.
    # orders-00000.parquet..orders-00029.parquet at SF100) -- treating it
    # as a single "orders-00000.parquet" file (as this map previously did)
    # silently read only the first ~3% of the orders table on every query
    # that joins it (Q3, Q12). Found via a real FileNotFoundError while
    # writing generate_and_upload_iceberg_data.py's equivalent map and
    # cross-checking against a real S3 listing, not assumed.
    prefix_map = {
        "lineitem": "lineitem-*.parquet",
        "part": "part-00000.parquet",
        "orders": "orders-*.parquet",
        "customer": "customer-00000.parquet",
    }
    # snappy keeps the original, un-suffixed path (tpch-data/sf100/) for
    # backward compatibility with data already generated before this
    # dimension existed; zstd (and any other codec) gets its own path so
    # both can coexist under the same scale factor for a direct comparison.
    sf_dir = f"sf{scale_factor}" if compression == "snappy" else f"sf{scale_factor}-{compression}"
    return f"{scheme}://{bucket}/tpch-data/{sf_dir}/{prefix_map[table]}"


def iceberg_table_ref(table: str, scale_factor: int, catalog: str = "bench") -> str:
    """Iceberg table identifier for a given TPC-H table -- written by
    generate_and_upload_iceberg_data.py (lineitem partitioned by
    years(l_shipdate), orders by years(o_orderdate); part/customer
    unpartitioned) into the REST catalog registered as "bench" in
    kernellake-server.yaml's iceberg.catalogs section (see
    terraform/kernellake_instance.tf) and, on the Spark side, via the same
    name's spark.sql.catalog.bench.* config (new_spark_session() below).
    One namespace per scale factor (tpch_sf<N>) so multiple scale factors'
    Iceberg tables can coexist in the same catalog without collision --
    parallels tpch-data/sf<N>/ for the flat-file path.
    """
    return f"{catalog}.tpch_sf{scale_factor}.{table}"


def s3_bytes_for_glob(s3_client, bucket: str, key_prefix: str) -> int:
    """S3-native analog of benchmark_three_way.py's local bytes_for_glob()
    -- sums real object sizes via ListObjectsV2 rather than glob.glob(),
    since there's no local filesystem to glob over."""
    total = 0
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for obj in page.get("Contents", []):
            total += obj["Size"]
    return total


def s3_table_key_prefix(scale_factor: int, table: str, compression: str = "snappy") -> str:
    file_prefix_map = {"lineitem": "lineitem-", "part": "part-", "orders": "orders-", "customer": "customer-"}
    sf_dir = f"sf{scale_factor}" if compression == "snappy" else f"sf{scale_factor}-{compression}"
    return f"tpch-data/{sf_dir}/{file_prefix_map[table]}"


def table_bytes_scanned(s3_client, s3fs, bucket: str, scale_factor: int, table: str, compression: str) -> dict:
    """Real compressed (S3 object size) and uncompressed bytes for a
    table's files at this scale factor -- both real, never estimated.
    Uncompressed comes from each Parquet file's own footer metadata
    (total_uncompressed_size, summed across every row group/column) --
    a footer-only read (a few KB per file over the network), not a full
    file download, matching this project's own "never fabricate
    precision" rule (see s3_bytes_for_glob()'s own real-listing
    convention). Flat-format only -- Iceberg mode would need the
    equivalent from manifest file-level stats instead, not implemented
    here yet."""
    import pyarrow.parquet as pq

    key_prefix = s3_table_key_prefix(scale_factor, table, compression)
    compressed_bytes = 0
    uncompressed_bytes = 0
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for obj in page.get("Contents", []):
            compressed_bytes += obj["Size"]
            meta = pq.ParquetFile(f"{bucket}/{obj['Key']}", filesystem=s3fs).metadata
            for rg in range(meta.num_row_groups):
                row_group = meta.row_group(rg)
                for col in range(row_group.num_columns):
                    uncompressed_bytes += row_group.column(col).total_uncompressed_size
    return {"compressed_bytes": compressed_bytes, "uncompressed_bytes": uncompressed_bytes}


def connect_kernellake(host: str, port: int = 31337):
    import adbc_driver_flightsql.dbapi as flightsql

    return flightsql.connect(f"grpc://{host}:{port}")


def restart_kernellake_server(ssh_host: str, ssh_key_path: str) -> None:
    """Real server restart for a genuine "cold" kernellake-server rep --
    over SSH, since the runner talks to the remote host's Flight SQL port,
    not a local subprocess like benchmark_three_way.py's own
    start_kernellake_server()."""
    import subprocess

    subprocess.run(
        [
            "ssh", "-i", ssh_key_path, "-o", "StrictHostKeyChecking=no",
            f"ubuntu@{ssh_host}",
            "cd /opt/kernellake-bench && sudo docker compose restart kernellake-server",
        ],
        check=True,
        capture_output=True,
    )
    # Poll the Flight SQL port rather than a fixed sleep -- same rationale
    # as benchmark_three_way.py's own start_kernellake_server().
    import socket

    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((ssh_host, 31337), timeout=1.0):
                return
        except OSError:
            time.sleep(1.0)
    raise RuntimeError(f"kernellake-server on {ssh_host} did not come back up within 60s of restart")


def run_kernellake_query(cursor, query_number: int, globs: dict, table_format: str = "flat") -> tuple:
    sql = kernellake_sql(
        query_number, globs["data"], globs.get("part_data"), globs.get("orders_data"), globs.get("customer_data")
    )
    if table_format == "iceberg":
        # The query files' FROM clauses are fixed text (read_parquet(...)) --
        # kernellake_sql() only substitutes the {data}/{part_data}/etc.
        # argument, not the function name itself. Rather than fork the
        # shared query files (also used by the local, non-AWS benchmark
        # suite) or duplicate kernellake_sql()'s substitution logic, swap
        # the function name after the fact: globs[...] already holds
        # Iceberg table identifiers (iceberg_table_ref()) in this mode, not
        # S3 globs, so read_parquet('bench.tpch_sf100.lineitem') simply
        # isn't valid syntax -- read_iceberg(...) is.
        sql = re.sub(r"\bread_parquet\(", "read_iceberg(", sql)
    start = time.perf_counter()
    cursor.execute(sql)
    table = cursor.fetch_arrow_table()
    elapsed = time.perf_counter() - start
    return table, elapsed


def run_pyspark_query(spark, query_number: int) -> tuple:
    import pyarrow as pa

    sql = spark_sql(query_number)
    start = time.perf_counter()
    pandas_df = spark.sql(sql).toPandas()
    elapsed = time.perf_counter() - start
    table = pa.Table.from_pandas(pandas_df, preserve_index=False)
    return table, elapsed


def new_duckdb_connection(region: str):
    """A fresh, S3-ready DuckDB connection: httpfs for s3:// reads, plus
    the aws extension's CREDENTIAL_CHAIN provider so it resolves this
    instance's real IAM role the same way boto3/the AWS SDK would --
    consistent with how kernellake-server/Spark authenticate to S3 here
    (InstanceProfileCredentialsProvider), not a static access key. DuckDB
    accepts the *same* 's3://' scheme and the *same* substituted SQL
    KernelLake does (see run_duckdb_query()'s own comment), unlike Spark's
    's3a://' + separate SQL rewrite.
    """
    import duckdb

    con = duckdb.connect()
    con.install_extension("httpfs")
    con.load_extension("httpfs")
    con.install_extension("aws")
    con.load_extension("aws")
    con.sql("CREATE OR REPLACE SECRET (TYPE S3, PROVIDER CREDENTIAL_CHAIN)")
    con.sql(f"SET s3_region = '{region}'")
    return con


def run_duckdb_query(con, query_number: int, globs: dict) -> tuple:
    # Same substituted SQL as KernelLake's own read_parquet(...) calls --
    # DuckDB accepts that syntax natively, no rewrite needed (see
    # tools/duckdb_compare.py's equivalent comment for the local benchmark).
    sql = kernellake_sql(
        query_number, globs["data"], globs.get("part_data"), globs.get("orders_data"), globs.get("customer_data")
    )
    start = time.perf_counter()
    # .arrow() returns a RecordBatchReader (lazy), not a materialized
    # pa.Table -- .read_all() is what actually pulls results and is also
    # what triggers execution to complete, same as duckdb_compare.py's
    # run_duckdb() already does for the local benchmark.
    table = con.sql(sql).arrow().read_all()
    elapsed = time.perf_counter() - start
    return table, elapsed


def spark_scan_metrics(spark) -> dict | None:
    """Best-effort scan-phase input bytes/duration for the *most recently
    completed* stage, via Spark's own REST API. This is a **supporting**
    metric (see the module docstring's "Primary goal" framing) -- a
    failure here degrades to None, it never fails the whole benchmark run,
    since scan throughput is explanatory detail, not the headline number.

    Spark's REST API is served by the *driver* (http://localhost:4040 by
    default) -- since new_spark_session() creates a client-mode session
    (the SparkSession lives in this orchestrator process, not on the
    remote cluster), "localhost" here really does mean this process, not
    the Spark master/worker hosts.
    """
    try:
        import requests

        app_id = spark.sparkContext.applicationId
        base = f"http://localhost:4040/api/v1/applications/{app_id}/stages"
        stages = requests.get(base, timeout=5).json()
        completed = [s for s in stages if s.get("status") == "COMPLETE"]
        if not completed:
            return None
        latest = max(completed, key=lambda s: s.get("stageId", 0))
        return {
            "input_bytes": latest.get("inputBytes"),
            "executor_run_time_ms": latest.get("executorRunTime"),
        }
    except Exception as e:  # noqa: BLE001 -- deliberately broad: any failure here just means "no supporting metric this time", never a run-failing error.
        print(f"WARNING: spark_scan_metrics() failed (non-fatal, supporting metric only): {e}", file=sys.stderr)
        return None


ICEBERG_SPARK_PACKAGES = "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.6.1,org.apache.iceberg:iceberg-aws-bundle:1.6.1"
# pyspark's pip wheel bundles Hadoop 3.3.4's client (confirmed via the
# actual jars/ directory on a real instance: hadoop-client-api-3.3.4.jar/
# hadoop-client-runtime-3.3.4.jar) but NOT the cloud connector JAR that
# provides org.apache.hadoop.fs.s3a.S3AFileSystem -- reading any s3a://
# path with no other config fails with a real ClassNotFoundException,
# confirmed on a live run before this was added. hadoop-aws pulls in a
# matching aws-java-sdk-bundle transitively via spark.jars.packages'
# normal Maven dependency resolution, no separate pin needed.
HADOOP_AWS_PACKAGE = "org.apache.hadoop:hadoop-aws:3.3.4"


def new_spark_session(
    master_host: str,
    iceberg_catalog_uri: str | None = None,
    iceberg_warehouse: str | None = None,
    executor_memory: str = "48g",
    driver_memory: str = "6g",
):
    from pyspark.sql import SparkSession

    # master_host="local" runs Spark entirely in-process (local[*]: one
    # JVM, using all local cores as "executors") instead of connecting to
    # a real standalone cluster -- for single-node testing (the
    # spark_worker_count=0 topology terraform/spark_cluster.tf supports),
    # this sidesteps an entire class of standalone-mode-specific problems
    # confirmed for real on a live run: Master/Worker daemon JMX-port
    # collisions when co-located, and SPARK_LOCAL_DIRS (a Worker-process
    # environment variable) silently overriding this function's own
    # spark.local.dir config below, entirely undetected until a
    # shuffle-heavy query filled a RAM-backed /tmp. Same driver-side
    # config (executor_memory/spark.local.dir/etc.) still applies in
    # local[*] mode, just without any separate daemon to get out of sync
    # with it. Only appropriate for single-node runs -- the cost-matched
    # multi-instance topology (spark_worker_count > 0) still needs a real
    # standalone cluster across multiple real EC2 instances.
    master_url = "local[*]" if master_host == "local" else f"spark://{master_host}:7077"

    packages = [HADOOP_AWS_PACKAGE]
    builder = (
        SparkSession.builder.appName("kernellake-aws-benchmark")
        .master(master_url)
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.InstanceProfileCredentialsProvider")
        # Never set before -- Spark defaults spark.executor.memory to 1g,
        # nowhere near enough for a SF100-scale join (600M lineitem rows).
        # Confirmed for real: executors repeatedly died with exit code 52
        # (Spark's own ExecutorExitCode.OOM) on Q12's join, each retry
        # immediately OOMing again on a fresh executor with the same
        # inadequate default, until Spark gave up after its max-retries
        # limit -- a real death spiral, not something that self-heals.
        # Defaults (48g/6g) assume the original m7i.4xlarge workers (64GB
        # RAM); --spark-executor-memory/--spark-driver-memory below MUST
        # be sized down for a smaller/single-node topology -- confirmed
        # for real that leaving these at the default against a 16GB
        # m7i.xlarge self-sufficient master hangs forever ("Initial job
        # has not accepted any resources"), not OOMs, since the executor
        # can never even start: Spark just keeps waiting for resources the
        # worker can never offer, with no error and no timeout.
        .config("spark.executor.memory", executor_memory)
        .config("spark.driver.memory", driver_memory)
        # Spark's default shuffle-spill dir is java.io.tmpdir (/tmp) --
        # fine on the original m7i.4xlarge/m7i.xlarge topology if /tmp
        # happens to be disk-backed there, but confirmed for real on this
        # topology that /tmp is RAM-backed tmpfs with only ~8GB total
        # (df -h showed 90GB free on the real root EBS volume vs. a hard
        # "No space left on device" shuffle failure on /tmp well before
        # that): a SF100 3-way join's shuffle spill blew straight through
        # it. Point shuffle spill at the real disk explicitly rather than
        # relying on whatever /tmp happens to be on a given AMI/instance.
        .config("spark.local.dir", "/var/spark-tmp")
    )
    if iceberg_catalog_uri:
        # "bench" here must match kernellake-server.yaml's iceberg.catalogs
        # key (terraform/kernellake_instance.tf) -- same REST catalog
        # instance, same registered name, so both engines address the same
        # tables via bench.<namespace>.<table>. io-impl=S3FileIO (not the
        # default HadoopFileIO) so Iceberg's own S3 reads go through the
        # AWS SDK for Java v2 and pick up this instance's IAM role the same
        # way iceberg-catalog-init.sh's REST catalog container already
        # does, not spark.hadoop.fs.s3a.* (a separate, s3a-connector-only
        # credential path that Iceberg's own S3FileIO doesn't use).
        packages.append(ICEBERG_SPARK_PACKAGES)
        builder = (
            builder.config("spark.sql.catalog.bench", "org.apache.iceberg.spark.SparkCatalog")
            .config("spark.sql.catalog.bench.type", "rest")
            .config("spark.sql.catalog.bench.uri", iceberg_catalog_uri)
            .config("spark.sql.catalog.bench.warehouse", iceberg_warehouse)
            .config("spark.sql.catalog.bench.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
        )
    builder = builder.config("spark.jars.packages", ",".join(packages))
    return builder.getOrCreate()


def register_spark_views(
    spark, bucket: str, scale_factor: int, compression: str = "snappy",
    table_format: str = "flat", iceberg_catalog: str = "bench",
) -> None:
    for table in ("lineitem", "part", "orders", "customer"):
        if table_format == "iceberg":
            spark.table(iceberg_table_ref(table, scale_factor, iceberg_catalog)).createOrReplaceTempView(table)
        else:
            # s3_data_glob() already returns a directly Spark-readable glob
            # for every table (including lineitem's multi-file
            # "lineitem-*.parquet") -- Spark's own reader natively supports
            # glob patterns in a path, no rewriting needed.
            spark.read.parquet(s3_data_glob(bucket, scale_factor, table, compression)).createOrReplaceTempView(table)


def benchmark_query(
    query_number: int,
    globs: dict,
    kernellake_cursor,
    spark,
    duckdb_con,
    modes: tuple,
    iterations: int,
    kernellake_ssh_host: str | None,
    ssh_key_path: str | None,
    spark_master_host: str | None,
    s3_bucket: str,
    scale_factor: int,
    compression: str = "snappy",
    table_format: str = "flat",
    iceberg_catalog_uri: str | None = None,
    iceberg_warehouse: str | None = None,
    aws_region: str = "us-east-1",
    spark_executor_memory: str = "48g",
    spark_driver_memory: str = "6g",
) -> tuple[dict, object, object]:
    """Returns (result, spark, duckdb_con) -- both session objects are
    returned because cold mode replaces each with a fresh one (see module
    docstring's "cold vs warm" definition); the caller must use the
    returned objects for subsequent queries, not the ones it passed in."""
    result: dict = {"query": query_number, "modes": {}}
    for mode in modes:
        kl_samples, spark_samples, duckdb_samples = [], [], []
        validated = None
        validated_duckdb = None
        for rep in range(iterations):
            is_first_rep = rep == 0
            if mode == "cold" and is_first_rep and kernellake_ssh_host and ssh_key_path:
                restart_kernellake_server(kernellake_ssh_host, ssh_key_path)
            if mode == "cold" and is_first_rep and spark is not None:
                spark.stop()
                spark = new_spark_session(
                    spark_master_host, iceberg_catalog_uri, iceberg_warehouse,
                    spark_executor_memory, spark_driver_memory,
                )
                register_spark_views(spark, s3_bucket, scale_factor, compression, table_format)
            if mode == "cold" and is_first_rep and duckdb_con is not None:
                duckdb_con.close()
                duckdb_con = new_duckdb_connection(aws_region)

            kl_table, kl_elapsed = run_kernellake_query(kernellake_cursor, query_number, globs, table_format)
            kl_samples.append(kl_elapsed)

            if duckdb_con is not None:
                duckdb_table, duckdb_elapsed = run_duckdb_query(duckdb_con, query_number, globs)
                duckdb_samples.append(duckdb_elapsed)
                if validated_duckdb is None:
                    validated_duckdb = rows_match(normalize(kl_table), normalize(duckdb_table))

            if spark is not None:
                spark_table, spark_elapsed = run_pyspark_query(spark, query_number)
                spark_samples.append(spark_elapsed)
                if validated is None:
                    validated = rows_match(normalize(kl_table), normalize(spark_table))
                if is_first_rep:
                    scan_metrics = spark_scan_metrics(spark)
                    if scan_metrics:
                        result.setdefault("spark_scan_metrics", {})[mode] = scan_metrics

        entry = {"kernellake": median_stats(kl_samples)}
        if spark_samples:
            entry["pyspark"] = median_stats(spark_samples)
            entry["results_match"] = validated
            if validated:
                entry["latency_speedup_ratio"] = entry["pyspark"]["median_seconds"] / entry["kernellake"]["median_seconds"]
            elif validated is False:
                print(f"WARNING: Q{query_number} ({mode}): KernelLake and PySpark results do NOT match -- "
                      "excluding this query/mode's timing from any headline number.", file=sys.stderr)
        if duckdb_samples:
            entry["duckdb"] = median_stats(duckdb_samples)
            entry["results_match_duckdb"] = validated_duckdb
            if validated_duckdb:
                entry["latency_speedup_ratio_duckdb"] = (
                    entry["duckdb"]["median_seconds"] / entry["kernellake"]["median_seconds"]
                )
            elif validated_duckdb is False:
                print(f"WARNING: Q{query_number} ({mode}): KernelLake and DuckDB results do NOT match -- "
                      "excluding this query/mode's DuckDB timing from any headline number.", file=sys.stderr)
        result["modes"][mode] = entry
    return result, spark, duckdb_con


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kernellake-host", required=True)
    parser.add_argument("--kernellake-port", type=int, default=31337)
    parser.add_argument("--kernellake-ssh-key", default=None, help="Path to the SSH private key, for real cold-mode server restarts. Omit to skip real restarts (cold mode then only means \"first rep of this run\", not \"freshly restarted server\").")
    parser.add_argument("--spark-master-host", default=None,
                         help="Omit to run KernelLake-only (no PySpark comparison, no cross-engine validation). "
                              "'local' runs Spark in-process (local[*]) instead of connecting to a real "
                              "standalone cluster -- see new_spark_session()'s own comment; appropriate for a "
                              "single-node run (spark_worker_count=0), not the cost-matched multi-instance one.")
    parser.add_argument("--duckdb", action="store_true",
                         help="Also run each query through DuckDB, in-process on this orchestrator host, "
                              "against the same real S3 data (via httpfs + the aws extension's IAM "
                              "instance-role credential chain). Flat table format only -- see "
                              "new_duckdb_connection()'s own comment for why it's excluded from cost "
                              "tables. Opt-in: off by default so existing invocations are unaffected.")
    parser.add_argument("--s3-bucket", required=True)
    parser.add_argument("--aws-region", default="us-east-1", help="Used for DuckDB's s3_region setting (--duckdb only).")
    parser.add_argument("--scale-factor", type=int, required=True)
    parser.add_argument("--compression", default="snappy", choices=["snappy", "zstd"],
                         help="Which generate_and_upload_data.sh --compression run to read "
                              "(snappy reads tpch-data/sf<N>/, zstd reads tpch-data/sf<N>-zstd/). "
                              "Ignored when --table-format iceberg (Iceberg tables always write zstd).")
    parser.add_argument("--table-format", default="flat", choices=["flat", "iceberg"],
                         help="flat: plain Parquet files via read_parquet()/spark.read.parquet() glob "
                              "(the --compression dimension applies). iceberg: real Iceberg tables "
                              "(lineitem/orders partitioned by years(shipdate)/years(orderdate)) via "
                              "read_iceberg()/the Spark Iceberg catalog -- see "
                              "generate_and_upload_iceberg_data.py and iceberg_table_ref(). Needs "
                              "--iceberg-catalog-uri and --iceberg-warehouse.")
    parser.add_argument("--iceberg-catalog-uri", default=None,
                         help="e.g. http://<iceberg-catalog-instance-ip>:8181 (terraform output iceberg_catalog_uri). "
                              "Required when --table-format iceberg and --spark-master-host is given -- "
                              "KernelLake's own catalog config lives server-side in kernellake-server.yaml "
                              "already, this is only for configuring Spark's matching catalog.")
    parser.add_argument("--iceberg-warehouse", default=None,
                         help="e.g. s3://<bucket>/warehouse/ -- required alongside --iceberg-catalog-uri.")
    parser.add_argument("--query", default="all",
                         help="'all', a single query number (e.g. 3), or a comma-separated list "
                              "(e.g. 1,6,12,14,19 -- to run everything except a specific query, "
                              "such as one that's currently GPU-memory-blocked)")
    # 2, not the earlier 3/5: a full --query all run (6 queries x 2 modes x
    # N iterations, cold mode restarting the server/session on every rep 1)
    # was taking too long in practice at real EC2 GPU-hour cost -- 2 still
    # gives a median across 2 samples per mode, just a cheaper/faster run.
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--modes", default="cold,warm")
    parser.add_argument("--spark-executor-memory", default="48g",
                         help="Must fit within a single Spark worker's real memory (spark_master_public_ip:8080's "
                              "own JSON reports each worker's 'memory' field) -- defaults assume the original "
                              "m7i.4xlarge workers (64GB). Too high hangs forever ('Initial job has not accepted "
                              "any resources'), not an OOM -- confirmed for real against a smaller/self-sufficient "
                              "single-node topology (spark_worker_count=0), see new_spark_session()'s own comment.")
    parser.add_argument("--spark-driver-memory", default="6g")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    modes = tuple(args.modes.split(","))
    queries = ALL_QUERIES if args.query == "all" else tuple(int(q) for q in args.query.split(","))
    if args.table_format == "iceberg" and args.spark_master_host and not (args.iceberg_catalog_uri and args.iceberg_warehouse):
        print("--table-format iceberg with --spark-master-host requires --iceberg-catalog-uri and --iceberg-warehouse", file=sys.stderr)
        return 1
    if args.duckdb and args.table_format == "iceberg":
        print("--duckdb only supports --table-format flat (no Iceberg REST catalog wiring for DuckDB here)", file=sys.stderr)
        return 1

    kernellake_conn = connect_kernellake(args.kernellake_host, args.kernellake_port)
    kernellake_cursor = kernellake_conn.cursor()

    spark = None
    if args.spark_master_host:
        spark = new_spark_session(
            args.spark_master_host, args.iceberg_catalog_uri, args.iceberg_warehouse,
            args.spark_executor_memory, args.spark_driver_memory,
        )
        register_spark_views(spark, args.s3_bucket, args.scale_factor, args.compression, args.table_format)

    duckdb_con = new_duckdb_connection(args.aws_region) if args.duckdb else None

    # Real compressed/uncompressed bytes scanned per table, cached (many
    # queries share lineitem) -- flat format only, see table_bytes_scanned()'s
    # own comment for why Iceberg mode isn't covered here yet.
    table_bytes_cache: dict[str, dict] = {}
    s3_client_for_bytes = None
    s3fs_for_bytes = None
    if args.table_format == "flat":
        import boto3
        import pyarrow.fs as pafs

        s3_client_for_bytes = boto3.client("s3")
        s3fs_for_bytes = pafs.S3FileSystem()

    results = []
    for query_number in queries:
        def table_ref(table: str) -> str:
            if args.table_format == "iceberg":
                return iceberg_table_ref(table, args.scale_factor)
            # scheme="s3" -- this builds KernelLake's globs specifically;
            # register_spark_views() above builds Spark's own separately
            # (scheme="s3a", the default) since the two engines need
            # different URI schemes for the same underlying S3 objects.
            return s3_data_glob(args.s3_bucket, args.scale_factor, table, args.compression, scheme="s3")

        query_tables = ["lineitem"]
        globs = {"data": table_ref("lineitem")}
        if query_number in QUERIES_WITH_SECOND_TABLE:
            key = QUERIES_WITH_SECOND_TABLE[query_number]
            table = {"part_data": "part", "orders_data": "orders"}[key]
            globs[key] = table_ref(table)
            query_tables.append(table)
        if query_number in QUERIES_WITH_THIRD_TABLE:
            key = QUERIES_WITH_THIRD_TABLE[query_number]
            globs[key] = table_ref("customer")
            query_tables.append("customer")

        print(f"=== Q{query_number} ===", file=sys.stderr)
        query_result, spark, duckdb_con = benchmark_query(
            query_number, globs, kernellake_cursor, spark, duckdb_con, modes, args.iterations,
            args.kernellake_host, args.kernellake_ssh_key,
            args.spark_master_host, args.s3_bucket, args.scale_factor, args.compression,
            args.table_format, args.iceberg_catalog_uri, args.iceberg_warehouse, args.aws_region,
            args.spark_executor_memory, args.spark_driver_memory,
        )

        if args.table_format == "flat":
            compressed_total = 0
            uncompressed_total = 0
            for table in query_tables:
                if table not in table_bytes_cache:
                    print(f"  (computing real bytes-scanned for {table}...)", file=sys.stderr)
                    table_bytes_cache[table] = table_bytes_scanned(
                        s3_client_for_bytes, s3fs_for_bytes, args.s3_bucket,
                        args.scale_factor, table, args.compression,
                    )
                compressed_total += table_bytes_cache[table]["compressed_bytes"]
                uncompressed_total += table_bytes_cache[table]["uncompressed_bytes"]
            query_result["bytes_scanned"] = {
                "compressed_bytes": compressed_total,
                "uncompressed_bytes": uncompressed_total,
                "tables": query_tables,
            }

        results.append(query_result)

    report = {
        "scale_factor": args.scale_factor,
        "compression": args.compression,
        "table_format": args.table_format,
        "s3_bucket": args.s3_bucket,
        "queries": results,
        "unsupported_queries": UNSUPPORTED_QUERIES,
        "cold_warm_definition": (
            "cold: first query against a freshly-restarted kernellake-server / freshly-created "
            "SparkSession / freshly-created DuckDB connection (--duckdb). warm: repeated query "
            "against an already-running server/session/connection. NOT the same concept as "
            "local-disk OS-page-cache cold/warm (there is no local page cache for S3-resident data "
            "on the client side) -- see this script's own module docstring."
        ),
    }
    Path(args.output).write_text(json.dumps(report, indent=2, default=str))
    print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
