# Concurrency harness design: PySpark/DuckDB additions

`runner/scaling_test.py` (M4 in `RUNBOOK.md`) already covers KernelLake:
real `ThreadPoolExecutor`-based concurrent client fan-out, round-robined
across N `kernellake-server` replicas, real latency percentiles
(median/p95/p99) and completed-queries/hour. Nothing in the code requires
N>1 -- passing a single host with multiple `--concurrent-clients` already
tests something the docstring doesn't advertise: whether *one* warm server
handles concurrent requests well at all.

That single-host case is worth running before anything below. Confirmed
by reading `gpu_execution_coordinator_gpu.cpp`: `GpuExecutionCoordinator::execute()`
wraps the entire `engine.execute()` call in a plain `std::mutex` -- only
one query runs on the GPU at a time today, regardless of how many
concurrent Flight SQL connections hit the server. `scaling_test.py
--kernellake-hosts <one-ip> --concurrent-clients 4` would very likely show
near-total serialization (throughput barely above single-client, latency
growing roughly linearly past 1 concurrent client) -- a real, cheap,
already-buildable measurement, and the actual gating signal
`docs/GPU_OPTIMIZATIONS.md` opportunity #2 (dropping this mutex) is
waiting on ("wait until there's a concrete reason -- observed queueing
under real load").

No PySpark/DuckDB equivalent exists yet. This is the design for both.

## Common shape

Same worker pattern as `scaling_test.py`: each worker thread opens its
own client connection, loops firing the same query against a stop event,
records per-query latency. Aggregate into the same output schema
`concurrency_table()` (`reporting/generate_report.py`) already expects --
`concurrent_clients`, `duration_seconds`, `queries_completed`,
`queries_per_hour`, `latency_median_seconds`, `latency_p95_seconds`,
`latency_p99_seconds` -- so both plug into the existing
`aggregate_results.py --scaling-results` / report pipeline without
changes there.

## PySpark: `spark_scaling_test.py`

**Needs Spark's Thrift Server** (`sbin/start-thriftserver.sh`), not the
in-process driver `pyspark_query_loop.py` uses -- that's the actual "warm
server, concurrent clients" analog to `kernellake-server`. Python side
connects via `pyhive.hive.Connection` (HiveServer2 wire protocol, which
Spark's thrift server implements).

**`spark.scheduler.mode=FAIR` is not optional for this test.** Spark's
default FIFO scheduler serializes jobs within one SparkContext -- the same
risk class as `GpuExecutionCoordinator`'s mutex, just not documented
anywhere as prominently. Running this test without FAIR mode would
understate Spark's real concurrent capability and isn't a fair
comparison; set it explicitly, don't rely on the default.

**Infra gap**: the Spark host's init currently only starts what
`pyspark_query_loop.py`'s in-process driver needs. Starting the thrift
server is a new step (either baked into `terraform/user-data/` or run
manually over SSH before this test, matching how `scaling_test.py`
assumes replicas are already up).

## DuckDB: `duckdb_scaling_test.py`

**No new infra.** This project has no persistent DuckDB server -- each
worker thread opens its own `duckdb.connect()` in-memory instance,
querying `s3://...` directly. No shared on-disk file, so no
file-locking concerns to design around.

**Real finding either way, not just a harness gap**: every connection in
one process shares one global thread pool (`PRAGMA threads=N`). This test
measures whether concurrent queries from separate connections actually
interleave through that shared pool, or just queue behind each other --
DuckDB's genuine concurrency behavior, confirmed rather than assumed.

## Methodology note (learned the expensive way, same session)

Run each engine's concurrency test **in isolation** -- never simultaneously
with another engine on a different host. Confirmed for real this session:
two engines on separate EC2 instances, reading the same S3 bucket through
the same VPC Gateway Endpoint at once, measurably degraded *both* engines'
numbers (a uniform 7-17% slowdown across queries) purely from shared S3
bandwidth -- "different host" alone does not rule out contention when
multiple readers share the same bucket/endpoint. A concurrency test
deliberately maximizes sustained throughput, so this confound would bite
harder here than it did on the plain latency benchmark that surfaced it.

## Not yet built

Both `spark_scaling_test.py` and `duckdb_scaling_test.py` are designed
above, not written. Building either is now unblocked (no open design
questions) whenever a session has AWS budget for it -- see
`RUNBOOK.md`'s M4 section for the existing KernelLake-only version's real
cost profile (up to 8x simultaneous GPU-instance-hours) as a sizing
reference; the Spark/DuckDB single-host versions are much cheaper (one
instance each, no replica sweep needed to answer "does concurrency work
at all").
