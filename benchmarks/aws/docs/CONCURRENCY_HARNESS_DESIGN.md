# Concurrency harness design: PySpark/DuckDB additions

`runner/scaling_test.py` (M4 in `RUNBOOK.md`) already covers KernelLake:
real `ThreadPoolExecutor`-based concurrent client fan-out, round-robined
across N `kernellake-server` replicas, real latency percentiles
(median/p95/p99) and completed-queries/hour. Nothing in the code requires
N>1 -- passing a single host with multiple `--concurrent-clients` already
tests something the docstring doesn't advertise: whether *one* warm server
handles concurrent requests well at all.

That single-host case is worth running before anything below -- though
what it should now be expected to show has changed twice since this
design was first written. As of 2026-08-17, `gpu_execution_coordinator_gpu.cpp`
did wrap the entire `engine.execute()` call in a plain `std::mutex` --
only one query ran on the GPU at a time, regardless of concurrent client
count, and that finding (real observed queueing at SF1000, see
`RUNBOOK.md`'s M4 section) was exactly the gating signal
`docs/GPU_OPTIMIZATIONS.md` opportunity #2 was waiting on. Opt #2
(2026-08-21) replaced that mutex with a `std::counting_semaphore` capped
by `EngineSection::max_concurrent_gpu_queries` (default 2) -- bounded, not
unconditional, concurrency. The Multi-GPU Tier 1 change (2026-08-24, see
`docs/GPU_OPTIMIZATIONS.md`'s "Multi-GPU Tier 1 implemented" section)
went further: `GpuExecutionCoordinator` now builds one `RmmEnvironment` +
semaphore pair per visible CUDA device and round-robins queries across
them, so an N-GPU node gets up to N x that per-device cap in total
concurrent throughput. On a single-GPU host (`cudaGetDeviceCount() == 1`,
true for this project's own dev GPU) this degrades back to exactly one
semaphore capped at `max_concurrent_gpu_queries` -- so `scaling_test.py
--kernellake-hosts <one-ip> --concurrent-clients 4` today should be
expected to show real (if capped) concurrency rather than near-total
serialization; worth re-running to confirm the semaphore cap behaves as
expected under real load, since the 2026-08-17 numbers this section
originally cited predate both fixes.

No PySpark/DuckDB equivalent exists yet. This is the design for both.

## Cache should be ON for this test -- the opposite of the single-query benchmark

Every latency benchmark this session ran deliberately cold/no-cache
(worst case: every query hits genuinely fresh, uncached S3 data). The
concurrency test is structurally different and needs the opposite
setting: each worker fires the *same* query repeatedly against one warm
connection/session for the whole test duration, so after the first
iteration every repeat is against data a real system would obviously have
cached by then.

Running this test cold would measure "does concurrent *cold S3 access*
scale" -- a real but different question, and one already confounded by
the shared-S3-bandwidth-contention effect documented below, which would
swamp the actual signal this test exists to surface (does concurrent
*execution* scale -- i.e. does `GpuExecutionCoordinator`'s per-device
semaphore cap, Spark's scheduler, or DuckDB's shared thread pool bottleneck
concurrent queries).
Warm/cached isolates that question properly.

Concretely: `duckdb_scaling_test.py` passes `enable_cache=True` to
`new_duckdb_connection()` (opposite of `duckdb_query_loop.py`'s own
cold-benchmark default). On the KernelLake side, this means
`kernellake-server.yaml`'s `storage.cache.enabled` needs to be `true` for
this test specifically -- a deployment-time config choice
`scaling_test.py` itself has no control over, unlike the disabled-cache
setup this session used for every cold-latency run. Spark has no
separate cache toggle to flip; whatever benefit it gets from repeat reads
comes from the OS page cache naturally, nothing to configure.

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
risk class `GpuExecutionCoordinator`'s bounded semaphore is designed
around, just not documented anywhere as prominently. Running this test
without FAIR mode would
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

## Built, not yet run for real

`runner/spark_scaling_test.py` and `runner/duckdb_scaling_test.py` exist
now, matching the design above (reuse each engine's existing query-loop
script's helpers rather than duplicating them; DuckDB caps
`SET threads TO <cores/N>` per connection to avoid oversubscription;
Spark registers real external tables over the Thrift connection since its
Python client can't reach the DataFrame API). Neither has been run
against real infra yet -- see `RUNBOOK.md`'s M4 section for the exact
invocations once budget allows. See `RUNBOOK.md`'s M4 section for the
existing KernelLake-only version's real cost profile (up to 8x
simultaneous GPU-instance-hours) as a sizing reference; the Spark/DuckDB
single-host versions are much cheaper (one instance each, no replica
sweep needed to answer "does concurrency work at all").
