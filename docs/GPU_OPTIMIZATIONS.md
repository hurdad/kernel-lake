# GPU optimization opportunities

Notes from a review of the current CUDA/cudf execution path
(`src/execution_gpu/`, `src/memory/rmm_environment.cpp`,
`src/api/query_engine_execute_gpu.cpp`, `src/server/gpu_execution_coordinator_gpu.cpp`).
Nothing here is implemented yet -- this is a prioritized list of
opportunities plus the tradeoff each one carries, to work from before
picking one.

## Current state (baseline, for context)

- Device memory goes through RMM: a `limiting_resource_adaptor` ->
  `statistics_resource_adaptor` -> `cuda_async_memory_resource` (or
  `pool_memory_resource`) stack, one instance per process
  (`RmmEnvironment`, `src/memory/rmm_environment.cpp`).
- No pinned (page-locked) host memory anywhere in the tree -- confirmed by
  grep across `src/`, `include/`, `tests/`. Host<->device transfers rely on
  whatever libcudf does internally by default.
- Execution is stream-based: one `CudaStream` per query
  (`query_engine_execute_gpu.cpp`), threaded through every operator via
  `ExecutionContext::stream`. Every cudf call in every operator passes it
  explicitly -- no operator falls back to the legacy default stream.
- `GpuExecutionCoordinator` (`gpu_execution_coordinator_gpu.cpp`) holds a
  mutex around `execute()`: only one query runs at a time, server-wide.
  Streams give async/overlapping kernel execution *within* a query, not
  concurrent queries.
- `CudaStream` (`cuda_utils.cpp`) is created via plain `cudaStreamCreate()`,
  not `cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)` -- despite the
  header comment above it calling it a "non-blocking" stream, which is
  wrong. This is load-bearing, not just stale wording: `to_arrow_host()` /
  `from_arrow()` in `arrow_bridge.cpp` don't take a stream argument, so they
  run on CUDA's default/null stream. That only reads correct data because a
  blocking-flagged stream implicitly synchronizes with the null stream
  (null-stream ops wait for prior work on any blocking stream to finish).
  If the flag were "fixed" to actually be non-blocking, the D2H copy in
  `to_arrow_record_batch()` would race against still-in-flight kernels on
  `context.stream`.

## Opportunities, roughly by expected payoff vs. effort

1. **Pinned host memory for the device-to-host path.** Every output batch
   currently pays a pageable-memory `cudaMemcpy` inside
   `cudf::to_arrow_host()` (`arrow_bridge.cpp`). Wiring up
   `rmm::mr::pinned_memory_resource` (or `cudf::set_pinned_memory_resource`)
   there would speed up that copy meaningfully, especially for large result
   sets.
   - Tradeoff: pinned memory is a scarcer, harder-to-size resource than
     pool device memory -- over-allocating it hurts the whole system
     (host RAM pressure, page-lock limits), not just this process.

2. **Drop the single-query mutex in `GpuExecutionCoordinator`.** The server
   currently serializes on one query at a time, so multi-stream async
   execution never gets to overlap across queries.
   - Tradeoff: `RmmEnvironment` is one shared pool/stats/limiter for the
     whole process. True concurrency needs either per-query sub-pools or a
     thread-safe stream-ordered suballocator -- a real architecture change,
     not a tweak.

3. **Overlap Parquet decode with compute across multiple streams** for
   multi-file/multi-fragment scans. Everything currently funnels through
   the one per-query stream sequentially.
   - Tradeoff: adds real complexity (stream sync points, multiple
     in-flight allocations to track against the query's memory limit).

4. **Retune `pass_read_limit_bytes`** (currently
   `rmm_environment.query_memory_limit_bytes() / 4`, see
   `query_engine_execute_gpu.cpp`). That divisor was set from one SF100
   measurement (TPC-H-derived Q1 at SF100); profiling more query shapes
   could recover headroom.
   - Tradeoff: low risk, but only matters if a workload is actually
     memory-constrained today -- easy to spend time tuning a number that
     isn't the bottleneck.

5. **Profile with `nsys` before acting on any of the above.** NVTX ranges
   are already wired in (`config_.profiling.nvtx`), but there's no evidence
   of an actual profiling session against real query shapes confirming
   where time goes.
   - Tradeoff: none, really -- this should come first. The D2H copy (#1)
     is the obvious *theoretical* win but might be a small fraction of
     wall time next to kernel execution itself; profiling first avoids
     optimizing the wrong thing.

## Recommendation

Start with #5 (profile) to confirm #1 (pinned memory) is actually worth
doing before investing in it. #2 (concurrent queries) is the biggest
structural change and should wait until there's a concrete reason
(observed queueing under real load) to take on the RMM-sharing redesign it
requires.

## Follow-up: GPU memory growing across benchmark runs

Observed during AWS GPU-instance benchmark testing (`benchmarks/aws/`):
device memory usage climbed with each successive benchmark run. Root cause
investigation below; not yet fixed.

**Mechanism.** `benchmarks/aws/runner/aws_benchmark_runner.py` talks to a
long-lived `kernellake-server` process over Flight SQL
(`flightsql.connect(f"grpc://{host}:{port}")`), not a fresh CLI process per
query. Server-side, `GpuExecutionCoordinator`
(`gpu_execution_coordinator_gpu.cpp:15`) constructs **one `RmmEnvironment`
for the whole process lifetime**, and every query reuses it through a
mutex-serialized `execute()` call -- unlike the CLI/one-shot path
(`QueryEngine::execute(sql)`, used by `src/cli/benchmark_tpch_command.cpp`),
which constructs and destroys a fresh `RmmEnvironment` every call. The
default allocator (`memory.use_async_allocator: true` in
`config/kernellake.yaml`) backs that long-lived environment with
`rmm::mr::cuda_async_memory_resource`, a `cudaMallocAsync`-style
stream-ordered pool. Pools like this cache freed device memory for reuse
rather than returning it to the driver immediately, so `nvidia-smi` for a
long-lived process reflects the pool's high-water mark, not live in-use
bytes -- exactly what repeated benchmark runs against a persistent server
would show.

**Ruled out** (so this isn't a leak, as far as KernelLake's own code goes):
- No raw `cudaMalloc`/`cudaFree` anywhere in `src/`/`include/` -- every
  device allocation goes through the tracked RMM resource.
- No `static`/`thread_local` device-memory-holding state in
  `src/execution_gpu/` -- operator trees are rebuilt fresh per query
  (`build_operator_tree()` in `query_engine_execute_gpu.cpp:123`) and fully
  destroyed before that call returns, even on the server path.
- `peak_gpu_memory_bytes` is correctly scoped per query (RMM's
  `statistics_resource_adaptor::push_counters()`/`pop_counters()` isolates
  each call), so it isn't a reporting artifact.

**Still open:** whether the growth plateaus (benign pool caching, consistent
with everything above) or is unbounded across many runs (would point to an
actual leak, worth reaching for `compute-sanitizer` next). Not yet verified
against a real run.

**Possible fix, if the growth is undesirable in practice** (e.g. cost/
capacity headroom on a shared EC2 box): explicitly trim the pool.
`cuda_async_memory_resource` exposes the underlying `cudaMemPool_t`; nothing
in `rmm_environment.cpp` today calls `cudaMemPoolTrimTo()` on it between
queries or on an idle timer. Switching `use_async_allocator: false` would
not help -- `pool_memory_resource` has the same "grows, never shrinks until
destroyed" behavior, and is the same process-lifetime singleton on the
server path.
