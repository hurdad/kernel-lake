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
  not `cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)`. This is
  load-bearing, not just an unset flag: `to_arrow_host()` / `from_arrow()`
  in `arrow_bridge.cpp` don't take a stream argument, so they run on
  CUDA's default/null stream. That only reads correct data because a
  blocking-flagged stream implicitly synchronizes with the null stream
  (null-stream ops wait for prior work on any blocking stream to finish).
  If the flag were changed to actually be non-blocking, the D2H copy in
  `to_arrow_record_batch()` would race against still-in-flight kernels on
  `context.stream`. **Fixed 2026-08-08:** the header comment used to
  mislabel this a "non-blocking" stream; `cuda_utils.hpp`/`cuda_utils.cpp`
  now document why the blocking flag is deliberate instead.

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

**Verified 2026-08-08, on real hardware (local RTX 5060 Ti via WSL2, not
EC2 -- see "Local GPU testing" below): plateau, not a leak.** Built
`kernellake-server` with `KERNELLAKE_WITH_CUDA=ON`, started it against
locally generated synthetic Parquet data (`kernellake generate-data`,
2M rows / 4 files), and drove it over real Flight SQL with a Python ADBC
client while sampling `nvidia-smi` after every query:

- 30x the same `GROUP BY` query: memory jumped once on the very first call
  (+202 MiB, server startup + first-touch pool growth), then held flat
  (±1 MiB noise) for the remaining 29.
- 4 different query shapes (aggregate, `GROUP BY` x2, `SELECT *`) x 4
  rounds (16 queries total): zero net growth across the whole run.
- Killed the server afterward: `nvidia-smi` usage dropped back to within
  1 MiB of the pre-server baseline -- the pool releases everything to the
  driver on process exit, no leftover driver-level leak either.

Conclusion: the growth seen during AWS benchmark testing is the pool
reaching a working-set high-water mark once (either at server startup, or
the first time each distinct query shape's peak memory need is seen) and
then staying there -- consistent with the "benign pool caching" mechanism
above, not an actual leak. **The `cudaMemPoolTrimTo()` fix described below
is not needed** unless that high-water mark itself turns out to be a
problem in practice (e.g. it's larger than expected, or capacity/cost
headroom on a shared box is tight) -- worth re-checking against the real
TPC-H query set at realistic scale factors before ruling that out
entirely, since this check used small synthetic data and simple queries,
not the full benchmark suite.

**Possible fix, if the plateau level itself is ever undesirable** (e.g.
cost/capacity headroom on a shared EC2 box): explicitly trim the pool.
`cuda_async_memory_resource` exposes the underlying `cudaMemPool_t`; nothing
in `rmm_environment.cpp` today calls `cudaMemPoolTrimTo()` on it between
queries or on an idle timer. Switching `use_async_allocator: false` would
not help -- `pool_memory_resource` has the same "grows, never shrinks until
destroyed" behavior, and is the same process-lifetime singleton on the
server path.

## Local GPU testing (before reaching for EC2)

This dev machine has a real GPU (RTX 5060 Ti, WSL2 passthrough, CUDA 12.4)
that the `gpu-dev`/`release` CMake presets weren't actually using --
`build/gpu-dev`'s cache was CLion-managed, pointed at a Windows toolchain
(`ninja.exe` under `/mnt/c/...`) unreachable from a plain Linux shell. A
fresh Linux-native build directory works fine and doesn't disturb that
CLion-managed one:

```bash
mkdir -p build/gpu-dev-wsl
cmake -S . -B build/gpu-dev-wsl -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DKERNELLAKE_WITH_CUDA=ON \
  -DKERNELLAKE_BUILD_TESTS=ON \
  -DKERNELLAKE_BUILD_BENCHMARKS=ON \
  -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
# Note: the gpu-dev preset hardcodes CMAKE_CUDA_COMPILER to
# /usr/local/cuda/bin/nvcc, which doesn't exist on this box -- this apt-
# installed toolkit puts nvcc at /usr/bin/nvcc instead. Override it explicitly
# rather than editing the preset, since /usr/local/cuda/bin/nvcc may well be
# right on other machines.
cmake --build build/gpu-dev-wsl -j"$(nproc)"                 # core lib + CLI + GPU tests
cmake -S . -B build/gpu-dev-wsl -DKERNELLAKE_BUILD_SERVER=ON  # add kernellake-server
cmake --build build/gpu-dev-wsl --target kernellake-server -j"$(nproc)"
```

All RAPIDS deps (`librmm`, `libkvikio`, `libcudf`, `nvcomp`) fetch as
prebuilt PyPI wheels per `cmake/ThirdPartyRapids.cmake` -- no libcudf
source compile, so this configures and builds in a few minutes, not hours.
`./build/gpu-dev-wsl/tests/gpu/kernellake_gpu_tests` passed all 82 tests on
real hardware.

This is a much cheaper and faster loop than provisioning the AWS GPU
instance (`benchmarks/aws/` -- fully torn down as of this writing, only the
S3 bucket with real TPC-H data survives) for anything that doesn't
specifically need EC2-scale data or the full benchmark harness. Reach for
EC2 only once a local check like this one has narrowed down what's worth
spending real money to verify at scale.

## New finding: Flight SQL server crashes on plain narrowed `SELECT` projections

Found while driving `kernellake-server` locally for the memory-growth check
above -- **not yet investigated further, root cause unconfirmed.**

Any query of the shape `SELECT col1, col2, ... FROM read_parquet(...)`
(explicit column list, no aggregate, not `SELECT *`) crashes when issued
through the Flight SQL server, every time, regardless of `WHERE`/`LIMIT`:

```
adbc_driver_manager.OperationalError: UNKNOWN: [FlightSQL]
vector::_M_range_check: __n (which is 3) >= this->size() (which is 3)
(Unknown; ExecuteQuery)
```

- `SELECT order_id, amount, region FROM ... WHERE amount > 990 ORDER BY amount DESC`
  -> `__n (which is 3) >= this->size() (which is 3)`
- `SELECT order_id, amount, region, category, discount FROM ... WHERE amount > 500 LIMIT 1000`
  -> `__n (which is 6) >= this->size() (which is 5)`
- `SELECT order_id, amount, discount FROM ... LIMIT 1000` (no `WHERE` at all)
  -> `__n (which is 3) >= this->size() (which is 3)`

The server catches it and stays up (returns a gRPC `Unknown` error rather
than crashing the process), but this is a real, easily reproduced
correctness/availability bug on an extremely common query shape. It's
specific to the Flight SQL server layer, not the query engine itself: the
same narrowed-projection pattern is exercised constantly by the existing
GPU unit test suite (`HashJoinQueryTest.WhereClauseAfterJoinFiltersCorrectly`
et al., all 82 passing) via direct in-process `QueryEngine::execute()`
calls that never touch `flight_sql_server.cpp`. `SELECT *` and `GROUP
BY`-aggregate queries both work fine over Flight SQL -- only a plain
explicit column list triggers it. Points at result-schema/ticket
construction in `src/server/flight_sql_server.cpp`, not the physical
planner. Worth a dedicated investigation; out of scope for the memory-growth
check that found it.
