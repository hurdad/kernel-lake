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

1. **Pinned host memory for the device-to-host path.** ~~Every output batch
   currently pays a pageable-memory `cudaMemcpy` inside
   `cudf::to_arrow_host()` (`arrow_bridge.cpp`).~~ **Measured 2026-08-08 (see
   "Profiling results" below): not worth it as a first move.** D2H is
   ~9.8% of wall time even for a query returning 2M rows, and ~0.1% for a
   typical small-result aggregate -- both dwarfed by Parquet decode
   (~43%/~33%). Revisit only after Parquet decode itself has been
   addressed, or for a workload that's confirmed to return unusually large
   result sets.
   - Tradeoff (if revisited): pinned memory is a scarcer, harder-to-size
     resource than pool device memory -- over-allocating it hurts the whole
     system (host RAM pressure, page-lock limits), not just this process.

2. **Drop the single-query mutex in `GpuExecutionCoordinator`.** The server
   currently serializes on one query at a time, so multi-stream async
   execution never gets to overlap across queries.
   - Tradeoff: `RmmEnvironment` is one shared pool/stats/limiter for the
     whole process. True concurrency needs either per-query sub-pools or a
     thread-safe stream-ordered suballocator -- a real architecture change,
     not a tweak.

3. **Overlap Parquet decode with compute across multiple streams** for
   multi-file/multi-fragment scans. Everything currently funnels through
   the one per-query stream sequentially. **Prototyped 2026-08-08 -- see
   "Overlap prototype" below: real, reproducible ~13% wall-time win.** Not
   yet implemented in the real engine.
   - Tradeoff: adds real complexity -- and more than the original framing
     here suggested. `cudf::io::chunked_parquet_reader::read_chunk()` is a
     *blocking* host call (it needs the resulting row count on the host
     side before it can return), so "use two streams" alone doesn't get
     overlap -- decoding chunk N+1 while computing on chunk N needs a
     background host thread prefetching the next chunk, not just a second
     stream from the same thread. That means real cross-thread
     synchronization (not just CUDA events) and rethinking
     `ParquetScanOperator`'s single-threaded pull-based `next()` contract,
     plus tracking allocations across two concurrently-active streams
     against the query's memory limit (`RmmEnvironment`'s stack assumes one
     stream today).

4. **Retune `pass_read_limit_bytes`** (currently
   `rmm_environment.query_memory_limit_bytes() / 4`, see
   `query_engine_execute_gpu.cpp`). That divisor was set from one SF100
   measurement (TPC-H-derived Q1 at SF100); profiling more query shapes
   could recover headroom.
   - Tradeoff: low risk, but only matters if a workload is actually
     memory-constrained today -- easy to spend time tuning a number that
     isn't the bottleneck.

5. ~~Profile with `nsys` before acting on any of the above.~~ **Done
   2026-08-08 -- see "Profiling results" below.** Not a full `nsys`
   kernel-level trace (the collector, `nsight-systems-target`, is installed,
   but converting its raw output needs a separate host-side importer binary
   that only ships in the full ~150MB `nsight-systems` package plus two more
   missing shared libs, all needing `sudo` not available in this
   environment) -- used the app's own `--stats` output
   (`elapsed_wall_seconds`/`parquet_decoding_seconds`/
   `gpu_execution_seconds`/`device_to_host_seconds`, already wired in
   `query_engine_execute_gpu.cpp`) instead, across a small-result and a
   large-result query on the local GPU. Sufficient to answer the question
   this item existed to answer (is #1 worth doing) without the GUI trace.

## Profiling results (2026-08-08, local RTX 5060 Ti)

Caveat on the first two tables below, called out already in
`rmm_environment.cpp`'s own comment: this GPU is a desktop card also used
interactively (not a dedicated headless one), and had other real GPU load
on it (confirmed via `nvidia-smi`: ~47% utilization, ~6 GiB used, no CUDA
compute context -- consistent with a game's graphics workload) during those
measurements. A clean re-check on an idle GPU (below, "Clean re-check on an
idle GPU") confirmed the relative conclusions held and tightened up the
absolute numbers -- treat that last table as authoritative, not the earlier
two.

Two query shapes, 3 runs each, 20M-row synthetic dataset (8 files) via the
CLI's `--stats` flag. Values are per-run averages; `compute-only` is
`gpu_execution_seconds - parquet_decoding_seconds` (`gpu_execution_seconds`
includes decode -- see `query_engine_execute_gpu.cpp`).

| Query shape | rows out | decode | compute-only | D2H | wall | D2H % of wall |
|---|---|---|---|---|---|---|
| `GROUP BY` aggregate (small result) | 80 | 0.144s | 0.126s | 0.0004s | 0.437s | 0.09% |
| `SELECT *` + filter (large result) | 2,001,849 | 0.219s | 0.102s | 0.050s | 0.507s | 9.8% |

**Conclusion:** Parquet decode dominates in both cases (33-43% of wall
time), and stays roughly flat in absolute terms regardless of result size
(it does the same scan work either way). D2H scales with result size, as
expected, but even at 2M rows returned it's a distant third behind decode
and compute -- confirming the doc's original hedge that pinned memory (#1)
"might be a small fraction of wall time next to kernel execution itself."
The bigger, unaddressed cost in this profile is Parquet decode itself, not
covered by any item on this list yet.

**Methodology caveat (resolved below):** the table above comes from a
*fresh CLI process per query* (`kernellake query --stats`) -- each run pays
its own CUDA context creation + cudf/RMM first-kernel-launch cost, which
lands inside `parquet_decoding_seconds` (the scan is the first operator to
touch the GPU), potentially inflating decode's share and flattening its
apparent scaling with data volume. Checked this by extending
`kernellake benchmark tpch`'s `IterationMetrics`
(`src/cli/benchmark_tpch_command.cpp`) to also report
`parquet_decoding_seconds`/`gpu_execution_seconds`/`device_to_host_seconds`
per iteration (previously only `wall_seconds`/`rows_returned`/
`peak_gpu_memory_bytes`), then ran the same aggregate query warm (2
warmup + 5 measured iterations, so cold-start cost is excluded) at two
scan sizes against one long-lived process -- 2M rows/1 file vs. 20M
rows/8 files, same 80-row output both times:

| Scanned rows | decode (avg) | compute-only (avg) | D2H (avg) | wall (avg) | decode % of wall |
|---|---|---|---|---|---|
| 2,000,000 | 0.0470s | 0.0304s | 0.0004s | 0.0828s | 56.7% |
| 20,000,000 (10x) | 0.1918s | 0.1137s | 0.0006s | 0.3366s | 57.0% |

Decode's ~57% share of wall time is essentially identical at both scales
(not an artifact of process-startup noise), and decode time itself scales
sub-linearly with scanned data volume (10x rows -> ~4.1x decode time,
consistent with fixed per-query overhead -- reader/row-group setup,
planning -- amortizing better at larger scans). This confirms the original
conclusion on firmer footing rather than overturning it: decode is a real,
data-volume-driven cost, not a cold-start artifact, and D2H's earlier 9.8%
figure is specific to *large result sets* (`SELECT *` returning 2M rows),
not large *scanned* input -- these are different, independent axes (a
`GROUP BY` scanning 20M rows but returning 80 still shows D2H at ~0.18% of
wall time). The benchmark tool's new per-iteration fields are a permanent,
reusable addition -- useful for any future warm-process profiling, not
just this investigation.

**Clean re-check on an idle GPU (2026-08-08, same day, later):** the
contention noted above was a concurrent game session on this desktop GPU;
re-ran the identical warm 2M-vs-20M-rows comparison once it ended
(`nvidia-smi` confirmed ~14% utilization / ~1.1 GiB used beforehand, back
near the untouched baseline), 3 warmup + 8 measured iterations this time:

| Scanned rows | decode (avg) | compute-only (avg) | D2H (avg) | wall (avg) | decode % of wall |
|---|---|---|---|---|---|
| 2,000,000 | 0.0165s | 0.0102s | 0.0003s | 0.0304s | 54.2% |
| 20,000,000 (10x) | 0.0930s (5.6x) | 0.0513s (5.0x) | 0.0003s (flat) | 0.1704s (5.6x) | 54.6% |

Same qualitative picture, now with tighter numbers and no contention
caveat needed: decode's share is consistent (54.2% vs 54.6%) across
scales, D2H stays flat and negligible regardless of scan size, and decode
now scales *closer to linearly* with data volume (5.6x time for 10x rows,
vs. the noisier 4.1x under contention) -- the sub-linear amortization
effect was partly a contention artifact, not purely fixed per-query
overhead. This is the number to treat as authoritative; the contended
measurements above are kept for the methodology record, not as the final
answer.

**Is decode I/O-bound? No -- checked directly.** `benchmark tpch --mode
cold` already exists for exactly this (`evict_from_page_cache()` in
`benchmark_tpch_command.cpp`, `posix_fadvise(..., POSIX_FADV_DONTNEED)`
before every iteration). Ran it back-to-back against `--mode warm` (page
cache left alone across iterations), same warm process, idle GPU, 20M-row
dataset, 8 measured iterations each:

| Mode | per-iteration decode (s) | avg decode |
|---|---|---|
| warm | 0.0922, 0.0935, 0.0933, 0.0930, 0.0941, 0.0951, 0.0928, 0.0968 | 0.0939s |
| cold (cache evicted every iteration) | 0.0908, 0.0923, 0.0937, 0.0940, 0.0923, 0.0920, 0.0943, 0.0923 | 0.0927s |

Indistinguishable -- the two distributions overlap completely, well within
normal per-iteration variance. Evicting the Linux page cache before every
single read has no measurable effect on decode time on this machine (WSL2,
NVMe-backed). That rules out OS-page-cache-level I/O latency as what's
driving decode cost here -- whatever's taking the time is downstream of
that: the actual read syscalls, the host-to-device copy of raw
(compressed) bytes, or cudf's own GPU-side decompression/decode kernels.
Distinguishing between *those* three needs either instrumentation inside
cudf itself or the full `nsys` kernel-level trace this investigation
couldn't get working (see "Fixed" item #5 above) -- not pursued further
here. (Caveat: this result is specific to this machine's storage stack --
a WSL2 VM's virtualized disk likely already goes through a fast host-side
cache Linux's own page-cache eviction can't touch, so "not I/O-bound" here
doesn't necessarily generalize to, say, real S3 access latency on the AWS
benchmark path.)

**Is it decompression, then? Yes -- confirmed directly, and it's the
biggest single lever found in this whole investigation.** Generated three
byte-identical (same rows, same schema, same dictionary encoding) 20M-row
files via `pyarrow`, varying only Parquet compression codec, then ran each
through the same warm `benchmark tpch` setup (idle GPU, 8 iterations):

| Codec | file size | decode (avg) | vs. uncompressed |
|---|---|---|---|
| none | 745.9 MB | 0.0733s | -- |
| snappy (KernelLake's current default -- `sample_data_generator.cpp`) | 662.3 MB | 0.0896s | +22.2% |
| zstd | 560.5 MB | 0.1038s | +41.6% (+15.8% vs. snappy) |

Striking: decode time moves in the *opposite* direction from file size --
zstd has the smallest file (least to read/transfer) but the slowest
decode, because it costs more to decompress than it saves in transfer
time. This directly confirms decode is decompression-bound, not
I/O-bound, on this machine: switching Snappy -> uncompressed cuts decode
time by ~18% (equivalently, uncompressed -> Snappy costs +22%), which
at decode's confirmed ~55% share of wall time works out to roughly a
12% *total wall-time* difference for Snappy vs. uncompressed, and ~23%
for zstd vs. uncompressed -- for a compression choice that's otherwise
invisible to query correctness and easy to change. (Caveat: this trades
against on-disk/on-S3 storage cost and real network transfer time, which
matter more over a slow network or against real object storage than on
this machine's local NVMe -- the AWS benchmark path, reading real S3 data,
is a different tradeoff than what's measured here.)

**Confirmed again on real TPC-H data and real benchmark queries** (not
just synthetic data/queries) -- generated real SF1 `lineitem` (6M rows) at
all three codecs via the project's own `tools/generate_tpch.py`
(`--compression none|snappy|zstd`, already supports this), then ran the
actual `benchmarks/tpch/queries/q01.sql` and `q06.sql` through
`benchmark tpch --mode warm` (idle GPU, 6 iterations):

| Query | none | snappy | zstd | snappy vs. none | zstd vs. none |
|---|---|---|---|---|---|
| Q1 (`GROUP BY` over lineitem) | 0.0210s (34.4% of wall) | 0.0335s (45.1%) | 0.0352s (45.4%) | +59.5% | +67.6% |
| Q6 (scalar aggregate) | 0.0170s (65.4% of wall) | 0.0263s (76.7%) | 0.0270s (76.1%) | +54.7% | +58.8% |

Same pattern, and actually a *larger* decode penalty than the synthetic
test (55-68% here vs. 22-42% synthetic) -- real TPC-H `lineitem` has more
columns and a different value distribution than the synthetic generator,
so the exact percentage isn't the point; the direction and the fact that
it holds on the project's real benchmark queries is. Also notable: decode's
share of wall time varies a lot by query shape -- 34-45% for Q1 (six
aggregates + a two-column `GROUP BY`, more GPU compute work competing for
wall time) vs. 65-77% for Q6 (a single filtered `SUM`, almost nothing but
scan+decode). Compression cost isn't a fixed tax; it matters more for
scan-heavy/compute-light queries than compute-heavy ones.

## Overlap prototype (opt #3): does decode/compute overlap actually help?

Before touching the real engine, checked whether overlapping decode with
compute is worth the real complexity outlined in opt #3 above -- decode
being confirmed decompression-bound (GPU-compute-bound) raised the
question of whether it and downstream compute would just contend for the
same SMs with no net benefit, since the classic case for stream overlap
(hiding *I/O* latency behind compute) doesn't apply here.

**Method.** A standalone prototype (not part of the real build -- linked
directly against the vendored `libcudf`/`librmm` using flags pulled from
`compile_commands.json`), not wired into KernelLake's operator tree.
Loads two real 6M-row SF2 `lineitem` files (snappy), and for each of 5
iterations runs:
- **Sequential baseline:** decode file 1 -> compute over file 1 (Q6's
  shape: filter `l_quantity < 24`, `SUM(l_extendedprice * l_discount)`) ->
  decode file 2 -> compute over file 2, all on one stream, one thread.
- **Overlap:** a background `std::thread` starts decoding file 2 on its
  own stream while the main thread decodes + computes over file 1 on a
  separate stream; main thread joins the background thread before
  computing over file 2.

**Result**, 3 independent runs of 5 iterations each, idle GPU:

| Run | sequential mean | overlap mean | speedup |
|---|---|---|---|
| 1 | 0.0973s | 0.0849s | 12.8% |
| 2 | 0.0927s | 0.0809s | 12.8% |
| 3 | 0.0932s | 0.0811s | 13.0% |

**Real and reproducible -- worth implementing for real.** ~13% wall-time
reduction, consistent across independent runs, confirms there's genuine
idle GPU capacity during at least one phase (likely: decode has real
host-side bookkeeping/synchronization overhead between its own internal
stages, not just GPU kernels, leaving SM time free for compute to use
concurrently) that overlap can exploit -- the "they'll just contend for
the same SMs" worry didn't hold. The catch found along the way: this only
works via a background *thread* prefetching the next chunk, not just a
second stream from the same thread, since `read_chunk()` blocks the host
until that chunk's row count is known (see opt #3's updated tradeoff note
above) -- a bigger change to `ParquetScanOperator`'s single-threaded
pull-based design than the original framing implied, but the prototype
result says it's worth doing.

**Implemented for real (2026-08-08, `c1f98f9`).** The prototype's pattern
is now in `ParquetScanOperator`/`operator_builder.cpp` for the
non-partitioned scan path (the common case -- see that operator's own
class comment for why the rarer partitioned path was left synchronous).
Both concerns the prototype had sidestepped turned out to already be
handled or were addressed directly: `RmmEnvironment` installs one resource
stack as the *device's* default resource (not scoped to any one stream),
so two concurrently-active streams' allocations are tracked/limited
identically with no redesign needed; cross-stream data-readiness is
handled with a `cudaStreamWaitEvent` (non-blocking on the host) rather
than a real host-side synchronization primitive, so the consumer's own
stream never actually blocks waiting for decode.

Re-measured post-implementation with a real controlled A/B -- the
pre-change and post-change binaries built from the same source tree
(`git worktree`), run back-to-back against identical local (not S3 --
network/object-store variance would otherwise confound the comparison)
SF10 Parquet data, 5 iterations each, cold mode:

| Query | wall (old) | wall (new) | delta |
|---|---|---|---|
| Q1  | 0.508s | 0.466s | -8.1% |
| Q6  | 0.213s | 0.193s | -9.5% |
| Q14 | 0.489s | 0.430s | -12.1% |
| Q19 | 0.724s | 0.638s | -11.8% |

Consistent with the standalone prototype's ~13% finding, holding up
end-to-end through the real engine (parse/plan/execute/result-materialize,
not just the isolated decode+compute microbenchmark the prototype
measured). Note: `QueryResult.parquet_decoding_seconds` moved by more
than the wall-time delta above (e.g. Q1: 0.228s -> 0.084s) -- this isn't
decode getting proportionally faster; it now measures time on
`decode_stream_` specifically, isolated from other stream traffic (D2H
copies, etc. still on the default/consumer stream), where the old
next()-self-time measurement's whole window could include incidental
cross-stream serialization that had nothing to do with decode itself.
Wall time is still the trustworthy end-to-end number; treat
`parquet_decoding_seconds` as a genuine measurement of decode-stream time,
not as directly comparable pre/post this change.

## Recommendation

#1 (pinned memory) is now de-prioritized based on the measurements above --
don't start there; it only matters for queries with genuinely large result
sets, not large scans. Parquet decode is the confirmed lever (~55% of wall
time, scales with data volume), and it's confirmed decompression-bound, not
I/O-bound, on this machine -- the compression codec KernelLake's own
`generate-data` writes by default (Snappy, `sample_data_generator.cpp`)
costs ~22% more decode time than uncompressed, ~15% less than zstd. This is
the single most concrete, actionable lever this whole investigation found:
no architecture change needed, just a codec choice -- worth surfacing as an
explicit, documented tradeoff (query speed vs. storage footprint) for
anyone generating or choosing Parquet data for KernelLake to read, rather
than defaulting to Snappy without discussion. Whether it's worth *changing*
the default depends on a tradeoff this doc can't resolve alone: storage/
transfer cost (especially over S3, not measured here) vs. decode speed.

#3 (decode/compute overlap) is now implemented for real (`c1f98f9`), not
just prototyped -- an 8-12% real wall-time reduction confirmed end-to-end
via a controlled A/B (see above), and it's a genuine architecture
improvement (not just a config choice like #1), so it stacks with the
compression finding rather than competing with it: an
uncompressed-or-lighter-codec dataset gets *both* less decode work and
that work overlapped with compute.

#2 (concurrent queries) remains the biggest structural change and should
wait until there's a concrete reason (observed queueing under real load)
to take on the RMM-sharing redesign it requires.

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

## Fixed: narrowed `SELECT` projections crashed when combined with `ORDER BY`/`LIMIT`

Found while driving `kernellake-server` locally for the memory-growth check
above. **Fixed 2026-08-08** (same session, once isolated) -- not actually a
Flight SQL server bug as first suspected; see below.

Any query of the shape `SELECT col1, col2, ... FROM read_parquet(...)`
(explicit column list, no aggregate, not `SELECT *`) crashed whenever
combined with `ORDER BY` and/or `LIMIT`:

```
vector::_M_range_check: __n (which is 3) >= this->size() (which is 3)
```

- `SELECT order_id, amount, region FROM ... WHERE amount > 990 ORDER BY amount DESC`
  -> `__n (which is 3) >= this->size() (which is 3)`
- `SELECT order_id, amount, region, category, discount FROM ... WHERE amount > 500 LIMIT 1000`
  -> `__n (which is 6) >= this->size() (which is 5)`
- `SELECT order_id, amount, discount FROM ... LIMIT 1000` (no `WHERE` at all)
  -> `__n (which is 3) >= this->size() (which is 3)`

**Root cause.** First reproduced through `kernellake-server`, but a plain
CLI `kernellake query` against the same file crashed identically -- ruling
out the Flight SQL layer and pointing at `src/io/physical_planner.cpp`
instead. `LogicalSort`/`LogicalLimit` are pass-through nodes (same schema as
their child) that can end up sitting directly between a surviving
`LogicalProjection` and the `Filter`/`Scan`/`Join` chain it needs to remap
its column indices against: `LogicalSort`, whenever a non-aggregate query
has an `ORDER BY` (`logical_planner.cpp` places it directly on the
Filter/Scan chain); `LogicalLimit`, whenever the optimizer's
`insert_limit()` pushes a `LIMIT` down through a `LogicalProjection` to sit
just above whatever it can actually benefit from (`optimizer.cpp`). The
projection's "does my child reference the scan schema, and therefore need
its `ColumnExpression` indices remapped against the pruned scan" check
(`items_reference_scan_schema` in `physical_planner.cpp`) only tested
directly for `LogicalFilter`/`LogicalScan`/`LogicalJoin` -- missing both
pass-through cases, so a surviving, genuinely-reordering projection kept its
*original* (pre-pruning) column indices instead of the narrowed ones,
causing the out-of-bounds access at execution time. `SELECT *` and `GROUP
BY` aggregates never hit it: `SELECT *` elides the projection entirely
(nothing to remap), and an aggregate-path reprojection's items reference the
*aggregate's* output schema, not the scan, so they're correctly never
remapped regardless of what sits in between.

**Fix:** added `references_scan_schema()` in `physical_planner.cpp`, which
looks through any interposed `LogicalSort`/`LogicalLimit` to find whether
the real underlying node is `Filter`/`Scan`/`Join`; replaced both the
projection's own check and the analogous (same-shaped, though not currently
reachable the same way) check in the `LogicalSort` conversion with it.
Regression tests: `PhysicalPlannerTest.SurvivingPlainProjectionRemapsThrough
AnInterposedSort` / `...InterposedLimit` in `tests/unit/physical_planner_test.cpp`.
Verified against real data on the local GPU: all four repro queries above
now return correct (correctly filtered/ordered/limited) results; full test
suite (326 unit + 82 GPU tests) passes with no regressions.

## Real-S3 scan throughput: root cause found, not yet fixed

Found running a real SF10 benchmark against real AWS infra (`benchmarks/aws/`,
kernellake host on `g6.2xlarge`, torn down after this session): the same
queries that hit 1.5-2.7 GB/s scan throughput locally (docker-compose
stack, MinIO on localhost -- see the local-docker-compose-stack finding
elsewhere in this repo's history) hit only 0.07-0.17 GB/s over real S3.
`parquet_decoding_seconds` looked tiny (~0.1s of a ~30s query) at first,
which was itself misleading -- see below.

**First hypothesis (wrong on the mechanism, real bug anyway).**
`ObjectStoreDatasource` (`src/execution_gpu/object_store_datasource.cpp`)
didn't override `cudf::io::datasource::host_read_async()`. That method's
base-class default wraps the synchronous `host_read()` in
`std::async(std::launch::deferred, ...)`, which does not run until the
returned future is waited on -- so a caller that kicks off N "concurrent"
reads and then waits on each in turn gets N fully serialized reads, no
overlap, each paying the backend's own round-trip latency back-to-back.
Fixed by overriding both `host_read_async()` overloads to launch on
`std::launch::async` instead.

**Confirmed, by instrumenting real calls against the real S3 bucket, that
this was not the actual bottleneck.** A controlled local A/B (pre/post
binaries from the same tree via a throwaway `git worktree`, same real S3
query) showed no measurable difference: ~32s either way. Temporarily
logging every `host_read`/`host_read_async` call for one query
(`s3://.../tpch-data/sf10/lineitem-*.parquet`, Q6) showed why:
`host_read_async()` was called **zero** times. Every one of 156 real reads
(533MB total -- properly bulk-sized, 4 bytes up to ~7.3MB, not pathologically
fragmented) went through the plain synchronous `host_read()`, one at a
time, from a single thread.

**Where the time actually goes -- a second, more important discovery: it's
inside `has_next()`, not `read_chunk()`.** Instrumenting both calls
separately: for the same pass, `has_next()` measured ~18-24s while
`read_chunk()` measured ~0.02-0.1s. `cudf::io::chunked_parquet_reader`
does its real per-pass I/O and decode work when `has_next()` determines
whether another chunk exists (it has to fetch and decompress enough data
to know), not when `read_chunk()` hands that already-prepared data back.
This means `ParquetScanOperator`'s own `decode_seconds_` timer (added
alongside the decode/compute overlap work earlier this session, see above)
was silently measuring the *wrong call* -- it only wrapped `read_chunk()`,
so `parquet_decoding_seconds` reported ~0.1s of decode cost when the real
figure was ~28s. **Fixed**: both `prefetch_loop()` and the partitioned
path's `next()` now time from before `has_next()`, not just around
`read_chunk()` -- confirmed by rerunning the same real S3 query:
`parquet_decoding_seconds` now reads ~27.7s of a ~31.5s total, an honest
number instead of a misleading one.

**Not yet fixed: the actual serialization.** 156 sequential synchronous
reads, one full round-trip at a time, appears to be inherent to
`chunked_parquet_reader`'s multi-datasource constructor (the one used for
any non-local scan) -- it never calls the async override at all, so
nothing at the `ObjectStoreDatasource` adapter layer can influence *when*
cudf chooses to issue a read. A real fix needs either a speculative
prefetch/read-ahead layer built into `ObjectStoreDatasource` itself (start
reading likely-next ranges before cudf asks for them), or a way to get
cudf to process the N per-file datasources concurrently at its own level
(unconfirmed whether one exists in this cudf version -- no local source to
check, only prebuilt-wheel headers). Both are real, riskier engineering
than what's been done here; not attempted yet. `host_read_async()` stays
overridden regardless (a real correctness fix for any future/other caller
that does use it), just documented as not the fix for this specific
regression.
