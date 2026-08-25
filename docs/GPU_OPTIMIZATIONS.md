# GPU optimization opportunities

Notes from a review of the current CUDA/cudf execution path
(`src/execution_gpu/`, `src/memory/rmm_environment.cpp`,
`src/api/query_engine_execute_gpu.cpp`, `src/server/gpu_execution_coordinator_gpu.cpp`).
Originally a prioritized list of opportunities plus the tradeoff each one
carries, none yet implemented -- several have since been investigated,
profiled, and (where it paid off) implemented for real; each opportunity
below is marked with its current status, and later sections in this file
document what was actually done and measured.

## Current state (baseline, for context)

- Device memory goes through RMM: a `limiting_resource_adaptor` ->
  `statistics_resource_adaptor` -> `cuda_async_memory_resource` (or
  `pool_memory_resource`) stack, one instance per process
  (`RmmEnvironment`, `src/memory/rmm_environment.cpp`). **Since updated
  (2026-08-24):** inside `kernellake-server`, `GpuExecutionCoordinator` now
  builds one such instance *per visible CUDA device*, not one for the whole
  process -- see "Multi-GPU Tier 1 implemented" below. Still exactly one
  process-wide instance for the CLI's one-shot `QueryEngine::execute(sql)`
  path, which never goes through `GpuExecutionCoordinator`.
- No pinned (page-locked) host memory anywhere in the tree -- confirmed by
  grep across `src/`, `include/`, `tests/`. Host<->device transfers rely on
  whatever libcudf does internally by default.
- Execution is stream-based: one `CudaStream` per query
  (`query_engine_execute_gpu.cpp`), threaded through every operator via
  `ExecutionContext::stream`. Every cudf call in every operator passes it
  explicitly -- no operator falls back to the legacy default stream.
  **Since updated:** `ParquetScanOperator` now also owns its own
  `decode_stream_` (see "Overlap prototype (opt #3)" below) for
  background-thread decode/compute overlap -- the *query* still has one
  primary stream, but that one operator is no longer single-stream
  internally.
- `GpuExecutionCoordinator` (`gpu_execution_coordinator_gpu.cpp`) holds a
  mutex around `execute()`: only one query runs at a time, server-wide.
  Streams give async/overlapping kernel execution *within* a query, not
  concurrent queries. **Superseded 2026-08-21 and 2026-08-24:** this
  describes the pre-"Opt #2" point-in-time snapshot. The mutex was replaced
  by a `std::counting_semaphore` sized to
  `EngineSection::max_concurrent_gpu_queries` ("Opt #2 implemented" below),
  and that semaphore is now per-device, not process-wide, since one
  `RmmEnvironment`/semaphore pair exists per visible GPU ("Multi-GPU Tier 1
  implemented" below) -- an N-GPU node allows up to
  `N x max_concurrent_gpu_queries` queries running at once, not one at a
  time server-wide.
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

2. ~~Drop the single-query mutex in `GpuExecutionCoordinator`.~~ **Implemented
   for real, 2026-08-21 -- see "Opt #2 implemented: bounded concurrent GPU
   queries" below.** Real measured result on this dev box's GPU: 2
   concurrent clients get 25% more throughput than 1 (25,806 -> 32,291
   queries/hour), a genuine improvement over the old mutex's confirmed-flat
   baseline (2026-08-17: identical throughput regardless of client count).
   Deliberately bounded (a bounded semaphore, `max_concurrent_gpu_queries`,
   not unconditional removal) rather than the originally-scoped unbounded
   version -- see that section for why.
   - Original tradeoff assumption (RmmEnvironment needing either per-query
     sub-pools or a thread-safe suballocator) was half right: the pool/
     limiter were already thread-safe (RMM's own documented guarantee,
     confirmed from source); the real, narrower blocker was
     `statistics_resource_adaptor`'s shared push/pop stack -- see below.

3. **Overlap Parquet decode with compute across multiple streams** for
   multi-file/multi-fragment scans. Everything currently funnels through
   the one per-query stream sequentially. **Prototyped 2026-08-08 -- see
   "Overlap prototype" below: real, reproducible ~13% wall-time win.**
   **Implemented for real the same session (`c1f98f9`), not just
   prototyped** -- see the "Recommendation" section below for the
   controlled-A/B real-engine result (8-12% wall-time reduction).
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
   - **A real SF100 Q3 (3-way join) GPU OOM, found in this session's AWS
     benchmark, turned out to have nothing to do with this knob** -- see
     "Fixed: predicate pushdown stopped at joins" below. `pass_read_limit_bytes`
     only sizes Parquet *scan* passes; `HashJoinOperator`'s build side has no
     size awareness at all, and that's what was actually overflowing. This
     item (a scan-sizing tweak) is still open on its own merits, but is not
     what Q3's OOM needed.

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

6. ~~Parallelize Parquet decode across files/streams, not just decode-vs-compute
   (opt #3 above already did the latter).~~ **Prototyped 2026-08-15 -- see
   "Parallel-decode prototype" below: no reliable win, and a real,
   reproducible regression past 2 concurrent streams. De-prioritized.**
   Originally motivated by code read (2026-08-15): for a multi-file,
   non-partitioned scan, `ParquetScanOperator` builds **one**
   `cudf::io::chunked_parquet_reader` over the entire file list, driven by
   **one** background thread (`decode_thread_`) on **one** `decode_stream_`,
   pulling through a single-slot read-ahead queue
   (`src/execution_gpu/parquet_scan_operator.cpp:73-100,138-204`) -- no
   thread pool, no per-file/per-row-group dispatch, no multiple concurrent
   readers. For a table with many files (e.g. SF1000 `lineitem`, ~1,586
   objects in the real S3 prefix), decode itself is fully serial across
   files: one chunk in flight at a time, however many files are queued
   behind it. Live `nvidia-smi dmon` sampling during a real SF1000 AWS run
   (2026-08-15) showed the GPU mostly idle (0-3% SM utilization on most
   1-second samples, bursting to 60% only occasionally) while memory stayed
   pegged near the query's ceiling throughout -- consistent with a single
   decode stream gating throughput. **That framing turned out to be
   misleading**: idle SM% doesn't mean idle decompression-engine/copy
   capacity, and the prototype below found splitting decode across streams
   contends for that shared capacity rather than exploiting spare room in
   it, the opposite of what opt #3 found for decode-vs-compute overlap.
   - Tradeoff (as originally scoped, now moot given the measured result):
     would have needed either N decode threads each with their own
     reader/stream, or a deeper multi-chunk read-ahead queue instead of
     today's single slot, plus weighing more concurrent in-flight GPU
     memory against the tight headroom found in the SF1000 Q3 OOM below.
     Not worth taking on any of that complexity given the prototype's
     result.

7. ~~Real GPUDirect Storage (GDS) instead of cuFile's compat-mode
   bounce-buffer copy.~~ **Closed out 2026-08-21 -- not viable on any
   instance type this project has tested or can realistically afford.**
   Tested for real across three attempts (`g6.8xlarge`/L4,
   `p5.4xlarge`/H100, `g7e.12xlarge`/RTX PRO 6000 Blackwell) -- see "Real
   GDS via FSx for Lustre + EFA" and "GDS take 3" below for the full
   investigation. Root cause found in AWS's own official client-setup
   tooling (`configure-efa-fsx-lustre-client.py`), not guessed: the real,
   authoritative supported-instance list is hardcoded as
   `p5.48xlarge`/`p5e.48xlarge`/`p5en.48xlarge`/`p6-b200.48xlarge` only --
   full 8-GPU sizes exclusively, contradicting both the smaller instances
   tried and the G7e product page's GDS claim. Not revisiting without an
   explicit decision to provision one of those four (all far more
   expensive, and still gated on real EC2 capacity plus a currently-0
   P-family on-demand vCPU quota).

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

## Parallel-decode prototype (opt #6): does splitting decode across files help? No.

Motivated by live `nvidia-smi` sampling during a real SF1000 AWS run
(2026-08-15, see opportunity #6 above) showing the GPU mostly idle on SM%
while `ParquetScanOperator` serializes every file's decode through one
`chunked_parquet_reader`/one thread/one stream. Before touching the real
operator, followed opt #3's own template: an isolated standalone
prototype, not wired into KernelLake, linked directly against the
vendored `libcudf`/`librmm` (flags pulled from `compile_commands.json`,
same as opt #3's prototype).

**Method.** Real local SF10 `lineitem` data (16 files, snappy, this dev
box's own `sf10-snappy` set -- a smaller stand-in for the SF1000 dataset
that motivated this, but real generated TPC-H data, not synthetic).
Reads 4 columns (`l_shipdate`, `l_quantity`, `l_extendedprice`,
`l_discount` -- Q6's shape, matching opt #3's own column choice) via
`cudf::io::chunked_parquet_reader`, on an idle GPU (confirmed via
`nvidia-smi`: 10% utilization, 1.2GiB used, no contention) this dev
machine's RTX 5060 Ti, 16GiB:
- **Baseline**: exactly mirrors `ParquetScanOperator::open()`'s current
  production shape -- one reader spanning all 16 files, one stream,
  `has_next()`/`read_chunk()` in a loop until exhausted.
- **Candidate**: files split round-robin into N groups (N in {2, 4, 8,
  16}), each group decoded by its own reader on its own stream from its
  own `std::thread`, all launched concurrently and joined at the end.

**Result**, 3 independent process runs (5 measured iterations each,
median reported, row counts cross-checked equal to baseline on every
run):

| N streams | run 1 | run 2 | run 3 |
|---|---|---|---|
| 2 | +12.3% | +3.3% | -3.6% |
| 4 | +0.8% | -8.0% | -17.7% |
| 8 | -20.5% | -26.7% | -41.1% |
| 16 | -62.0% | -73.4% | -90.7% |

(Positive = faster than baseline, negative = slower.)

**Real, reproducible -- and negative, past N=2.** N=2's result straddles
zero across three independent runs (noise, not a real effect); N=4, 8,
and 16 are consistently and increasingly *worse* than the single-reader
baseline, in every run, by a wide and growing margin. This is the
opposite of opt #3's own finding: decode-vs-compute overlap (one decode
stream running alongside the consumer's compute stream) found genuine
idle capacity to exploit, because the two streams do fundamentally
different kinds of work. Decode-vs-decode across N streams does the
*same* kind of work N times concurrently -- Parquet decompression on this
GPU is not free-standing SM-bound work that idle SM% implies there's room
for; multiple concurrent chunked-reader instances appear to contend for
shared decompression-engine/copy-engine capacity (and, less measurable
here, host-thread/driver-level serialization around each reader's own
`has_next()`/`read_chunk()` calls), so adding more concurrent streams adds
pure overhead once that shared capacity is saturated -- past a very low
concurrency threshold, immediately in this measurement.

**Conclusion: don't implement this.** The `nvidia-smi` SM%-idle
observation that motivated this item was a real observation but a
misleading diagnosis -- idle SM time during decode doesn't mean there's
spare decode throughput to parallelize into, the same way idle compute
time during I/O doesn't always mean an I/O-bound wait is parallelizable
if the underlying resource is already saturated by something else (here,
a single GPU's decompression/copy engines, not SMs). Opportunity #6 above
is marked de-prioritized; `ParquetScanOperator`'s single-reader/
single-stream design for the non-partitioned path is correct as-is. (This
result is specific to a desktop RTX 5060 Ti under WSL2 -- it's possible a
datacenter GPU with more/dedicated decompression engines behaves
differently, but that would need its own real measurement, not an
assumption, before revisiting this.)

**Re-verified for real on AWS `g6.4xlarge`'s NVIDIA L4 (2026-08-19), not
just assumed to generalize.** The prototype's original source was never
committed (scratch file) -- reconstructed from this section's own
documented methodology, cross-checked locally first (reproduced the same
qualitative degradation pattern on the dev box before trusting it), then
the compiled binary + its shared-library dependencies (`libcudf.so`,
`libnvcomp.so.5`, `librapids_logger.so`, `libcudart.so.12` -- copied
directly rather than rebuilt on the instance, since the dev box's
already-compiled artifacts run fine against the L4's newer driver via
standard CUDA forward compatibility) were copied to a real `g6.4xlarge`
and run against the same SF10 `lineitem` data. Real result, 2 clean reps
after rep 0's expected cold-start noise:

| N streams | rep 1 | rep 2 |
|---|---|---|
| 2 | -49.8% | -48.7% |
| 4 | -59.6% | -52.1% |
| 8 | -103.4% | -105.7% |
| 16 | -193.7% | -191.3% |

**Same monotonic degradation pattern holds on real datacenter hardware --
if anything, more severe and more consistent than the dev box.** N=2 was
noisy enough to straddle zero on the RTX 5060 Ti; on the L4 it's a clean,
unambiguous ~49% regression in both reps. This closes off the "maybe
it's a desktop/WSL2 artifact, a real datacenter GPU might behave
differently" open question from above -- it doesn't. The shared
decompression/copy-engine contention is real on production hardware too,
not something a bigger/better GPU sidesteps on its own. Directly bears on
the concurrency-mutex plan (see "Root cause is narrower..." section
above): decode-heavy concurrent queries should be expected to hit this
same contention, not just theorized to.

## Getting around decompression-bound decode (2026-08-17 discussion, not yet tested)

Given decode is confirmed decompression-bound (above) and, separately, that
concurrent decode streams on this GPU already measured as contending for
shared decompression/copy-engine capacity rather than scaling (opt #6
above), options to actually reduce or route around the cost, roughly
cheapest/most-proven first:

1. **Cheaper codec, or none.** Already the confirmed lever above (snappy
   +22-60%, zstd +41-68% vs. uncompressed). Not yet acted on -- real
   storage/S3-transfer cost tradeoff never priced.
2. **Try lz4.** Untested here. Typically decodes faster than both snappy
   and zstd for a middling ratio -- a real candidate to add to the same
   A/B if the codec question gets revisited.
3. **Reduce bytes decompressed, not decompression speed.** Already
   banked, not a new lever: `ParquetScanOperator` already pushes column
   projection (`.column_names(columns_)`) and row-group pruning
   (`.row_groups(...)`) into `cudf::io::parquet_reader_options` -- confirmed
   by reading `parquet_scan_operator.cpp` directly, not assumed.
4. **Bigger decode batches.** `pass_read_limit_bytes` (opt #4, never
   retuned) sizes how much cudf's nvCOMP-backed batched decompression
   processes per pass; bigger batches generally get better occupancy and
   amortize launch overhead. Unexplored, plausibly cheap, untested.
5. **Check for dedicated hardware decompression.** If nvCOMP dispatches
   this codec/GPU combo through a fixed-function decompression unit
   rather than an SM-resident kernel, decode wouldn't contend with
   compute (or with other queries' compute) for SMs at all -- would
   directly bear on the concurrency-mutex plan below. Not checked.

   **Candidate hardware identified (2026-08-17), corrected same day.**
   Originally mis-attributed to Hopper (H100/H200) -- corrected: it's
   **Blackwell** (B200/B300) that has the dedicated hardware
   **Decompression Engine**, not Hopper. (Still recalled from NVIDIA's
   own architecture materials, not verified against this repo's vendored
   `cudf`/`nvCOMP` v26.06.00 -- whether that build actually dispatches
   Parquet Snappy decode through it remains open regardless of which
   generation has the hardware.) Every GPU in this project's range so far
   (L4/`g6*`, L40S/`g6e*`, A10G/`g5*`, A100/`p4d*`, H100/H200/`p5*`) is
   Hopper or older -- no such engine, decompression runs as ordinary SM
   kernels, consistent with opt #6's measured contention. `p5.4xlarge`
   (H100, $6.88/hr) is therefore **not** a Decompression-Engine test after
   all -- still potentially useful as a bigger-GPU data point, but not
   evidence about the hardware-offload hypothesis.

   **Real constraint found (2026-08-17, via AWS EC2/Pricing API, not
   guessed): no affordable way to test this on AWS right now.** Unlike
   Hopper's single-GPU `p5.4xlarge`, Blackwell has no equivalent small
   SKU in this account/region -- only `p6-b200.48xlarge` and
   `p6-b300.48xlarge`, both fixed at **8x GPU**. Real on-demand price:
   **$113.93/hr** for `p6-b200.48xlarge`. That's not a bounded
   validation-test cost anymore -- a different category of spend than
   this investigation's other real-hardware checks, and not something to
   provision without an explicit decision to spend at that level.

   **Revised plan:** still worth running step 1 below on the cheap,
   already-planned instance -- it validates whether the contention is
   real in production at all, independent of any hardware-offload
   question. Step 2 (a genuine Decompression-Engine A/B) is blocked until
   either a smaller Blackwell SKU becomes available or there's a specific
   decision to spend at the 8x-B200 price point.
   1. Re-run the opt #6 N-concurrent-decode-stream prototype on
      `g6.4xlarge` itself (production instance type -- opt #6 only ever
      ran on the dev box's desktop RTX 5060 Ti under WSL2, never on the
      real production GPU). Confirms whether the contention is real in
      production or a desktop/WSL2 artifact. Cheap, do this regardless.
   2. A genuine hardware A/B against Blackwell -- on hold pending a
      smaller/cheaper SKU or an explicit spend decision at $113.93/hr.
6. **CPU-side decompression offload.** Bigger, more invasive: decompress
   on host cores (measured well under saturation on the AWS network-
   instance investigation) and DMA only the decompressed bytes to the
   GPU. Real potential, but risks just moving the bottleneck to PCIe
   transfer -- needs its own measurement before committing.

**Bears directly on the concurrency-mutex plan** (see "Root cause is
narrower..." section above): opt #6's already-measured result -- N
concurrent decode streams contending for shared decompression/copy-engine
capacity, getting *worse* past N=2, not better -- is the same shape of
resource contention that concurrent *queries* would hit if their workloads
are decode-heavy. Don't assume dropping the mutex yields even a modest
throughput gain for decode-heavy query mixes without checking; it's
plausible it comes back flat or negative on this same GPU, mirroring opt
#6's result. The planned `scaling_test.py` re-run after the mutex fix
should specifically watch for this, not just confirm throughput moved off
flat.

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
than defaulting to Snappy without discussion.

**Storage/transfer-cost side of the tradeoff, finished (2026-08-17).**
Previously left open ("depends on a tradeoff this doc can't resolve
alone"). Resolved using the already-measured file-size/decode-time table
above (the 20M-row synthetic test -- none 745.9 MB/0.0733s, snappy 662.3
MB/0.0896s, zstd 560.5 MB/0.1038s) plus real current AWS S3 pricing (`aws
pricing get-products --service-code AmazonS3`, us-east-1, confirmed live
via the Pricing API, not assumed: **$0.023/GB-month**, first-50TB tier --
the relevant one at this project's data scale).

*Storage cost*: negligible at any scale this project tests. Using the
measured 1.126x size ratio (none/snappy), SF1000's already-documented
168.8GB snappy `lineitem` alone would grow to ~190GB uncompressed -- ~21GB
extra, **~$0.49/month**. One EC2 instance-hour in this project's own
benchmark fleet ($1.32-$1.39/hr) costs more than two and a half months of
that storage delta. Storage was never the load-bearing part of this
tradeoff.

*Transfer-time cost* (the side that actually matters -- same-region S3->EC2
data transfer isn't billed, but the extra bytes still take wall-clock time
to move, which is what's actually being paid for via instance-hours):
solving for the S3 throughput at which uncompressed's transfer-time
penalty exactly cancels its decode-time saving, using the same paired
file-size/decode-time numbers above --

| Comparison | extra bytes (larger file) | decode time saved (smaller file's cost) | crossover throughput |
|---|---|---|---|
| none vs. snappy | 83.6 MB | 0.0163s | **~5.0 GB/s** |
| none vs. zstd | 185.4 MB | 0.0305s | **~5.9 GB/s** |
| snappy vs. zstd | 101.8 MB | 0.0142s | **~7.0 GB/s** |

Below its crossover throughput, the *smaller* (more compressed) file wins
on total wall-clock time (transfer + decode combined), not just decode
time alone -- above it, the bigger/less-compressed file wins. Every real
throughput number this project has ever measured is far below all three
crossovers: even the best-case *local* NVMe scan throughput this
investigation found (1.5-2.7 GB/s, opt #3/#6 sections above) doesn't clear
the lowest crossover (~5.0 GB/s vs. snappy), and real S3 throughput was
measured slower than local in every case tested here (0.07-0.17 GB/s
pre-fix, several-x faster post the `device_read` fix but not stated as a
clean GB/s figure in that writeup -- see "Real-S3 scan throughput fixed"
above). No realistic near-term improvement to this project's S3 client
gets within range of ~5 GB/s per query.

**Conclusion: keep Snappy as the default. Don't switch to uncompressed,
and zstd is confirmed *not* worth its extra decode cost either** (its own
crossover vs. snappy, ~7.0 GB/s, is even further out of reach) --
KernelLake's existing default already comes out ahead on both storage
*and* transfer-adjusted wall-clock cost, not just "acceptable," among the
three codecs actually tested. This closes out the previously-open item;
revisit only if a future S3-client change is independently measured to
push real per-query scan throughput into multi-GB/s territory, which
nothing in this investigation's history suggests is close.

#3 (decode/compute overlap) is now implemented for real (`c1f98f9`), not
just prototyped -- an 8-12% real wall-time reduction confirmed end-to-end
via a controlled A/B (see above), and it's a genuine architecture
improvement (not just a config choice like #1), so it stacks with the
compression finding rather than competing with it: an
uncompressed-or-lighter-codec dataset gets *both* less decode work and
that work overlapped with compute.

#2 (concurrent queries) remains the biggest structural change, but now
has both the concrete real-load signal that was missing and a real,
narrower root-cause finding than "redesign RMM sharing" implied.

**Real signal, confirmed 2026-08-17/18**: `scaling_test.py` run for real
against a single warm `kernellake-server`, SF1000, cache enabled -- 1
concurrent client: 73 queries/911s (288/hour, median 12.46s). 4 concurrent
clients: 76 queries/947s (289/hour, median 49.74s). Throughput is
identical regardless of client count; per-query latency scales almost
exactly linearly with client count. `GpuExecutionCoordinator::execute()`
(`src/server/gpu_execution_coordinator_gpu.cpp`) wraps the entire
`engine.execute()` call in a plain `std::mutex` -- confirmed for real,
not just from reading the code, that this fully serializes GPU execution
today regardless of concurrent Flight SQL connections.

**Root cause is narrower than "the whole RMM stack isn't thread-safe"**:
read `rmm_environment.cpp` and RMM's own `per_device_resource.hpp`
directly rather than guessing. The actual device memory allocator
(`cuda_async_memory_resource`/`pool_memory_resource`) is RMM's own
generally-thread-safe concurrent-allocation machinery -- not the blocker.
The genuinely unsafe piece is much narrower:
`RmmEnvironment::track_query()` calls `impl_->stats.push_counters()` /
`pop_counters()` -- RMM's `statistics_resource_adaptor` push/pop is a
single shared stack, designed for sequential nested scoping. Two threads
calling `track_query()` concurrently would interleave pushes/pops against
that one stack, corrupting per-query peak/current memory attribution
(silently wrong `QueryResult::peak_gpu_memory_bytes`, not a crash) --
that's the actual, narrow correctness gap, not GPU memory-allocation
safety in general.

**Confirmed real constraint, not guessed**: `rmm::mr::set_current_device_resource()`
installs into a `std::map<device_id, resource>` (RMM's
`per_device_resource.hpp`, read directly from the vendored source) -- one
resource per device, process-wide, not per-thread or per-call. So a
naive "just don't share the stats object" fix can't simply install a
different resource per query via that API.

**A real path forward, not yet audited**: this codebase already threads
an explicit `context.memory_resource` through the operator tree (e.g.
`hash_join_operator.cpp`'s spill path passes it explicitly to cudf
calls, not relying on the implicit global "current" resource). If every
GPU allocation in `execution_gpu/` consistently uses that explicit
per-call resource, each concurrent query's `ExecutionContext` could carry
its own `statistics_resource_adaptor` (correct per-query memory
attribution) layered over the *same* shared limiter/pool underneath --
without touching the single global per-device slot, which would stay
fixed to the shared base resource for the process's whole lifetime.

**Also confirmed this session, before the audit below**: `QueryEngine::execute()`
(`src/api/query_engine_execute_gpu.cpp`) already constructs `const CudaStream
stream;` as a local -- every call already gets its own fresh CUDA stream.
Stream isolation across concurrent queries needs no fix. The remaining gap
is narrower still: that same function initializes
`context.memory_resource` from `rmm::mr::get_current_device_resource_ref()`
-- the shared global resource -- not a fresh per-query wrapper. That one
line is where per-query `statistics_resource_adaptor` isolation would plug
in, *if* the audit below comes back clean.

**Audit done (2026-08-17): it did not come back clean.** Grepped every
`cudf::`/`rmm::` allocation call site across all 16 files in
`execution_gpu/`. Most operators (`filter_operator.cpp`,
`projection_operator.cpp`, `hash_aggregate_operator.cpp`,
`hash_join_operator.cpp`, `sort_operator.cpp`, `scalar_aggregate_operator.cpp`,
the per-batch decode path in `parquet_scan_operator.cpp`) are clean --
every real allocation explicitly passes `context.stream, context.memory_resource`,
confirmed line-by-line including multi-line call continuations. Two real,
concrete gaps found:

1. **`ObjectStoreDatasource::device_read()`** (`object_store_datasource.cpp:111`):
   `rmm::device_buffer buffer(size, stream);` -- omits the `mr` argument,
   so it silently defaults to `rmm::mr::get_current_device_resource_ref()`,
   the shared global resource, not `context.memory_resource`. Root cause:
   `cudf::io::datasource::device_read()` is a `cudf`-owned virtual
   signature -- `(offset, size, stream)`, no `mr` parameter exists to
   receive it. Every Parquet scan's raw compressed-page staging buffer
   (read from S3/local storage, before cudf's own decode) allocates
   through this path. Fix: give `ObjectStoreDatasource` a stored
   `rmm::device_async_resource_ref` member (passed in at construction,
   where `ParquetScanOperator::open()` already has `context.memory_resource`
   in scope) and pass it explicitly instead of relying on the omitted
   default arg.

2. **Literal-scalar construction** (`cudf_adapter.cpp`'s `literal_to_scalar()`/
   `make_decimal_scalar()`, and the `cudf::numeric_scalar`/`string_scalar`/
   `fixed_point_scalar`/`timestamp_scalar` constructors they call):
   constructed with no explicit `stream`/`mr` arguments at all, so both
   default -- `cudf::get_default_stream()` and
   `cudf::get_current_device_resource_ref()`. Three of four call sites
   (`ProjectionOperator::compile_value()`, `ScalarAggregateOperator::compile_expr()`,
   `HashAggregateOperator::compile_expr()`) run at plan-compile time, with
   no `ExecutionContext` in scope at all -- not a missed pass-through, the
   plumbing to carry a resource/stream there doesn't exist yet. The fourth
   (`parquet_scan_operator.cpp:285`, partition-column literals) *does*
   have `context` sitting right there in scope one line above a correct
   `context.stream, context.memory_resource` use, and still doesn't pass
   it to `literal_to_scalar()` -- the cheapest of the four to fix.

   **Open question, not resolved this session**: whether `cudf::get_default_stream()`
   resolves to the CUDA legacy/null stream (which implicitly
   synchronizes with *every* other stream on the device, including every
   other concurrent query's fresh per-query stream) or a per-thread
   default stream, depends on whether the vendored `libcudf` was built
   with `CUDF_USE_PER_THREAD_DEFAULT_STREAM`. Could not confirm from
   source in this environment -- `libcudf_cu12`'s pip package ships
   headers only, no `default_stream.cpp` implementation to read, and RAPIDS'
   own published packages are believed (not independently source-verified
   here) to build with that flag OFF. If it resolves to the legacy stream,
   gap #2 is a real cross-query serialization hazard on top of being a
   stats-attribution gap -- verify empirically (e.g. compare
   `stream.value()` against `cudaStreamLegacy`) before relying on "stream
   isolation is already free" as a blanket statement.

**Practical read**: gap #1 matters for every query (every Parquet scan
goes through it) but is bytes-scale, not correctness-scale, for memory
*accounting* -- the real allocator underneath is still the shared
thread-safe pool, so nothing corrupts, only per-query stats attribution
is off by however many staging buffers were live during the scan. Gap #2
is small in bytes (single-value scalars) but is the one that needs the
stream question resolved before concurrency work proceeds, since it's the
only place a query's device work could still be implicitly coupled to
another concurrent query's stream even after per-query streams and
resources land everywhere else.

**Revised next-session task**: fix gap #1 (thread a stored resource
through `ObjectStoreDatasource`) and gap #4-of-gap-2 (pass `context.stream,
context.memory_resource` into `parquet_scan_operator.cpp:285`'s
`literal_to_scalar()` call) first -- both are small, mechanical, low-risk
diffs. Then resolve the per-thread-default-stream question empirically.
Only then does implementing per-query `statistics_resource_adaptor`
wrappers (dropping the mutex, per the plan in this doc's earlier section)
become safe to attempt -- and even then, the three compile-time
`literal_to_scalar()` call sites (no `ExecutionContext` in scope) remain a
known, accepted gap unless/until compile-time plumbing is added too.

#6 (parallel decode across files) is now de-prioritized based on a real
prototype (see above) -- the opposite outcome from #3: no reliable win at
any tested concurrency, and a reproducible regression past 2 streams.
Don't pursue this; `ParquetScanOperator`'s existing single-reader design
for the non-partitioned path already appears to be the right shape. The
SF1000 GPU-idle observation that motivated this item is better explained
by opportunity #4 (`pass_read_limit_bytes` retuning -- confirmed real
below, don't shrink it below auto-sizing) or by the memory-pressure
findings in the SF1000 Q3 OOM section above than by a decode-parallelism
gap.

## Opt #2 implemented: bounded concurrent GPU queries (2026-08-21)

Follow-up to the 2026-08-17 investigation above -- implements the "revised
next-session task" it ended on, plus one more real gap that investigation's
own audit missed.

**Design decision made before touching code: bounded, not unbounded
concurrency.** The original opt #2 framing ("drop the mutex") implied
unconditional removal. Two real risks argued against that: (1)
`pass_read_limit_bytes`/`build_side_budget_bytes`
(`query_engine_execute_gpu.cpp`) are each sized as a fraction of the
*entire* device memory ceiling for one query -- N unbounded concurrent
queries can collectively demand up to N times that against one real,
shared GPU budget; (2) the opt #6 prototype (above) already found
concurrent decode streams on this GPU degrade past ~N=2 (up to -190% at
N=16) -- more concurrency isn't automatically more throughput here.
Landed on a configurable semaphore (`EngineSection::max_concurrent_gpu_queries`,
default 2) instead -- real concurrency, but capped.

**Root cause re-verified directly against vendored RMM/cudf source before
implementing** (not re-trusting the 2026-08-17 finding blind):
`rmm::mr::statistics_resource_adaptor` maintains one shared,
**non-thread-local** counter stack (confirmed from its own header:
`read_lock_t`/`write_lock_t` protect *the* stack, singular) -- concurrent
`push_counters()`/`pop_counters()` calls from different query threads
would race and pop the wrong frame. `rmm::mr::limiting_resource_adaptor`
(atomics-based) and `rmm::mr::pool_memory_resource`/`cuda_async_memory_resource`
are both already documented thread-safe -- confirmed, not the blocker,
exactly as 2026-08-17 found.

**Fix: keep the memory *ceiling* global/shared, make only per-query
*reporting* independent.** New `QueryMemoryTracker`
(`kernellake/memory/query_memory_tracker.hpp`) is a small RAII type
wrapping a **fresh** `statistics_resource_adaptor` per query, layered over
`RmmEnvironment`'s existing, already-thread-safe `limiting_resource_adaptor`
-- no shared stack, no push/pop, each query's counters are a genuinely
separate object. `RmmEnvironment::make_query_tracker()` (replacing
`track_query(std::function)`) hands one out per call. This was already a
partially-scaffolded idea: `ExecutionContext::memory_tracker` existed as a
forward-declared-but-never-defined `QueryMemoryTracker*` field before this
change -- filled in for real, not new API surface.
`GpuExecutionCoordinator`'s `std::mutex` became a `std::counting_semaphore`
sized from the new config field, with a small local RAII guard (the
standard library has no `std::lock_guard` equivalent for semaphores).

**The 2026-08-17 audit's gap #1 (`ObjectStoreDatasource::device_read()`)
fixed exactly as scoped**: threaded a `rmm::device_async_resource_ref`
into its constructor (both real construction sites in
`parquet_scan_operator.cpp` already had `context.memory_resource` in
scope), passed explicitly into the one `rmm::device_buffer` allocation
that was missing it.

**Gap #2 (literal-scalar construction) turned out less structural than
2026-08-17 assessed.** That investigation's own audit said 3 of 4
`literal_to_scalar()` call sites "run at plan-compile time, with no
`ExecutionContext` in scope at all -- not a missed pass-through, the
plumbing to carry a resource/stream there doesn't exist yet." Re-checked
directly this session: all three (`ProjectionOperator::compile_value()`,
`HashAggregateOperator::compile_expr()`, `ScalarAggregateOperator::compile_expr()`)
are in fact called from each operator's own `open(ExecutionContext&
context)` -- `context` *was* in scope one level up the whole time. The
real gap was narrower than described: `compile_value()`/`compile_expr()`'s
own signatures (and their private recursive CASE/CAST/LIKE/EXTRACT
helpers) just didn't thread it through. Fixed by adding `ExecutionContext&
context` to `literal_to_scalar()`, `make_decimal_scalar()`, and every one
of these functions' signatures, passing `context.stream`/
`context.memory_resource` into every `cudf::scalar` constructor call
(confirmed from vendored `cudf/scalar/scalar.hpp`: every one --
`numeric_scalar`, `string_scalar`, `timestamp_scalar`, `fixed_point_scalar`
-- defaults both when not passed).

**A third, real gap the 2026-08-17 audit missed entirely, found while
fixing the above**: `ExpressionCompiler::make_literal()`
(`expression_compiler.cpp`) is a *second*, independent literal-scalar
construction path -- used by `FilterOperator`/`SortOperator`/
`HashAggregateOperator`/`ScalarAggregateOperator`/`ProjectionOperator` for
`cudf::ast::literal` nodes, entirely separate from `cudf_adapter.cpp`'s
`literal_to_scalar()`. Same bug: every scalar constructor call in its
13-case type-dispatch switch omitted `stream`/`mr`. The audit's "16 files
in `execution_gpu/`... two real, concrete gaps found" undercounted by one
-- `ExpressionCompiler::compile()` and its 6 private
`compile_column`/`compile_literal`/`compile_binary`/`compile_unary`/
`compile_between`/`compile_cast` helpers all needed the same
`ExecutionContext&` threading, plus updating all 5 operators' own call
sites (`SortOperator::compile_key()` needed the identical treatment, one
level up again).

**The 2026-08-17 "open question" (does `cudf::get_default_stream()`
resolve to the legacy/null stream, a potential cross-query serialization
hazard) is now moot rather than answered.** Every call site that used to
rely on that default (all of the above) now passes `context.stream`
explicitly -- there's no code path left in `execution_gpu/` that still
depends on what the default resolves to. Not resolved as a general cudf
fact, just eliminated as a concern for this codebase specifically.

**Verification**: full build + test suite (145/145 GPU tests, 772/772
relevant unit tests -- 1 pre-existing, unrelated OTel-log-bridge test
ordering flake, confirmed via `git diff` showing zero changes to that
file and the test passing cleanly in isolation) all pass. New regression
test, `GpuExecutionCoordinatorConcurrencyTest.ConcurrentQueriesDoNotMixResultsOrMemoryAccountingAcrossThreads`
(`tests/unit/gpu_execution_coordinator_test.cpp`, `KERNELLAKE_WITH_CUDA`-gated):
4 real concurrent threads (2x the configured `max_concurrent_gpu_queries=2`,
forcing real semaphore queueing) against a real local `GpuExecutionCoordinator`
and real GPU, half filtering `region='A'` (expected SUM 35.0) and half
`region='B'` (expected SUM 110.0) -- confirms zero cross-thread result
mixing and every result's `peak_gpu_memory_bytes` independently sane, the
exact bug class the old shared push/pop stack was vulnerable to.

**Real-hardware throughput confirmed, not just unit-test correctness**:
local `kernellake-server` + real SF10 data + real gRPC Flight SQL clients
on this dev box's GPU (`build/gpu-dev-wsl`), TPC-H Q6, 20s per run:

| Concurrent clients | Queries completed | Queries/hour | Median latency |
|---|---|---|---|
| 1 | 144 | 25,806 | 0.136s |
| 2 | 180 | 32,291 (**+25%**) | 0.219s |
| 4 | 182 | 32,158 (flat) | 0.440s |

Real, positive throughput gain at `N=2` -- a genuine improvement over the
old mutex's confirmed-flat baseline (2026-08-17: 288 vs. 289 queries/hour,
1 vs. 4 clients, effectively zero gain). `N=4` correctly plateaus rather
than climbing further (`max_concurrent_gpu_queries=2` doing its job -- the
extra two threads queue on the semaphore, showing up as roughly doubled
median latency rather than more completed queries), confirming the bound
is real and working as designed, not just present in config. The
sub-2x gain at `N=2` (not a naive-hoped linear 2x) is consistent with opt
#6's own finding that concurrent GPU decode work contends for shared
decompression/copy-engine capacity -- exactly the risk that motivated
choosing a conservative bounded default over assuming free linear
scaling. Whether a larger `max_concurrent_gpu_queries` value nets out
better or worse on this hardware wasn't tested this session -- worth a
follow-up sweep (3, 4, 8) if concurrent-query throughput becomes a live
tuning question.

## Follow-up: GPU memory growing across benchmark runs

**Superseded 2026-08-21**: the mechanism description below (a single
mutex-serialized `execute()` call, `push_counters()`/`pop_counters()`
isolating each query) describes this investigation's own point-in-time
snapshot (2026-08-08), before "Opt #2 implemented: bounded concurrent GPU
queries" below replaced the mutex with a `std::counting_semaphore` and the
push/pop mechanism with a fresh `QueryMemoryTracker`/
`statistics_resource_adaptor` per query (`RmmEnvironment::make_query_tracker()`).
The investigation's own conclusion (device-memory growth across runs is
the async pool's high-water mark, not a leak) is unaffected by that later
change -- only the specific mechanism description below is now historical,
not current.

Observed during AWS GPU-instance benchmark testing (`benchmarks/aws/`):
device memory usage climbed with each successive benchmark run. **Root
cause found and confirmed benign, not a leak** -- see the verified
conclusion below; no fix was needed.

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
`config/kernellake-server.yaml`) backs that long-lived environment with
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

## Fixed: real-S3 scan throughput (device_read/device_read_async)

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

**Fixed: the actual serialization, via `device_read`/`device_read_async`.**
Cloned cudf `v26.06.00` (the exact pinned version -- `cmake/ThirdPartyRapids.cmake`)
to read the real source rather than guess further. In
`cpp/src/io/parquet/reader_impl_preprocess_utils.cu`'s
`read_column_chunks_async()`:

```cpp
if (source->is_device_read_preferred(io_size)) {
  auto fut = source->device_read_async(io_offset, io_size, dest, stream);
  read_tasks.emplace_back(std::move(fut));           // taken as-is -- real concurrency possible
} else {
  read_tasks.emplace_back(std::async(std::launch::deferred, [source, ...] {
    auto const read_buffer = source.get().host_read(io_offset, io_size);  // bypasses host_read_async() entirely
    ...
  }));
}
// ... later, sequentially:
for (auto& task : read_tasks) { task.get(); }
```

For any datasource with `supports_device_read() == false` (the base-class
default -- what `ObjectStoreDatasource` had), cudf never calls
`host_read_async()` at all; it calls the plain synchronous `host_read()`
directly, wraps *that specific call* in its own `std::launch::deferred`,
then waits on every chunk's future in sequence. But for a
*device*-preferred source, cudf takes whatever future `device_read_async()`
returns completely unmodified -- if that future represents work already
launched (not deferred), every column-chunk read for a pass gets kicked
off concurrently in the loop above, and the later sequential `.get()`
calls just wait for work already in flight.

**Fix:** `ObjectStoreDatasource::supports_device_read()` now returns
`true`, with real `device_read()`/`device_read_async()` overrides --
`device_read_async()` launches a real `host_read()` on
`std::launch::async` (same reasoning as the `host_read_async()` overrides
above) and copies the result into device memory via a stream-ordered
`cudaMemcpyAsync`. Safe regarding the host buffer's lifetime because
`ReadAt()` returns *pageable* (not pinned) memory -- CUDA's own documented
behavior is that a host-to-device `cudaMemcpyAsync` from pageable memory
blocks the calling thread until the copy completes, so the buffer is never
touched-after-free despite the call being nominally "async". Metadata/footer
reads are a separate code path (not `read_column_chunks_async`), so
they're unaffected either way -- this targets exactly the large
column-chunk reads that dominated the earlier 533MB/156-call measurement.

**Validated for real, twice:**
- **Correctness**: real DuckDB-cross-validated runs at SF1 and SF10 against
  the local docker-compose stack's MinIO (a genuinely non-local
  `ObjectStoreDatasource` path, just low-latency) -- all 6 queries pass at
  both scale factors, through the new concurrent-copy code path.
- **Performance, real EC2 g6.2xlarge against real S3** (not local-machine-
  to-S3, which an earlier attempt at this measurement used by mistake --
  ~13 MiB/s from this dev box vs. ~60+ MiB/s from an actual same-region EC2
  instance, not a representative comparison): pre/post binaries built from
  the same source tree via `git worktree`, both Docker-built and run on the
  same real `g6.2xlarge` instance, real bucket:

| Query | wall before | wall after | speedup | decode before | decode after |
|---|---|---|---|---|---|
| Q1  | 8.22s | 3.51s | **2.34x** | 6.19s | 0.76s |
| Q6  | 9.07s | 3.07s | **2.95x** | 7.26s | 0.70s |
| Q14 | 13.43s | 2.80s | **4.80x** | 11.52s | 0.88s |
| Q19 | 13.36s | 3.04s | **4.40x** | 11.40s | 0.89s |

Row counts matched exactly between old and new for every query (on top of
the DuckDB cross-validation above). Confirmed the instance's S3 traffic
routes through the real VPC Gateway Endpoint (`aws_vpc_endpoint.s3`, prefix
list `pl-63a5400a`), not the public internet path, so this isn't measuring
an accidentally-suboptimal network route.

These numbers measure `device_read_async` in isolation, against a cold
object store with nothing sitting in front of it. An NVMe-backed local
object cache (`src/storage/nvme_object_cache.cpp`, see
`docs/ARCHITECTURE.md`'s "NVMe cache tier" section) was added the day
after this fix and now sits in front of this same read path whenever
`storage.cache.enabled` is set -- a *repeat* real-S3 scan in the current
codebase would also hit that cache layer, not just this fix's concurrent-
copy path, and hasn't been re-measured with the cache enabled to see how
the two interact. The numbers above remain accurate as a measurement of
this specific fix on a cold/uncached path.

## Fixed: sequential per-file metadata inspection (SF100 latency reversal)

Found investigating a real, full reversal in the SF10-vs-SF100 AWS
benchmark comparison (see `benchmarks/aws/docs/COST_ESTIMATES.md`):
KernelLake beat PySpark on 8/12 query/mode combinations at SF10, but
PySpark won all 10 measured comparisons 2-4x at SF100, even after the
`device_read`/`device_read_async` fix above was already in place. Ran
`kernellake query --stats`/`benchmark tpch` standalone against the same
real SF100 Q1 data (`s3://.../tpch-data/sf100/lineitem-*.parquet`, 120
files) to get a full cost breakdown rather than guess:

```
metadata_inspection_seconds: 5.51s   <- previously invisible in benchmark JSON output
parquet_decoding_seconds:    7.9s
gpu_execution_seconds:       17.5s
wall_seconds:                28.0s
```

`resolve_table()` in `src/io/table_resolution.cpp` was a plain sequential
loop over every discovered file, calling `inspect_parquet_file()` (a full
footer-read round-trip to the backend) one at a time:

```cpp
for (const ObjectInfo& file : files) {
  metadata.push_back(inspect_parquet_file(store, file.uri));
}
```

Same architectural bug class as the cudf `read_column_chunks_async()`
serialization fixed above, but in KernelLake's own code, not a dependency
-- 120 files x ~45ms/file round-trip serialized is almost exactly the
measured 5.5s. SF10 has only 12 files (~0.5s), which is why this never
showed up as a problem until SF100's 120-file lineitem table.

**Fix:** `std::async(std::launch::async, ...)` per file, same pattern
already validated for the `device_read_async` fix -- `ObjectStore::open()`/
the underlying Arrow filesystem's read path are safe for concurrent use
from multiple threads (the same guarantee that fix already relies on),
and each future targets a different, independent file. Futures are
collected in original order (not as they complete) since callers below
(Hive partition-segment extraction, schema validation) assume
`metadata[i]` corresponds to `files[i]`.

**Hardened later** (a separate fix, not part of this SF100 investigation):
the original version above launched one giant `std::async`-per-file burst
for the whole table at once, which was fine for SF100's 120-file
`lineitem` table but would spawn thousands of OS threads at once for a
table accumulated over time with many more, smaller files -- a real
`RLIMIT_NPROC` exhaustion / thundering-herd risk against one object-store
prefix. `resolve_table()` now processes files in fixed-size batches of
`kMaxConcurrentFileInspections = 64` (a network-round-trip-bound
workload, not CPU-bound, so this is deliberately higher than a typical
core-count-based cap), each batch fully awaited before the next batch's
`std::async` calls are made -- see that constant's own comment in
`src/io/table_resolution.cpp`.

**Validated for real, on the same EC2 kernellake host against the same
SF100 data:**

| | before | after | speedup |
|---|---|---|---|
| `metadata_inspection_seconds` | 5.51s | 0.38s | **14.5x** |
| `elapsed_wall_seconds` (Q1) | 28.0s | 22.5s | 1.24x |

Deployed to the live benchmark server and the SF100-vs-PySpark comparison
rerun with both fixes in place; see `benchmarks/aws/docs/COST_ESTIMATES.md`
for the updated head-to-head numbers.

## Fixed: predicate pushdown stopped at joins (SF100 Q3 GPU OOM)

Real OOM found running the SF100 AWS benchmark: TPC-H-derived Q3 (`customer
JOIN orders JOIN lineitem WHERE c_mktsegment = 'BUILDING' AND o_orderdate
< ... AND l_shipdate > ...`) crashed on the g6.2xlarge's single L4 (24GB)
with `RMM failure: Exceeded memory limit (failed to allocate 1.432 GiB)`.
The doc's own opportunity #4 (`pass_read_limit_bytes` retuning) was the
default suspect, but that constant only sizes Parquet *scan* passes --
`HashJoinOperator::open()` (`src/execution_gpu/hash_join_operator.cpp`)
materializes its entire build side via one unbounded `cudf::concatenate()`
call with no size awareness at all, so it was never in the code path that
could have OOM'd from that knob.

**Root cause: `src/optimizer/optimizer.cpp`'s `annotate_scan()` explicitly
did not push predicates across a `LogicalJoin`** ("Predicate pushdown does
not cross a join in this version" -- its own prior comment). For Q3, the
WHERE clause sat in a single `LogicalFilter` above the *entire* 3-way join
chain, so `customer JOIN orders` ran fully unfiltered before anything
checked `c_mktsegment`/`o_orderdate`. Confirmed with real generated data
before writing any fix (not guessed): at SF1, `c_mktsegment = 'BUILDING'`
keeps only 20% of customer, and `o_orderdate < 1995-03-15` keeps 46.4% of
orders -- but because neither filter reached a scan, and orders/customer
have referential integrity, the unfiltered `customer JOIN orders` matched
**~100% of orders** (20.0% when computed correctly, i.e. 5x too many rows
feeding into `HashJoinOperator::open()`'s single unbounded concatenate for
the *outer* join with `lineitem`).

**Fix:** `push_predicate_through_join()` (new, `optimizer.cpp`) applies
`sigma_p(A JOIN B) = sigma_p(A) JOIN B` whenever a WHERE conjunct's columns
belong entirely to one side of a join -- unconditionally valid for the
INNER-only joins this engine supports (`HashJoinOperator`'s own doc
comment: no LEFT/RIGHT/FULL, so no null-extension semantics to worry
about). Splits a WHERE clause's top-level AND conjuncts by which side of
the join their columns reference (column-index range, mirroring the
existing required-columns split `annotate_scan()` already did for
projection pushdown), shifts a right-side conjunct's column indices back
to that side's own local schema, and wraps each pushable conjunct in a new
`LogicalFilter` directly above the correct child -- recursing through
`rewrite_plan()` so it telescopes through an entire N-way left-deep join
chain one level at a time, not just a single join. A conjunct referencing
both sides (or neither, e.g. a constant) is left exactly where it was,
still applied post-join. Once a predicate lands directly on a
`LogicalFilter(LogicalScan, ...)` shape, it's indistinguishable from any
ordinary single-table WHERE clause to every downstream pass -- the
existing `annotate_scan()` scan-pruning path and `find_scan_schema()`-based
physical-planner remapping both already handled that shape, so no other
code needed to change.

**Verified real, three ways:**
- **Plan shape** (`explain --format text` against the real SF1 local
  dataset, `customer`/`orders`/`lineitem` generated via
  `tools/generate_tpch.py --scale-factor 1`): before, one `Filter` sat
  above the outermost `HashJoin` with all three conjuncts ANDed together;
  after, each of the three `ParquetScan`s has its own single-column
  `Filter` directly above it, and no `Filter` remains above either
  `HashJoin`.
- **Correctness**: `tools/validate_tpch.py --query all` against the same
  real SF1 data -- all 6 supported queries (Q1/Q3/Q6/Q12/Q14/Q19) match
  DuckDB row-for-row, including Q3's now-pushed-down 3-way join. New
  optimizer-level regression tests added
  (`tests/unit/optimizer_test.cpp`): a 2-way join with one predicate per
  side, the full 3-way Q3 shape, and a negative case (a cross-side
  predicate correctly stays unpushed, above the join). 332 unit + 102 GPU
  tests pass.
- **Real memory win**, controlled local A/B (pre/post binaries from the
  same source tree via a throwaway `git worktree`, real GPU, identical
  SF1 query): `peak_gpu_memory_bytes` for Q3 dropped from **1.28 GiB to
  0.42 GiB (3.07x)**.
- **Confirmed at real SF100 scale, on a real EC2 g6.2xlarge, against the
  actual data that originally OOM'd.** With only this fix, Q3 no longer
  OOM'd on the join -- but ran into a second, unrelated, pre-existing
  guard: `HashAggregateOperator`'s `max_distinct_keys` safety cap (real
  SF100 Q3 produces ~10.8M distinct `(l_orderkey, o_orderdate,
  o_shippriority)` groups before its `ORDER BY ... LIMIT 10` trims the
  result, just over the old 10M default). Fixed by doubling that default
  to 20M (a fixed safety cap with no cardinality estimation behind it, so
  headroom matters more than precision -- see the constant's own updated
  comment in `hash_aggregate_operator.hpp`). With both fixes, **Q3
  completes end to end at real SF100**: `peak_gpu_memory_bytes` 9.9 GiB
  (well under the L4's 24GB), 10 correct rows returned,
  `elapsed_wall_seconds` 32.2s. The original crash
  (`RMM failure: Exceeded memory limit`) is gone.

**Open again at SF1000 (2026-08-15) -- same failure mode, 10x the scale.**
A real SF1000 AWS run (same `g6.2xlarge`/L4, both fixes above already in
place) hit the identical OOM shape on Q3:
`RMM failure: Exceeded memory limit (failed to allocate 328.06 MiB)`,
via `adbc_driver_manager.OperationalError` over Flight SQL, and it takes
the whole `aws_benchmark_runner.py` process down with it (no per-query
exception handling there -- one query's failure aborts the run before any
results are written, a separate harness-level gap from the OOM itself).
Not yet root-caused the way the SF100 case was -- SF100's fix (predicate
pushdown + `max_distinct_keys` 10M -> 20M) got real headroom (9.9 GiB
peak, well under 24GB) at that scale, but SF1000's ~10x larger post-filter
join inputs and proportionally larger distinct-group cardinality
apparently exceed it again. Live `nvidia-smi` sampling during this same
run showed device memory pegged at ~20.6/23GB (~89.5%) even on the
queries that *don't* OOM (Q1/Q6/Q12/Q14/Q19), which is what opportunity #6
above cross-references -- headroom is already tight before Q3 runs at
all, so anything that increases concurrent in-flight GPU memory (like
opportunity #6's multi-stream decode) needs to be weighed against this,
not layered on top of it for free. Worked around for now by excluding Q3
from the SF1000 `--query` list rather than blocking the rest of the run
on a live root-cause; retuning `pass_read_limit_bytes` (opportunity #4,
still open) or moving to a larger-memory GPU instance
(`g6.4xlarge`/`g6.8xlarge`, unlocked by the 2026-08-11 quota increase) are
the two most direct levers, but neither has been tried yet at this scale.

**Actually root-caused 2026-08-16 -- neither of the above levers was it.**
The real cause: `HashJoinOperator` already has a grace-hash partitioned/
disk-spilling build-side path (added `b915063`, 2026-08-13, for the
Q12/Q14/Q19 host-RAM OOMs -- bounds *device* memory to one partition/batch
at a time, not just host RAM as an earlier read of this history assumed;
`docs/ARCHITECTURE.md`'s own build-side-selection section was stale on
this point until today), and it targets exactly this GPU-OOM failure mode
when it engages. It just never engaged for Q3: `choose_partition_count()`
(`hash_join_operator.cpp`) sizes off `HashJoinNode::estimated_build_rows()`,
which for Q3's `(customer JOIN orders) JOIN lineitem` left-deep chain came
from `estimate_row_count()`'s own `min(left, right)` fallback for a nested
join used as an outer join's operand (`physical_planner.cpp`, same
function `estimate_selectivity()` was added to for the Q12 fix below).
`customer JOIN orders` is a foreign-key join -- most of the *larger* side
(orders) survives, since real referential integrity means most orders
match some surviving customer row -- so the true inner-join output tracks
`orders_filtered` (the larger input), not `min(customer_filtered,
orders_filtered)`. At SF1000 that `min()` badly undersized the estimate,
kept it under `build_side_budget_bytes`, `choose_partition_count()`
decided partitioning wasn't needed, and the outer join's real (much
larger) build side then hit `HashJoinOperator::open()`'s unbounded
`cudf::concatenate()` -- the exact `RMM failure: Exceeded memory limit`
this section describes, now explained down to the specific wrong number
that caused it.

**Fix:** `estimate_row_count()`'s `HashJoinNode` case now returns
`max(left, right)` instead of `min(left, right)`. Deliberately the
conservative direction to be wrong in -- see that function's own updated
comment -- overestimating a nested join's size can only trigger
partitioning/build-side swaps more readily than strictly necessary (cheap),
never less (the failure mode this bug actually caused). Verified so far:
real pre/post `git worktree` A/B confirms Q12's own build-side swap still
picks correctly (this change is a different code path, but shares the
same function); Q3 matches DuckDB row-for-row on real local SF10 data
(`peak_gpu_memory_bytes` 2.89 GiB there, well under any budget at that
scale, so SF10 doesn't exercise the actual bug); full unit + GPU test
suites pass (one pre-existing, unrelated `QueryTracingTest` flake, not
caused by this change). **Not yet confirmed at real SF1000 scale** -- that
needs a real AWS run with Q3 included, the same way every other fix in
this section was ultimately validated. Q3 is still excluded from the
`--query` list pending that confirmation.

**Update 2026-08-17**: confirmed at real SF1000 scale. Q3 completed
successfully in the SF1000 v3 AWS run (queries 1/6/12/14/19) -- but Q3
itself still had to be excluded from that run, for an unrelated reason:
`HashAggregateOperator`'s `max_distinct_keys` cap doesn't converge at this
scale (real cardinality confirmed >77M and climbing, the cap was already
raised once from 20M to 75M and still wasn't enough -- see
`docs/ARCHITECTURE.md`'s `HashAggregateOperator` row and the commit
history around `02b8293`). The join-OOM this section fixes and the
aggregate-cap issue are separate bugs in separate operators; fixing one
doesn't fix the other.

## Confirmed: opportunity #4 (`pass_read_limit_bytes` retuning), real A/B (2026-08-17)

Real controlled A/B against SF1000 `lineitem` on AWS (`g6.4xlarge`),
via `kernellake benchmark tpch` run directly (bypassing Flight SQL --
the CLI binary ships inside the published `runtime-gpu` image at
`/opt/kernellake/bin/kernellake`, usable standalone with `--config`).
Auto-sizing (`engine.query_memory_limit_bytes: 0`, 90% of free VRAM at
startup, `resolve_query_memory_limit_bytes()`) vs. an explicit, heavily
constrained override:

| Config | `pass_read_limit_bytes` | wall | decode | compute-only | peak GPU mem |
|---|---|---|---|---|---|
| auto (≈21GB limit) | ≈5.3GB | 209.3s | 118.9s (57% of wall) | 79.6s | 3.17GB |
| explicit `query_memory_limit_bytes: 2GiB` | ≈536MB | 495.0s | 408.9s (83% of wall) | 76.1s | 288MB |

Shrinking the pass size ~10x made decode **3.44x slower** and wall time
**2.36x slower**, while compute-only time stayed flat (79.6s vs 76.1s,
within noise) -- clean confirmation this is a scan/decode pass-count
effect (more, smaller `cudf::chunked_parquet_reader` passes, each paying
fixed per-pass `has_next()` round-trip overhead -- see the real-S3 fix
above for why `has_next()`, not `read_chunk()`, is where the real cost
lives), not a GPU compute artifact.

**Conclusion: don't shrink `query_memory_limit_bytes` below what
auto-sizing already picks.** Could not test going *bigger* than auto in
this same experiment -- auto already claims ~90% of free VRAM (little
room left before the literal device ceiling), and this query's actual
peak usage (3.17GB) was already far below even the auto ceiling (~21GB),
so raising the limit further wouldn't have grown the effective pass size
anyway. **Still open**: whether there's a sweet spot *between* auto's
~5.3GB and some larger explicit value, or whether it plateaus once passes
are "big enough" -- would need a query/dataset where auto-sizing's actual
peak usage gets closer to its own ceiling to test meaningfully.

## S3 vs. local NVMe: quantifying how much of "cold" wall time is network, not compute (2026-08-18/19)

Motivated directly by the SF1000 v3/v4 finding that the cold/S3
benchmark methodology is "S3-request-latency-bound, not compute-bound"
(sustained S3 RX 250-700MB/s, GPU bursty 4-99% never pegged, CPU never
above ~8% -- see `project_sf1000_v3_benchmark.md`). Rather than continue
reasoning about this qualitatively, staged the real SF1000 S3 dataset
(181.25GB, confirmed via `aws s3 ls --summarize`) onto a `g6.4xlarge`'s
own local NVMe instance-store (`/opt/dlami/nvme`, already unconditionally
bind-mounted to the container's `/cache` regardless of the cache
*feature* flag -- an existing fix from the SF1000 v3 run, see
`kernellake-host-init.sh`) via `aws s3 sync` (8m22s, same-region, no
transfer cost), then re-ran the same 5 queries pointed at the local path
instead of `s3://`.

**Result -- S3-cold vs. local-NVMe-cold, same queries, same data, `event_loop_threads=4` baseline:**

| Query | S3 (v4 baseline) | Local NVMe | Speedup |
|---|---|---|---|
| Q1 | 180.6s | 57.2s | 3.2x |
| Q6 | 163.9s | **9.5s** | **17.3x** |
| Q12 | 236.7s | 40.3s | 5.9x |
| Q14 | 227.7s | 79.6s | 2.9x |
| Q19 | 214.1s | 81.1s | 2.6x |

Confirms and quantifies the qualitative "S3-latency-bound" hypothesis --
S3 request/network latency is the majority of cold wall time, not GPU
decode/compute, for every query tested. Q6 is the extreme case: 163.9s
over S3 collapses to 9.5s locally, meaning **~94% of KernelLake's S3-cold
wall time on Q6 was pure network wait**, not GPU work at all -- consistent
with Q6 being the query earlier profiling flagged as almost pure
scan+decode (65-77% of wall time), the shape with the least other
CPU-side work to hide network latency behind.

**Same test run for DuckDB and PySpark on the identical machine/disk/data**
(not a separate CPU instance -- see below for why), giving a genuine
same-hardware three-way comparison with S3 removed entirely:

| Query | KernelLake | DuckDB | PySpark (`local[*]`) |
|---|---|---|---|
| Q1 | 57.2s | 49.0s | 207.8s |
| Q6 | **9.5s** | 36.8s | 34.2s |
| Q12 | 40.3s | 45.8s | 118.4s |
| Q14 | 79.6s | 68.2s | 92.3s |
| Q19 | 81.1s | 67.1s | 71.7s |

With S3 removed, the head-to-head **flips from "DuckDB wins every query"
(the S3-bound v3/v4 result) to a mixed picture**: KernelLake wins Q6
(3.9x) and Q12 (12%) decisively; DuckDB wins Q1, Q14, Q19 by a real but
modest 10-17% margin. PySpark's single-JVM `local[*]` mode is clearly the
weakest of the three (3-4x slower than the other two on Q1 especially) --
not a real multi-executor cluster, so not necessarily representative of
Spark's best possible showing, just what's comparable on identical
hardware.

**DuckDB's own S3-vs-local speedup is much smaller than KernelLake's**
(computed from the same clean v4 DuckDB S3 numbers: Q1 164.7s, Q6 147.5s,
Q12 207.7s, Q14 236.6s, Q19 250.0s): DuckDB's Q6 speedup going local is
4.0x vs. KernelLake's 17.3x. This makes sense given the mechanism above --
KernelLake's real GPU compute time for Q6 is so fast (9.5s) that S3 wait
dominated its S3-cold number almost completely, while DuckDB's slower CPU
compute (36.8s) was already a bigger fraction of its own S3-cold wall
time, so removing S3 recovers proportionally less.

**Why this matters beyond just "S3 is slow": it changes what "the DuckDB
gap" actually means.** The earlier S3-bound v3/v4 result (DuckDB wins
5/5) was measuring mostly network latency, not either engine's query
execution quality -- a real result for that specific cold/no-cache/S3
methodology, but not evidence about which engine computes faster. The
local-NVMe result above is the first real, clean measurement of that
question, and it says KernelLake's actual query execution beats DuckDB's
on 2/5 tested queries and trails by only 10-17% on the other 3 -- a much
smaller and more mixed gap than the S3-bound numbers implied.

### Hardware notes gathered alongside this test

**GPU instance (`g6.4xlarge`) real specs, confirmed not assumed:**
AMD EPYC 7R13 (AWS's own custom Milan variant), 1 socket / 8 physical
cores / 16 threads, ~3.6GHz observed, AVX2 (no AVX-512), single 600GB
NVMe instance-store disk. Real `fio` sequential read (`iodepth=32`,
`numjobs=4`, `ioengine=libaio`, `direct=1`, 1M blocks -- a naive
`iodepth=1` single-job run badly undersells modern NVMe, real number
needs queue depth): **1072 MiB/s (1124 MB/s)**.

**Went looking for a same-price/same-spec CPU-only instance for a fair
NVMe comparison, found real constraints along the way**: `r6in.4xlarge`
(this project's existing CPU-engine instance type) has **no local
instance store at all** (`InstanceStorageInfo: null`, confirmed via the
EC2 API) -- it's EBS-only, not a fair "local NVMe" comparison target.
`r5d.4xlarge` (closest price match, $1.152/hr) has real local NVMe but as
**two** 279GB disks, not one 600GB disk like `g6.4xlarge` -- single-disk
`fio` read there measured 548 MiB/s (574 MB/s), about half of the GPU
instance's number, but understates what RAID-0'ing both disks would give.
No CPU-only instance type in the whole current-generation catalog has an
exact single 600GB disk -- confirmed by querying every instance type with
`Disks[0].Count==1`, that specific size is GPU-family-specific (g6/g6e/
g4ad only); closest single-disk CPU match is ~474-480GB
(`m6id.2xlarge`/`r6id.2xlarge`/`i3.large`).

**Found the real "g6.4xlarge minus the GPU" equivalent**: `m5ad.4xlarge`
-- confirmed via `ProcessorInfo.Manufacturer` that `g6.4xlarge` is
AMD-based (matches the `/proc/cpuinfo` EPYC 7R13 read above), and
`m5ad.4xlarge` matches vCPU (16), RAM (64GB) exactly, same 600GB total
NVMe capacity (though still 2x300GB disks, not 1x600GB), same AMD
lineage, at $0.824/hr (62% of `g6.4xlarge`'s price -- roughly what the L4
GPU itself costs on top). Not provisioned/tested this session -- a real
candidate for a future fair CPU-vs-GPU comparison if one is needed again.

**Ultimately used a simpler, confound-free design instead of any separate
CPU instance**: ran DuckDB and PySpark directly on the GPU instance's own
16 real vCPUs, reading the identical already-staged local NVMe data --
eliminates the entire "is CPU-instance NVMe comparable to GPU-instance
NVMe" question by construction (one physical disk, both engines). This is
the three-way table above. The separate-CPU-instance research (previous
two paragraphs) remains useful groundwork if a *dedicated*, unshared CPU
benchmark host is ever wanted again, but wasn't necessary for this
specific comparison.

**Infra**: fully torn down via `teardown.sh` (no `--purge-data`) --
confirmed via `describe-instances` (no running/stopped/pending instances)
and via the expected `BucketNotEmpty` teardown error, confirming the S3
bucket (real SF1000 data, 181.25GB/1586 objects) survived intact.

## CPU decode offload: real-scale investigation (2026-08-19)

Motivated by the decode-bound finding above plus two facts already on
record: CPU utilization sits near-idle during real cold scans (~8%, SF1000
v3/v4), and Snappy (this project's default codec) is specifically designed
to decompress fast on CPU, not just GPU. Investigated whether offloading
some decode work to the otherwise-idle CPU cores could raise combined
throughput, via a real `g6.4xlarge` (never assumed from the dev box).

**Warm (page-cache-resident) comparison misleads badly.** Same SF10
`lineitem` data, same 4 columns (`l_shipdate`, `l_quantity`,
`l_extendedprice`, `l_discount`, Q6's shape): GPU decode 0.19-0.21s vs.
`pyarrow` (Python) 0.46-0.48s -- GPU 2.3-2.5x faster. This number does
**not** hold at real scale or under real cold conditions (see below) --
it was measuring pure in-memory decompression throughput, not what
actually matters for a real cold scan.

**Real SF1000 cold-scale comparison, same instance, same data, 3 clean
reps each (after rep-0 cold-start noise, same pattern as every other
`--stats` cold measurement this project uses):**

| | rep 1 | rep 2 | rep 3 |
|---|---|---|---|
| GPU (`cudf`, `kernellake benchmark tpch --mode cold`) | 53.406s | 53.415s | 53.404s |
| CPU, `pyarrow` (Python, streamed via `iter_batches`) | 54.209s | 54.212s | 54.211s |
| CPU, native `libarrow`/`libparquet` (C++, same streaming discipline) | 54.227s | 54.218s | 54.256s |

**CPU is within ~1.5% of GPU at real scale -- essentially tied**, a world
away from the warm test's 2.3-2.5x GPU win. Confirmed the mechanism, not
just the number: native C++ (eliminating all Python-layer overhead) shows
the *same* ~1.5% gap as `pyarrow`, so none of it was Python overhead --
it's a small, genuine CPU-vs-GPU decode-speed difference specifically
under real cold-disk conditions. The warm test's advantage for GPU
disappears once disk I/O dominates wall time: when threads spend most of
their time blocked in a `read()` syscall (which releases the GIL, and by
extension hides most language-layer overhead behind the wait), the
critical path becomes the NVMe device's own aggregate service time, not
CPU-side bookkeeping -- same principle as this project's own
decode/compute stream-overlap architecture (`ParquetScanOperator::prefetch_loop()`),
just operating between disk I/O and CPU compute instead of GPU decode and
GPU compute.

**Real bug found and fixed along the way, worth keeping in mind for any
future CPU-side prototyping**: `pyarrow.parquet.ParquetDataset(...).read()`
materializes the *entire* result table in host RAM at once. A first
attempt at the cold SF1000 test OOM-killed (`dmesg` confirmed:
`anon-rss:62122300kB` on a 60GB-RAM `g6.4xlarge`) -- SF1000 `lineitem`'s
~6B rows across 4 columns doesn't fit as a single materialized table.
Fixed by streaming per-file via `ParquetFile.iter_batches()`, discarding
each batch after counting -- the same bounded-memory discipline `cudf`'s
own `chunked_parquet_reader` already uses (`has_next()`/`read_chunk()`
passes, never the whole table at once). Any real hybrid-decode
implementation needs this same discipline on the CPU side, not a naive
whole-table read.

### Concurrent GPU+CPU decode: real contention, not free parallelism

Given CPU and GPU decode are now confirmed near-equal at real scale, the
natural next question: does running them *concurrently* on independent
subsets of data actually deliver combined throughput, or is there hidden
contention (echoing opt #6's GPU-side finding above)? Split 1200 real
`lineitem` files in half, decoded each half on GPU (`half-a`) and CPU
(`half-b`) at the same time.

**On real disk**: GPU half (600 files) -- 26.25s, essentially exactly
proportional to solo (53.4s / 2 = 26.7s expected), no apparent slowdown.
CPU half (600 files) -- 35.09s, vs. ~27s expected if scaling proportionally
from solo -- a real ~30% slowdown. Asymmetric: GPU unaffected, CPU
degrades.

**Isolated whether NVMe bandwidth was the cause, per a real suggestion to
test on a RAM disk instead of real disk** -- 35GB `tmpfs` mounted inside
the already-bind-mounted `/cache` tree (real Docker lesson hit here too:
a tmpfs mounted *after* a container starts does not propagate into that
container's existing bind mount, even though the host-side path is
identical -- confirmed via `/proc/mounts` inside the container still
showing the underlying `ext4`, not the new `tmpfs`; fixed by
`docker compose up -d --force-recreate` so the container's mount
namespace re-resolves against current host mount state). With 200 files
(100 each) copied as real bytes (not symlinks -- relative symlinks
crossing a narrower bind-mount boundary was a second real bug hit and
fixed along the way, resolved by mounting the shared parent directory
instead of just the target subdirectory) into the ramdisk:

| | Solo (ramdisk) | Concurrent (ramdisk) | Change |
|---|---|---|---|
| GPU | 0.862s | 0.870s | +0.9% |
| CPU | 1.238s | 1.502s | **+21%** |

**Contention persists even with disk I/O completely eliminated, same
asymmetric shape.** This rules out NVMe bandwidth as the sole
explanation -- there is no disk involved at all here. Root-caused via a
fact already on record from the earlier cuFile/GDS investigation: this
GPU is confirmed stuck in cuFile *compat mode* (`gdscheck`, both the dev
box and real `g6.4xlarge` -- see "GPU concurrency mutex" section above),
meaning GPU decode already routes through real host CPU threads for its
bounce-buffer copy (`execution.max_io_threads`, default 4) -- "GPU
decode" on this hardware was never purely GPU-side work to begin with.

**Tested whether reducing that thread count recovers the lost CPU
throughput -- it does not.** Built a custom `cufile.json`
(`max_io_threads: 1`, bind-mounted into a recreated container) and
re-ran the same ramdisk concurrent test:

| `max_io_threads` | Solo CPU | Concurrent CPU | Slowdown |
|---|---|---|---|
| 4 (default) | 1.238s | 1.502s | +21% |
| 1 | 1.193s | 1.432s | +20% |

**Essentially identical degradation regardless of thread count** -- the
hypothesis that this is thread/core-scheduling contention is wrong.
Better explanation, consistent with what didn't change: **host memory
bandwidth**, not thread count. GPU's compat-mode path moves real bytes
through host RAM (the bounce-buffer copy) regardless of how many threads
do it -- one thread saturating memory bandwidth contends with CPU
decompression's own memory-bandwidth-heavy work (read compressed, write
decompressed, all through RAM) just as much as four threads would. This
is a physical resource constraint, not a tunable config value.

**Practical implication for any future hybrid-decode work**: the
achievable ceiling is not the naively-hoped-for ~2x from "two independent
resources" -- it's bounded by shared host memory bandwidth, with CPU
paying a real but bounded ~20-21% tax whenever it runs concurrently with
GPU's own compat-mode I/O. Still very likely a net positive (CPU
near-parity with GPU minus a ~20% tax under contention is still
meaningfully better than GPU decode alone), but the design target should
be that real number, not an idealized one -- and the memory-bandwidth
constraint is not something a `max_io_threads`-style config knob can fix.

**Infra**: torn down via `teardown.sh` (no `--purge-data`) after this
investigation, S3 data confirmed to survive.

### Decompression-only cost, isolated from disk I/O, confirmed at real scale (2026-08-19)

Motivated by a real design question raised while evaluating a possible
decoded-result cache (see below): how much of the ~54s real cold-scale
decode number above is genuinely decompression *compute*, versus disk
I/O wait? Answers this directly rather than inferring it from the
CPU-vs-GPU convergence alone. Method: read every file's raw bytes into an
in-memory `arrow::Buffer` first (untimed -- real disk I/O happens here,
once), then time *only* the Parquet decode step running against
`arrow::io::BufferReader` (in-memory, zero syscalls in the timed portion
at all).

**Real bug hit and fixed along the way, same class as the earlier
`pyarrow.read()` OOM**: the first attempt read whole files (all ~16
`lineitem` columns) into memory rather than just the 4 columns actually
needed -- at real SF1000 scale that's the full ~169GB table, and the
process was killed (`exit 137`) approaching the 60GB RAM ceiling before
even reaching the timed phase. Fixed by testing against a 300-file
subset (1/4 of the full 1200-file dataset, ~34GB of whole-file bytes --
comfortably inside the RAM budget) and scaling the result, rather than
building the more complex column-projected-byte-range version this
session didn't have time for.

**Real result**: 300 files, 1.5B rows, decode-only (in-memory) = **2.886s**.
Scaled 4x for the full dataset: **~11.5s** decode-only, against the real
full-dataset cold combined time of ~53.4-54.2s (GPU and CPU converged
there, see above) -- **~21-22% decompression compute, ~78-79% disk I/O
wait**. Closely matches a same-night SF10-based extrapolation (~24%/76%)
made before this real-scale confirmation, validating that the linear
scaling assumption held up -- this is now a measured number at real
scale, not a projection.

**Why this matters for a decoded-result cache design (discussed the same
night, not yet built)**: a cache storing decoded results on *disk* would
save at most that ~21-22% decompression-compute slice, while making I/O
*worse* (decoded data is larger than compressed, by construction) -- very
likely a net loss for repeated cold-disk reads, not a win. A cache
storing decoded results in *host RAM* instead eliminates both the
decompression compute *and* the disk I/O wait, targeting the ~79% that
actually dominates real decode cost, not the ~21% sliver. This is a real,
now evidence-backed reason to prefer a RAM-tier cache design over a
disk-tier one for this specific workload, not just an architectural
preference.

A concrete candidate for such a cache, discussed but not built this
session: `payload-manager` (a separate project,
github.com/hurdad/payload-manager) -- its C++ client has explicit
`TIER_RAM`/`TIER_GPU` allocation, tight native `arrow::Buffer` integration
(`buffer->data()`/`mutable_data()` are literally `arrow::Buffer`'s own
API), and lease-based concurrent-read safety. Real gaps found by reading
its actual example code (not just documentation): no semantic-key lookup
(payloads are server-generated UUIDs; KernelLake would need its own
file+row-group+columns -> UUID index), no chunked/incremental write for a
single large payload (cache at row-group/file granularity instead, which
fits `cudf`'s own chunking discipline anyway), and automatic
tier-eviction-under-pressure is claimed in the README but not confirmed
in the examples reviewed. None of these are disqualifying, but worth
confirming before committing engineering time.

**Infra**: torn down via `teardown.sh` again after this follow-up test,
S3 data confirmed to survive.

## Real GDS via FSx for Lustre + EFA: tested for real, found a real kernel-level ceiling (2026-08-19/20)

Follow-up to the compat-mode findings above. NVIDIA's own GDS docs state
cloud environments generally run in compat mode, but call out one
specific AWS-certified real-GDS path: Amazon FSx for Lustre with EFA
(Elastic Fabric Adapter), not local instance-store NVMe at all -- a
categorically different mechanism (RDMA over the network to a remote
Lustre server, bypassing host memory on the client end) than the
local-disk DMA path `g6.4xlarge`'s compat-mode result was about. Tested
this directly rather than staying at the docs-reading stage.

**Real setup, built manually via AWS CLI (not terraform -- kept
exploratory)**: `aws_fsx_lustre_file_system`-equivalent via `aws fsx
create-file-system`, `PERSISTENT_2` deployment, `EfaEnabled=true`,
`ImportPath` linked to the real SF1000 S3 data; `g6.4xlarge`, our
existing GPU class, is not EFA-capable (confirmed via the API --
`EfaSupported: false`) -- `g6.8xlarge` (same L4, 32 vCPU, `$2.01/hr`) is
the smallest EFA-capable option on this GPU class, launched with an
explicit `InterfaceType: efa` network interface.

**Real, load-bearing minimum-size constraint found**: EFA-enabled Lustre
filesystems have a fixed minimum *aggregate throughput floor* (~4.7 GB/s,
confirmed empirically -- AWS's own creation-time validation rejected
1200 GiB with "minimum storage capacity ... is 19200" at
`PerUnitStorageThroughput=250`, and rejected again at 4800 with
`PerUnitStorageThroughput=1000` before finally accepting), independent of
the general 1200 GiB minimum that applies to non-EFA filesystems. Real
minimum for an EFA-enabled filesystem: **4800 GiB, ~4687.5 MB/s
provisioned throughput** -- roughly $4.29/hr for FSx alone (storage +
throughput), on top of `g6.8xlarge`'s $2.01/hr.

**Real infra bugs found and fixed en route** (same pattern as every other
live AWS session this project has run):
- FSx creation failed with `InvalidNetworkSettings` until the security
  group had an explicit *self-referencing, all-protocol* (not just TCP)
  inbound *and* outbound rule -- the existing security group's broad
  `0.0.0.0/0` egress wasn't sufficient; EFA specifically requires the
  self-reference.
- **Ubuntu 26.04 (this project's current AMI baseline) doesn't support
  the Lustre client at all** -- AWS's own client-repo docs only list
  16.04/18.04/20.04/22.04. Confirmed for real: `apt-get update` against a
  freshly-added FSx-Lustre apt repo hung indefinitely for a codename
  ("resolute") the repo doesn't carry packages for. Fixed by switching
  this one test instance to the Ubuntu 22.04 Deep Learning AMI (published
  2 days prior) -- the GDS question doesn't care which Ubuntu version,
  only this one client-install step does.
- Requesting `InterfaceType: efa` at launch only provisions the
  *hardware* interface -- the EFA *software* stack (kernel driver,
  libfabric, RDMA userspace bits) needs the separate
  `aws-efa-installer` package, confirmed via a real
  `fi_pingpong` test showing actual data transfer post-install.
- No `/etc/cufile.json` exists by default on any instance tested this
  project (same finding as the compat-mode investigation) -- cuFile has
  no RDMA device configured to even attempt loading
  `libcufile_rdma.so` against. Fixed by finding the real device name via
  `ibv_devices` (`rdmap47s0` on this instance) and setting
  `rdma_dev_addr_list` in a real `/etc/cufile.json` -- flipped
  `--rdma devices` from "Not configured" to "Configured" (tried both the
  raw device name and the interface's IP address, `10.90.1.78` -- same
  result either way, ruling out a naming-format issue specifically).

**Real, final result: a genuine kernel-level ceiling, not a config gap.**
Even with RDMA devices "Configured", `gdscheck` still reported
`--rdma_device_status: Up: 0 Down: 1` and `use_compat_mode: true`.
`dmesg` gave the real, concrete root cause:
```
efa 0000:2f:00.0 rdmap47s0: Failed to process command REG_MR (opcode 7) err -22
efa 0000:2f:00.0 rdmap47s0: Failed to register mr [-22]
efa: Acquired peer memory using P2P
```
`err -22` is `EINVAL`. The sequence is telling: peer-memory *acquisition*
of the GPU's pages succeeds (`Acquired peer memory using P2P`), but the
actual hardware *memory-region registration* with the EFA NIC fails
immediately after -- the same failure the EFA installer's own
`fi_pingpong` smoke test had already flagged
(`Failed to register CUDA buffer with the EFA device`). `ulimit -l` was
confirmed `unlimited`, ruling out the classic locked-memory RDMA gotcha.

**This is consistent with something under-weighted earlier**: AWS's own
docs specifically validate GDS-over-FSx-for-Lustre-EFA for **P5, Trn1,
and Hpc7a** -- not "any EFA-capable GPU instance." `g6.8xlarge` genuinely
has EFA hardware (confirmed via the API), but L4 may not be validated for
the GPU-memory-region-registration path this needs the way H100 (on
`p5`) is. This reframes the earlier "EFA doesn't work with 4xlarge"
observation -- it's not about instance size, it's that AWS's own
supported list was the real constraint the whole time, and this session
independently, empirically reproduced *why* (a real kernel-level REG_MR
failure) rather than just citing the docs.

**Not yet tested**: `p5.4xlarge` (H100, real EFA support, on AWS's
validated list, $6.88/hr, fits the current 32 vCPU quota) -- the natural
next test of this same hypothesis if it's ever revisited, now with a
concrete, reproducible failure signature (`REG_MR err -22` in `dmesg`) to
check against rather than a blank `use_compat_mode: true`.

**Infra**: FSx filesystem and EFA-enabled EC2 instance both created via
raw AWS CLI (deliberately not terraform, per explicit instruction, to
keep this exploratory work out of the shared benchmark module) --
torn down after this investigation.

**Attempted same night: re-test on `p5.4xlarge` (H100, on AWS's
validated GDS list) -- inconclusive, blocked on real capacity, not a
technical finding.** Real `InsufficientInstanceCapacity` errors across
repeated attempts in both `us-east-1a` and `us-east-1b` over ~30+ minutes
(a background retry loop alternating both AZs every 45s, ~18+ rounds,
consistently failed) -- AWS's own error messages listed *different*
"available" AZs on nearly every retry, suggesting genuinely volatile,
fast-moving H100 demand region-wide rather than a stable per-AZ gap. Cut
losses on 2 real FSx-filesystem create/delete cycles chasing AZs the
error messages claimed had capacity, before switching to a
launch-first/FSx-second retry strategy (still unsuccessful) -- worth
remembering next time: check instance launch succeeds *before*
committing to an FSx filesystem in that AZ, and expect `p5.4xlarge`
specifically to need real patience or off-peak timing, not something to
assume is readily available on-demand. Not a finding about GDS itself --
whether H100 actually clears the `REG_MR` failure L4 hit is still
genuinely open.

**Final capacity-chase result (2026-08-20): confirmed sustained,
region-wide shortage across all three US regions that offer
`p5.4xlarge`, not a transient blip.** Two more retry loops, run in
parallel:

- `us-east-1`, all 6 AZs (`1a`-`1f`), full EFA-enabled launch config
  (the actual test config): 60 rounds x 6 AZs = 360 attempts total,
  **zero successes**, every single one `InsufficientInstanceCapacity`.
  This exhausted the retry budget completely -- a definitive result, not
  an inconclusive one.
- `us-east-2` + `us-west-2` (the only other two US regions offering
  `p5.4xlarge`; checked via `describe-instance-type-offerings` across all
  AWS regions -- it's also offered in `eu-west-2`, `ap-south-1`,
  `ap-northeast-1`, `ap-southeast-2`, `sa-east-1`, not tested), all 7 AZs
  combined, plain-launch capacity probe (no EFA, just checking raw
  availability): stopped by user request at round 18/60, also zero
  successes.

Conclusion: as of 2026-08-20, `p5.4xlarge` (H100) capacity is
genuinely scarce across all of `us-east-1`/`us-east-2`/`us-west-2`
simultaneously, not just one region having a bad day. Whether H100
clears the L4 `REG_MR err -22` failure remains untested and open --
this was purely an availability problem, never reached the point of
running the actual GDS test. If revisited, worth trying non-US regions
(offered but unchecked) or an on-demand-capacity-reservation /
EC2 Capacity Blocks approach instead of spot-checking `RunInstances`.

Leftover AWS state from this chase (all empty/no cost beyond the
control-plane resources themselves, not yet torn down): `us-east-1`
VPC `vpc-0cefd9a0c17162a9c` (6 subnets, 1 security group, 1 IAM
instance profile) from the original EFA-config attempt, plus new
probe VPCs `vpc-04b0de8b7d171e6e9` (`us-east-2`, 3 subnets) and
`vpc-05221f4b6e3df1737` (`us-west-2`, 4 subnets). No FSx filesystem
or running EC2 instances exist from this round.

**Round 2, same night: re-run in full, same result -- confirms this
isn't a transient blip.** Both loops re-launched against the same
existing infra (same VPCs/subnets/AMIs). `us-east-1`: full 60-round x
6-AZ budget again, 360/360 attempts, zero successes. `us-east-2` +
`us-west-2`: full 60-round x 7-AZ budget this time (round 1 was
user-stopped early at round 18 previously), 420/420 attempts, zero
successes. Two consecutive full-budget exhaustions across all three
US regions offering `p5.4xlarge` now on record. Still untested: the 5
non-US regions that also offer it (`eu-west-2`, `ap-south-1`,
`ap-northeast-1`, `ap-southeast-2`, `sa-east-1`), and EC2 Capacity
Blocks / On-Demand Capacity Reservations as an alternative to
spot-checking `RunInstances`.

## Real GDS via FSx for Lustre + EFA, take 2: `g7e.12xlarge` (RTX PRO 6000 Blackwell), different failure mode (2026-08-21)

While `p5.4xlarge` stayed capacity-blocked, AWS's new `g7e` family
(NVIDIA RTX PRO 6000 Blackwell Server Edition, launched ~2026-01)
turned out to also support real GDS via FSx+EFA on its multi-GPU
sizes -- confirmed via AWS's own product page, not guessed. Smallest
capable size, `g7e.12xlarge` (2 GPU, 48 vCPU, 400 Gbps EFA,
$8.29/hr), is offered in the same three regions
(`us-east-1`/`us-east-2`/`us-west-2`) the `p5.4xlarge` chase already
had infra in. Needed a G/VT vCPU quota bump first (48 > the
then-current 32 cap) -- the pending request from the earlier
`p5.4xlarge` capacity investigation (case `178718106100494`, desired
64) turned out to already cover this once approved (closed
2026-08-21, quota now 64).

**Real capacity found on the first launch attempt** (`us-east-1b`,
after `us-east-1a` rejected the instance type outright as
region/AZ-unsupported, not a capacity error -- a different, cheaper
failure mode than anything hit chasing `p5.4xlarge`). Real hardware
confirmed: 2x RTX PRO 6000 Blackwell (97.9 GiB each), driver
595.91.07, CUDA 13.2, Ubuntu 22.04.5 with **kernel 6.8** out of the
box -- exactly meeting FSx Lustre's stated EFA/GDS kernel floor
(22.04+/6.8+), unlike the `g6.8xlarge` test which needed an AMI swap
to dodge a kernel-compat gap. Lustre client, EFA installer (clean
`fi_pingpong: SUCCESS!`), and FSx creation (`fs-06467cd654a539d1c`,
same `PERSISTENT_2`/`EfaEnabled=true`/4800 GiB config as before) all
went smoothly -- no repeat of any of the 5 infra blockers from the
`g6.8xlarge` test.

**Result: still compat mode, but a genuinely different, better-
characterized cause than the L4 test's kernel `REG_MR` failure.**
No kernel-level RDMA registration error this time -- instead,
`cufile.log` names the exact gap: `nvidia_peermem.ko is not loaded.
Disabling UserSpace RDMA access.` cuFile's RDMA library
(`libcufile_rdma.so`) hard-requires the legacy `nvidia_peermem`
kernel module, which registers GPU memory via Mellanox OFED's
`ib_peer_mem` symbols -- AWS's EFA driver stack ships its own
`efa_nv_peermem` module instead and never exports those symbols, so
`modprobe nvidia_peermem` fails with `EINVAL` even after unloading
`efa_nv_peermem` to free the registration slot (confirmed against
NVIDIA's own developer forums: this is a known, general
Mellanox-OFED-symbol-missing failure mode, not something specific to
this instance). Real GDS I/O against actual FSx-mounted SF1000
`lineitem` data ran successfully via `gdsio`, just through the
host-memory bounce buffer (~27-840 MB/s depending on transfer size,
no crash) rather than true zero-copy RDMA.

**Confirmed this isn't a stale-package issue**: upgraded GDS in place
from the AMI's bundled 13.2 (`gdscheck` release `1.13.1.3`) to the
latest available, 13.3.1 (`gdscheck` release `1.18.1.6`) -- identical
`nvidia_peermem.ko is not loaded` result, byte-for-byte, on the newer
version too. NVIDIA's own GDS troubleshooting guide only documents
`nvidia_peermem` as required for WekaFS/IBM SpectrumScale, not
Lustre, but empirically `libcufile_rdma.so` requires it universally
regardless of backing filesystem.

**Conclusion**: this is a real, version-independent architectural gap
between AWS's EFA driver stack and cuFile's current RDMA
implementation on general-purpose DLAMIs -- consistent with why
NVIDIA/AWS's validated FSx+EFA+GDS list names only P5/Trn1/Hpc7a
specifically (those combinations likely carry AWS-internal driver
support this AMI doesn't have). Not something a GDS/cuFile version
bump, a `cufile.json` tweak, or more debugging on this AMI is likely
to fix -- would need either an AWS-provided GDS-validated AMI for
P5/Trn1/Hpc7a (not tried -- the `g6.8xlarge`/`p5.4xlarge` attempts
both used a general Ubuntu 22.04 DLAMI) or direct AWS/NVIDIA support
engagement to go further. Instance and FSx torn down after this
investigation.

## GDS take 3: found AWS's real, authoritative supported-instance list -- neither `g7e.12xlarge` nor `p5.4xlarge` were ever eligible (2026-08-21)

Went looking for why `g7e.12xlarge` still hit `nvidia_peermem.ko is not
loaded` despite a clean setup, on the theory that the manual FSx-client
configuration (raw `mount -t lustre ...@tcp:/...`) skipped a required
step. It had: AWS's official client setup is a two-part process
(`install-fsx-lustre-client.sh`, then
`configure-efa-fsx-lustre-client.sh --optimized-for-gds`) documented at
[Configuring EFA clients](https://docs.aws.amazon.com/fsx/latest/LustreGuide/configure-efa-clients.html)
-- the second script's job is specifically to add the EFA interface to
Lustre's LNet routing layer (`lnetctl net add --net efa --if <device>
--peer-credits 32`). The manual mount used in both this session's GDS
tests never did this -- Lustre traffic was routed over plain TCP the
whole time, not EFA, on both the `g6.8xlarge` and first `g7e.12xlarge`
attempts. (The doc also confirms Deep Learning AMIs -- what both tests
used -- ship the Lustre client, EFA driver, and GDS driver
pre-installed; verified true on the re-tested `g7e.12xlarge` instance,
packages held at image-build time, matching kernel.)

Re-ran the whole `g7e.12xlarge` + FSx setup from scratch (new instance
`i-0f391da068ea50a5d`, new filesystem, `us-east-1d` after `us-east-1b`
came back capacity-constrained this time) specifically to run the
official `configure-efa-fsx-lustre-client.sh --optimized-for-gds`
script this time instead of a manual mount.

**Definitive result: AWS's own script hard-rejects `g7e.12xlarge`
before doing anything else.** `RuntimeError: Error: Instance type
g7e.12xlarge does not support Lustre GDS.` Not a driver gap, not a
config issue -- reading the script's source
(`bin/configure-efa-fsx-lustre-client.py`) shows the real,
authoritative supported-instance list, hardcoded, independent of
anything in FSx/EC2 marketing pages:

```python
GDS_SUPPORTED_INSTANNCES = [
    "p5.48xlarge",
    "p5e.48xlarge",
    "p5en.48xlarge",
    "p6-b200.48xlarge",
]
```

This retroactively resolves several open threads from this
investigation at once:

- **`g7e.12xlarge` is genuinely, officially unsupported for GDS**,
  contradicting the G7e product page's "multi-GPU G7e instances
  support NVIDIA GPUDirect Storage with FSx for Lustre" claim. The
  `nvidia_peermem.ko` gap found in the first `g7e.12xlarge` attempt was
  real but moot -- this list rules it out before that would ever
  matter.
- **`p5.4xlarge` -- this whole session's original capacity-chase
  target -- was never on the real list either.** Only the full 8-GPU
  `p5.48xlarge` is validated, not the smaller `.4xlarge`/`.8xlarge`
  sizes. Every AZ/region/spot retry loop earlier in this investigation
  was chasing an instance size that would have failed this same
  official check even if AWS capacity had appeared. The
  `REG_MR err -22` kernel failure found on `g6.8xlarge`/L4 was real and
  correctly diagnosed, but the broader lesson -- check the *official*
  `GDS_SUPPORTED_INSTANNCES` list (or run the actual configure script
  in `--configure-once` mode as a cheap eligibility check) before
  provisioning any FSx/EFA infra for a candidate instance type -- would
  have short-circuited both the `g6.8xlarge` and every `p5.4xlarge`
  attempt immediately, at zero cost.
- `p6-b300.48xlarge` is notably **not** on the list (only
  `p6-b200.48xlarge` is), despite both carrying the Blackwell
  Decompression Engine -- a separate, real constraint on that
  unrelated investigation thread if it's ever revisited.
- Trn1/Hpc7a (named in AWS's EFA-throughput announcement blog) don't
  appear here either, consistent with them being non-GPU instances --
  GDS doesn't apply to them at all, EFA's general throughput benefit is
  a separate claim from GDS eligibility.

**Bottom line for any future GDS attempt on this project**: the only
real path forward is `p5.48xlarge` (8x H100, full-size) or
`p6-b200.48xlarge` (8x B200) -- both fixed at 8 GPUs, both far more
expensive than anything tried this session ($6.88/hr x-many for
p5.48xlarge scaling, ~$113.93/hr territory for p6-b200 per the earlier
Decompression Engine pricing check) and both would still need to clear
real capacity *and* the P-family on-demand quota (currently 0, see the
earlier quota-gap note) before even reaching the point of testing
actual GDS behavior. Given the cost/complexity, this line of
investigation is being closed out here rather than continued further
without an explicit decision to spend at that level. Instance and FSx
torn down after this test.

## Multi-GPU Tier 1 implemented: one `RmmEnvironment` per visible device (2026-08-24)

Both `p5.48xlarge` and `p6-b200.48xlarge` above are fixed 8-GPU boxes,
which is what prompted `docs/MULTI_GPU_SCALING.md`'s three-tier plan --
this closes out Tier 1 ("concurrent queries, one GPU each, single
node"), the only tier recommended without a separate scope decision
first.

**What changed**, all four of that doc's "concrete pieces":

1. `GpuExecutionCoordinator` (`gpu_execution_coordinator_gpu.cpp`) no
   longer holds one process-wide `RmmEnvironment` pinned to
   `config.engine.device_id`. Its constructor now calls
   `cudaGetDeviceCount()` and builds one `RmmEnvironment` per visible
   device, each from its own `EngineConfig` copy (`device_id`
   overridden to that device's index) constructed while a
   `CudaDeviceGuard` for that index is active --
   `set_current_device_resource()` really is one slot per device, so
   this is mechanical, exactly as the planning doc expected.
2. `execute()` round-robins across devices via an ever-growing atomic
   counter modulo device count, rather than always targeting device 0.
   Each device gets its own `std::counting_semaphore<>` sized to
   `max_concurrent_gpu_queries` -- that field is now a **per-device**
   cap, not process-wide (see its own updated doc comment in
   `config.hpp`): an N-GPU node gets up to N times the total
   concurrent-query throughput a single-GPU node does, at the same
   per-device concurrency risk profile opt #2 already validated. (This
   field itself later moved from `EngineSection` to
   `ServerConfig::max_concurrent_gpu_queries` -- see the "Follow-up"
   section below.)
3. `ExecutionContext::cuda_device_id` is now threaded through for
   real. `RmmEnvironment` gained a `device_id()` accessor (the value
   it was actually constructed with); `query_engine_execute_gpu.cpp`'s
   `execute(physical, rmm_environment)` reads the target device from
   `rmm_environment.device_id()` instead of `config_.engine.device_id`
   for both its `CudaDeviceGuard` and the `ExecutionContext` it builds.
   This turned out to need no operator-level changes at all: a grep
   audit found no operator in `execution_gpu/` calls `cudaSetDevice()`
   or reads an implicit current-device resource -- every allocation
   already goes through `ExecutionContext::memory_resource` explicitly,
   and the query's one `CudaStream` is created immediately after the
   (now-correct) device guard, so it's already bound to the right
   device. The planning doc flagged this as "most likely to surface
   latent assume-device-0 bugs" -- it didn't, because opt #2's own
   `mr`/`stream` explicitness audit had already closed that gap.
4. Per-device metrics needed no new work: `TrackingMemoryResource`
   (`gpu_memory_tracking_resource.hpp`) already took a `device_id` and
   fed it to `GpuMemoryMetricsRegistry`, which was already keyed by
   device -- once each `RmmEnvironment` is actually constructed with
   its own real `device_id`, the existing `kernellake.gpu.memory.*`
   OTel metrics report correctly per device with no changes needed.

**What didn't change**: no new operator types, no cross-GPU data
movement, `config.engine.device_id`'s meaning for the CLI's one-shot
`QueryEngine::execute(sql)` overload (still the single device that
call uses, unaffected -- it never goes through
`GpuExecutionCoordinator`). A single query still runs entirely on
whichever one device the coordinator dispatched it to; spanning one
query across multiple GPUs is Tier 2, not attempted here.

**Verified**: built and run against this project's real dev GPU
(`gpu-dev-wsl` preset) -- one visible device, so this can't confirm
two *different* physical GPUs each ran a query for real, but it does
confirm the round-robin/semaphore-per-device machinery works
end-to-end and degrades to exactly the old single-device behavior when
`cudaGetDeviceCount() == 1`. Added
`GpuExecutionCoordinatorConcurrencyTest.RoundRobinDeviceSelectionStaysCorrectAcrossManyCalls`
(`tests/unit/gpu_execution_coordinator_test.cpp`) to exercise the
round-robin index arithmetic well past a full wrap of the counter, and
`RmmEnvironment.DeviceIdAccessorReflectsConstructionArgument`
(`tests/gpu/rmm_environment_test.cpp`) for the new accessor. Real
8-GPU verification (confirming queries actually land on distinct
physical devices, and a real per-device-throughput measurement) still
needs actual `p5.48xlarge`/`p6-b200.48xlarge` hardware -- not done this
session, no such instance was provisioned.

## Follow-up: `EngineConfig`/`CliConfig`/`ServerConfig` split, plus explicit GPU selection (2026-08-24)

Two real follow-ups landed the same day as the section above, both
prompted by a closer look at how little of `config.engine.device_id`
actually applied to the server post-Tier-1:

1. **The config type itself split.** `EngineConfig` used to be one
   struct covering everything, including `device_id` (CLI-only in
   practice, silently ignored by the server) and `max_concurrent_gpu_queries`/
   `ServerSection` (server-only, unused by the CLI) all mixed together.
   It's now the *shared* sections only (storage, memory, logging,
   profiling, observability, and an `engine` section without
   `device_id`); `CliConfig` (`engine_config` + `device_id` +
   `benchmark`) and `ServerConfig` (`engine_config` + `server` +
   `max_concurrent_gpu_queries` + `gpu_device_ids`, see point 2 below)
   are the two binaries' own top-level types. `RmmEnvironment` and
   `QueryEngine` both gained an explicit `device_id` constructor
   parameter (defaulting to 0) instead of reading it from config --
   this also simplified `GpuExecutionCoordinator`'s own constructor,
   which no longer needs to build a per-device `EngineConfig` copy with
   `device_id` overridden (item 1 in the section above); it now passes
   the device index straight through. `parse_config()`/`default_config()`
   split into three matching pairs (`parse_cli_config()`/
   `default_cli_config()`, `parse_server_config()`/
   `default_server_config()`, plus the shared `parse_config()`/
   `default_config()` both call internally) so a single YAML file can
   still configure both binaries -- each parser just reads its own
   top-level keys and ignores the other's.
2. **`ServerConfig::gpu_device_ids` added**: real deployments can't
   always assume the server should use *every* visible GPU (a shared
   box may need some left for other workloads) -- empty (default) means
   every device `cudaGetDeviceCount()` reports, Tier 1's original
   behavior; a non-empty list pins the server to exactly those ordinals
   instead. `GpuExecutionCoordinator` validates each configured ordinal
   against the real device count at construction time. Considered GPU
   UUIDs instead of plain CUDA ordinals and deliberately didn't: every
   CUDA call this project makes is ordinal-only regardless, so a UUID
   layer would still need to resolve back to an ordinal before use, and
   this project's actual target deployments (single-tenant bare-metal/VM
   boxes) don't need that extra machinery. Documented the one real
   caveat instead: CUDA's default device enumeration order
   (`CUDA_DEVICE_ORDER=FASTEST_FIRST`) isn't guaranteed to match
   `nvidia-smi`'s (PCI bus ID order), so an operator picking ordinals by
   cross-referencing `nvidia-smi -L` should set
   `CUDA_DEVICE_ORDER=PCI_BUS_ID` in the server's environment.

Verified end-to-end on this project's real dev GPU: `kernellake query
--backend gpu`/`--backend cpu` both still work via `CliConfig`;
`kernellake-server` starts correctly with an explicit
`gpu_device_ids: [0]`; and a deliberately out-of-range
`gpu_device_ids: [5]` on this 1-GPU box fails fast with a clear
`ConfigurationError` ("... reports only 1 visible device(s) ...")
instead of a confusing CUDA-level failure later. Full unit/GPU test
suites (`kernellake_unit_tests`, `kernellake_gpu_tests`) pass.
