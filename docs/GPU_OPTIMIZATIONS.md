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
  (`RmmEnvironment`, `src/memory/rmm_environment.cpp`).
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

## Follow-up: GPU memory growing across benchmark runs

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
