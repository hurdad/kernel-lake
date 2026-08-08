#pragma once

#include <cuda_runtime.h>
#include <cudf/io/parquet.hpp>

#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <vector>

#include "kernellake/execution_gpu/cuda_utils.hpp"
#include "kernellake/execution_gpu/operator.hpp"
#include "kernellake/planner/physical_plan.hpp"
#include "kernellake/storage/object_store.hpp"

namespace kernellake {

// Reads the files/row-groups selected by pruning (PhysicalFileFragment,
// from ParquetScanNode) via cudf::io::chunked_parquet_reader, which bounds
// GPU memory per read regardless of dataset size -- "datasets larger than
// GPU memory through iterative processing" from the spec.
//
// `pass_read_limit_bytes` (0 = unlimited) bounds the temporary
// decompression memory used per internal read pass; there is no exact
// "rows per batch" knob on cudf's chunked reader (it is byte-budget based,
// not row-count based), so this is the closest available control and is
// deliberately named for what it actually does rather than implying an
// exact row count.
//
// `store` resolves each fragment's Uri to bytes. Fragments whose scheme is
// "file" use cudf's own local-path source_info constructor directly (no
// extra indirection); any other scheme (s3/gs/gcs/abfs/abfss/az) routes
// through store.open() and an ObjectStoreDatasource wrapper instead -- see
// object_store_datasource.hpp. `store` must outlive this operator.
//
// `partition_columns` (empty for a plain, non-partitioned scan -- the
// common case, unaffected by anything below) names columns whose values
// come from each fragment's file location (Hive-style `key=value`
// directory segments, see kernellake/io/table_resolution.hpp) rather than
// being physically present in the file, using PhysicalFileFragment's own
// parallel `partition_values`. When non-empty, this operator reads one
// fragment at a time via its own chunked_parquet_reader (rather than one
// reader spanning every fragment, this operator's normal fast path) --
// cudf's chunked reader can legitimately batch rows from *multiple* source
// files into a single returned chunk when they fit within
// pass_read_limit_bytes together, and there is no way to recover, after
// the fact, how many of a chunk's rows came from which file, which a
// per-file constant partition value absolutely needs to know. This trades
// away cross-file pass batching specifically for partitioned scans (a
// single large partition's own file can still stream across multiple
// passes/chunks normally) in exchange for provable correctness, rather
// than guessing at a chunk-to-file boundary that isn't actually exposed.
//
// Decode/compute overlap (docs/GPU_OPTIMIZATIONS.md opt #3, prototyped
// 2026-08-08 with a real ~13% wall-time win): the non-partitioned fast path
// decodes on a dedicated background thread + its own CudaStream
// (`decode_stream_`), one chunk read ahead of the consumer, instead of
// blocking the caller of next() inside cudf's own blocking
// read_chunk() call. `cudf::io::chunked_parquet_reader::read_chunk()`
// itself is a synchronous, blocking host call (it must know the resulting
// row count before returning), so overlap can't come from just handing it
// a second stream from the same thread -- it needs a second *thread*
// issuing that blocking call while the consumer thread is still consuming
// the previous chunk. Only the background thread ever touches `reader_`
// once it starts, so no locking is needed around the reader itself, only
// around the single-slot handoff queue between the two threads. GPU memory
// accounting is unaffected by which of the two streams issued an
// allocation: RmmEnvironment installs one resource stack as the *device's*
// default resource (not scoped to any one stream), so both streams'
// allocations are tracked and limited identically.
//
// The partitioned (partition_columns_ non-empty) path is deliberately left
// synchronous, unchanged from before this overlap was added: it is the
// rarer path, and interleaving its own fragment-boundary bookkeeping
// (open_current_fragment() below) with a background thread is a separate,
// higher-risk change not attempted here.
class ParquetScanOperator final : public PhysicalOperator {
 public:
  ParquetScanOperator(OperatorId id, std::vector<PhysicalFileFragment> fragments,
                      std::vector<std::string> columns, std::shared_ptr<const Schema> schema,
                      ObjectStore& store, std::size_t pass_read_limit_bytes = 0,
                      std::vector<PartitionColumn> partition_columns = {});

  // Stops and joins decode_thread_ if it's still running -- the same
  // shutdown logic close() runs, as a safety net for the case close() is
  // never called at all (today's pipeline driver, query_engine_execute_gpu.cpp,
  // skips it on any exception mid-scan -- see stop_decode_thread()'s own
  // comment). Without this, an in-flight decode_thread_ at destruction time
  // is a still-joinable std::thread being destroyed, which calls
  // std::terminate() -- turning any mid-scan error into a process crash
  // instead of a clean exception, a real regression this destructor exists
  // to prevent.
  ~ParquetScanOperator() override;

  void open(ExecutionContext& context) override;
  std::optional<DeviceBatch> next(ExecutionContext& context) override;
  void close(ExecutionContext& context) override;

  [[nodiscard]] std::string_view name() const noexcept override { return "ParquetScan"; }
  [[nodiscard]] OperatorId id() const noexcept override { return id_; }

  // Real cumulative time spent inside reader_->read_chunk() across every
  // chunk this scan has produced so far (both the non-partitioned
  // background-thread path and the partitioned synchronous path -- see
  // class comment -- time their own read_chunk() calls into the same
  // accumulator, so this is accurate regardless of which path a given scan
  // took). Thread-safe to call at any time, but only reflects the *final*
  // total once decode_thread_ (non-partitioned path) has been joined --
  // see close()/the destructor -- callers wanting the final total should
  // read this after close(), not concurrently with an in-progress scan.
  [[nodiscard]] double decode_seconds() const;

  [[nodiscard]] std::optional<double> resource_seconds() const override { return decode_seconds(); }

 private:
  // Opens fragments_[current_fragment_index_] as a fresh single-file
  // chunked_parquet_reader, used only in the partition_columns_-non-empty
  // path (see class comment).
  void open_current_fragment(ExecutionContext& context);

  // Shared shutdown/drain logic between close() and ~ParquetScanOperator()
  // (see that destructor's own comment for why both need it): signals
  // prefetch_loop() to stop, joins decode_thread_ if it was ever started,
  // and releases any chunk it decoded but next() never consumed (e.g. a
  // LIMIT stopped pulling before the scan was exhausted, or an exception
  // elsewhere in the pipeline unwound past this scan while it still had a
  // chunk in flight). Safe to call more than once (close() then the
  // destructor, in the normal case) -- a no-op the second time.
  void stop_decode_thread();

  // One decoded-but-not-yet-consumed chunk, handed from the background
  // decode thread to next()'s caller thread. `ready_event` marks the point
  // on `decode_stream_` after which `table`'s device memory is safe to
  // read -- the consumer's own stream must wait on it (cudaStreamWaitEvent)
  // before any downstream kernel touches `table`, since the two streams
  // otherwise give CUDA no ordering guarantee between them.
  struct PrefetchedChunk {
    std::unique_ptr<cudf::table> table;
    cudaEvent_t ready_event = nullptr;
  };

  // Runs on decode_thread_ for the lifetime of one non-partitioned scan:
  // pulls chunks from reader_ (blocking on cudf's own read_chunk() call,
  // the whole point of doing this off the consumer's thread) and hands
  // each one to next() via queue_, one chunk of read-ahead at a time
  // (queue_ never holds more than one entry -- see queue_mutex_/queue_cv_
  // below). Any exception reader_ throws is captured, not rethrown here
  // (there is nothing meaningful to do with it on this thread); next()
  // rethrows it on the consumer thread instead, matching this operator's
  // pre-overlap behavior of simply propagating whatever read_chunk() threw.
  void prefetch_loop();

  OperatorId id_;
  std::vector<PhysicalFileFragment> fragments_;
  std::vector<std::string> columns_;
  std::shared_ptr<const Schema> schema_;
  ObjectStore& store_;
  std::size_t pass_read_limit_bytes_;
  std::vector<PartitionColumn> partition_columns_;
  std::unique_ptr<cudf::io::chunked_parquet_reader> reader_;
  std::size_t current_fragment_index_ = 0;

  // Decode/compute overlap state (non-partitioned fast path only -- see
  // class comment). decode_stream_ is only constructed when that path is
  // actually taken, so a partitioned scan or an all-pruned-away scan
  // (fragments_.empty()) never pays for a CUDA stream it won't use.
  std::optional<CudaStream> decode_stream_;
  std::thread decode_thread_;
  mutable std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::queue<PrefetchedChunk> queue_;
  bool producer_done_ = false;
  bool stop_requested_ = false;
  std::exception_ptr producer_exception_;
  // Guarded by queue_mutex_, same as everything else the background thread
  // and next()/decode_seconds() share -- see decode_seconds()'s own
  // comment. Also written (without contention, decode_thread_ never runs
  // concurrently with this path) by the partitioned/synchronous next() path.
  double decode_seconds_ = 0.0;
};

}  // namespace kernellake
