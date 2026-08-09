#include "kernellake/execution_gpu/parquet_scan_operator.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/table/table.hpp>
#include <fmt/format.h>

#include <algorithm>
#include <chrono>

#include "kernellake/common/errors.hpp"
#include "kernellake/execution_gpu/cudf_adapter.hpp"
#include "kernellake/execution_gpu/object_store_datasource.hpp"
#include "kernellake/expression/expression.hpp"

namespace kernellake {

ParquetScanOperator::ParquetScanOperator(OperatorId id, std::vector<PhysicalFileFragment> fragments,
                                         std::vector<std::string> columns,
                                         std::shared_ptr<const Schema> schema, ObjectStore& store,
                                         std::size_t pass_read_limit_bytes,
                                         std::vector<PartitionColumn> partition_columns)
    : id_(id),
      fragments_(std::move(fragments)),
      columns_(std::move(columns)),
      schema_(std::move(schema)),
      store_(store),
      pass_read_limit_bytes_(pass_read_limit_bytes),
      partition_columns_(std::move(partition_columns)) {}

ParquetScanOperator::~ParquetScanOperator() {
  stop_decode_thread();
}

void ParquetScanOperator::open(ExecutionContext& context) {
  if (fragments_.empty()) return;  // Every file was pruned away entirely; next() reports empty.

  if (!partition_columns_.empty()) {
    // Per-fragment mode (see class comment): reader_ is opened lazily, one
    // fragment at a time, from next()/open_current_fragment() -- nothing to
    // do here beyond leaving current_fragment_index_ at its initial 0.
    return;
  }

  std::vector<std::vector<cudf::size_type>> row_groups;
  row_groups.reserve(fragments_.size());
  for (const PhysicalFileFragment& fragment : fragments_) {
    row_groups.emplace_back(fragment.selected_row_groups.begin(), fragment.selected_row_groups.end());
  }

  // All-local is the common case and keeps cudf's own local-path source_info
  // constructor with zero extra indirection. Any fragment with a non-"file"
  // scheme routes *every* fragment through ObjectStoreDatasource instead --
  // cudf's chunked reader takes one uniform source list, and a mixed local/
  // remote scan is rare enough not to deserve its own fast path.
  const bool all_local =
      std::all_of(fragments_.begin(), fragments_.end(),
                  [](const PhysicalFileFragment& fragment) { return fragment.file.scheme() == "file"; });

  // decode_stream_, not context.stream: reader_ (and therefore
  // prefetch_loop(), once started below) does all of its decode work on
  // this dedicated stream so it can genuinely run concurrently with
  // whatever the consumer is doing on context.stream -- see this class's
  // header comment on decode/compute overlap.
  decode_stream_.emplace();

  try {
    if (all_local) {
      std::vector<std::string> file_paths;
      file_paths.reserve(fragments_.size());
      for (const PhysicalFileFragment& fragment : fragments_) file_paths.push_back(fragment.file.value());

      cudf::io::parquet_reader_options options =
          cudf::io::parquet_reader_options::builder(cudf::io::source_info(file_paths))
              .column_names(columns_)
              .row_groups(row_groups)
              .build();
      reader_ = std::make_unique<cudf::io::chunked_parquet_reader>(
          /*chunk_read_limit=*/0, pass_read_limit_bytes_, options, decode_stream_->get(),
          context.memory_resource);
    } else {
      std::vector<std::unique_ptr<cudf::io::datasource>> sources;
      sources.reserve(fragments_.size());
      for (const PhysicalFileFragment& fragment : fragments_) {
        sources.push_back(std::make_unique<ObjectStoreDatasource>(store_.open(fragment.file)));
      }

      cudf::io::parquet_reader_options options =
          cudf::io::parquet_reader_options::builder().column_names(columns_).row_groups(row_groups).build();
      reader_ = std::make_unique<cudf::io::chunked_parquet_reader>(
          /*chunk_read_limit=*/0, pass_read_limit_bytes_, std::move(sources),
          /*parquet_metadatas=*/std::vector<cudf::io::parquet::FileMetaData>{}, options,
          decode_stream_->get(), context.memory_resource);
    }
  } catch (const std::exception& e) {
    decode_stream_.reset();
    throw StorageError(fmt::format("failed to open Parquet source for scanning: {}", e.what()));
  }

  decode_thread_ = std::thread(&ParquetScanOperator::prefetch_loop, this);
}

void ParquetScanOperator::open_current_fragment(ExecutionContext& context) {
  const PhysicalFileFragment& fragment = fragments_[current_fragment_index_];
  const std::vector<cudf::size_type> row_groups(fragment.selected_row_groups.begin(),
                                                fragment.selected_row_groups.end());

  try {
    if (fragment.file.scheme() == "file") {
      cudf::io::parquet_reader_options options =
          cudf::io::parquet_reader_options::builder(
              cudf::io::source_info(std::vector<std::string>{fragment.file.value()}))
              .column_names(columns_)
              .row_groups(std::vector<std::vector<cudf::size_type>>{row_groups})
              .build();
      reader_ = std::make_unique<cudf::io::chunked_parquet_reader>(
          /*chunk_read_limit=*/0, pass_read_limit_bytes_, options, context.stream, context.memory_resource);
    } else {
      std::vector<std::unique_ptr<cudf::io::datasource>> sources;
      sources.push_back(std::make_unique<ObjectStoreDatasource>(store_.open(fragment.file)));

      cudf::io::parquet_reader_options options =
          cudf::io::parquet_reader_options::builder()
              .column_names(columns_)
              .row_groups(std::vector<std::vector<cudf::size_type>>{row_groups})
              .build();
      reader_ = std::make_unique<cudf::io::chunked_parquet_reader>(
          /*chunk_read_limit=*/0, pass_read_limit_bytes_, std::move(sources),
          /*parquet_metadatas=*/std::vector<cudf::io::parquet::FileMetaData>{}, options, context.stream,
          context.memory_resource);
    }
  } catch (const std::exception& e) {
    throw StorageError(fmt::format("failed to open Parquet source '{}' for partitioned scanning: {}",
                                   fragment.file.value(), e.what()));
  }
}

void ParquetScanOperator::prefetch_loop() {
  try {
    while (true) {
      // Timed from before has_next(), not just around read_chunk(): cudf's
      // chunked_parquet_reader does its real I/O/decode work for the next
      // pass *inside* has_next() (it has to fetch and decompress enough
      // data to know whether another chunk exists), not inside
      // read_chunk() -- confirmed for real by instrumenting both calls
      // separately against a real S3 dataset: read_chunk() alone measured
      // ~0.1s while has_next() measured ~20s for the exact same pass. A
      // timer around read_chunk() alone (this operator's first cut at
      // decode_seconds_) looked plausible but was silently measuring the
      // wrong call.
      const auto decode_start = std::chrono::steady_clock::now();
      const bool has_next = reader_->has_next();
      if (!has_next) {
        const std::lock_guard<std::mutex> lock(queue_mutex_);
        decode_seconds_ +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start).count();
        break;
      }
      cudf::io::table_with_metadata result = reader_->read_chunk();
      const double chunk_decode_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start).count();
      if (result.tbl->num_rows() == 0) {
        // An empty chunk (e.g. an empty source file) is valid but
        // uninteresting to downstream operators; keep pulling until a
        // non-empty chunk or genuine exhaustion, same as the pre-overlap
        // synchronous loop did. Its decode time still counts -- real work
        // happened even though nothing was produced.
        const std::lock_guard<std::mutex> lock(queue_mutex_);
        decode_seconds_ += chunk_decode_seconds;
        continue;
      }

      cudaEvent_t event = nullptr;
      check_cuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                 "cudaEventCreateWithFlags for prefetched Parquet chunk");
      check_cuda(cudaEventRecord(event, decode_stream_->get()),
                 "cudaEventRecord for prefetched Parquet chunk");

      std::unique_lock<std::mutex> lock(queue_mutex_);
      decode_seconds_ += chunk_decode_seconds;
      // queue_ holds at most one chunk (deliberate one-chunk read-ahead,
      // not an unbounded producer racing arbitrarily far ahead of the
      // consumer and holding unbounded extra GPU memory) -- wait for
      // next() to have drained the previous chunk before decoding another.
      queue_cv_.wait(lock, [this] { return queue_.empty() || stop_requested_; });
      if (stop_requested_) {
        lock.unlock();
        check_cuda(cudaEventDestroy(event), "cudaEventDestroy on prefetch shutdown");
        return;
      }
      queue_.push(PrefetchedChunk{std::move(result.tbl), event});
      lock.unlock();
      queue_cv_.notify_all();
    }
  } catch (...) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    producer_exception_ = std::current_exception();
    queue_cv_.notify_all();
    return;
  }
  std::lock_guard<std::mutex> lock(queue_mutex_);
  producer_done_ = true;
  queue_cv_.notify_all();
}

std::optional<DeviceBatch> ParquetScanOperator::next(ExecutionContext& context) {
  if (fragments_.empty()) return std::nullopt;

  if (partition_columns_.empty()) {
    if (!decode_thread_.joinable()) return std::nullopt;  // open() never started a scan (see above)

    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_cv_.wait(lock, [this] { return !queue_.empty() || producer_done_ || producer_exception_; });
    if (producer_exception_) {
      const std::exception_ptr exc = producer_exception_;
      lock.unlock();
      std::rethrow_exception(exc);
    }
    if (queue_.empty()) {
      return std::nullopt;  // producer_done_, nothing left to hand out
    }
    PrefetchedChunk chunk = std::move(queue_.front());
    queue_.pop();
    lock.unlock();
    // Wake the producer now that queue_'s one slot is free again, so it can
    // start decoding the *next* chunk concurrently with this call's caller
    // computing on the chunk being returned below.
    queue_cv_.notify_all();

    // context.stream (the consumer's stream) has no ordering relationship
    // with decode_stream_ (chunk.table was decoded there) -- without this,
    // a kernel a downstream operator launches on context.stream could race
    // the decode that produced chunk.table's device memory. The wait
    // itself is enqueued onto context.stream, not a host-side block, so it
    // doesn't cancel out the overlap this exists to get.
    check_cuda(cudaStreamWaitEvent(context.stream, chunk.ready_event, 0),
               "cudaStreamWaitEvent for prefetched Parquet chunk");
    check_cuda(cudaEventDestroy(chunk.ready_event), "cudaEventDestroy for prefetched Parquet chunk");
    return DeviceBatch(std::move(chunk.table), schema_);
  }

  // Per-fragment mode: pull from the current fragment's own reader,
  // advancing to the next fragment (opening its own fresh reader) whenever
  // the current one is exhausted, until every fragment has been read. Every
  // chunk returned here, by construction, comes entirely from
  // fragments_[current_fragment_index_], so appending that one fragment's
  // partition_values as constant columns is always correct -- see the
  // class's own comment for why a single reader spanning every fragment
  // (this operator's normal fast path) can't offer that same guarantee.
  while (true) {
    if (!reader_) {
      if (current_fragment_index_ >= fragments_.size()) return std::nullopt;
      open_current_fragment(context);
    }
    // Timed from before has_next(), same reasoning as prefetch_loop()'s own
    // comment: has_next() is where cudf's chunked reader does its real
    // per-pass I/O/decode work, not read_chunk().
    const auto decode_start = std::chrono::steady_clock::now();
    if (!reader_->has_next()) {
      {
        const std::lock_guard<std::mutex> lock(queue_mutex_);
        decode_seconds_ +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start).count();
      }
      reader_.reset();
      ++current_fragment_index_;
      continue;
    }

    cudf::io::table_with_metadata result = reader_->read_chunk();
    {
      const std::lock_guard<std::mutex> lock(queue_mutex_);
      decode_seconds_ +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start).count();
    }
    if (result.tbl->num_rows() == 0) {
      continue;  // empty chunk; keep pulling from the same (still-open) fragment reader.
    }

    const cudf::size_type num_rows = result.tbl->num_rows();
    std::vector<std::unique_ptr<cudf::column>> columns = result.tbl->release();
    const PhysicalFileFragment& fragment = fragments_[current_fragment_index_];
    for (std::size_t i = 0; i < partition_columns_.size(); ++i) {
      const LiteralExpression literal(fragment.partition_values[i], partition_columns_[i].type);
      const std::unique_ptr<cudf::scalar> scalar = literal_to_scalar(literal);
      columns.push_back(
          cudf::make_column_from_scalar(*scalar, num_rows, context.stream, context.memory_resource));
    }
    auto table = std::make_unique<cudf::table>(std::move(columns));
    return DeviceBatch(std::move(table), schema_);
  }
}

double ParquetScanOperator::decode_seconds() const {
  const std::lock_guard<std::mutex> lock(queue_mutex_);
  return decode_seconds_;
}

void ParquetScanOperator::close(ExecutionContext&) {
  stop_decode_thread();
}

void ParquetScanOperator::stop_decode_thread() {
  if (decode_thread_.joinable()) {
    {
      // Wakes prefetch_loop() out of either wait it could be blocked in:
      // waiting for queue_ space to push a just-decoded chunk, or (already
      // handled by producer_done_/producer_exception_, not this flag)
      // nothing to change there. A decode already in flight inside
      // reader_->read_chunk() itself is not interrupted -- there is no safe
      // way to cancel mid-decode -- so this join() can block for up to one
      // chunk's decode time, same as the pre-overlap synchronous close()
      // could already block on a still-in-flight cudf call.
      const std::lock_guard<std::mutex> lock(queue_mutex_);
      stop_requested_ = true;
    }
    queue_cv_.notify_all();
    decode_thread_.join();
  }

  // Drain any chunk the producer decoded but next() never consumed (e.g. a
  // LIMIT stopped pulling before the scan was exhausted) -- otherwise its
  // event leaks and its cudf::table leaks the device memory it holds.
  while (!queue_.empty()) {
    PrefetchedChunk chunk = std::move(queue_.front());
    queue_.pop();
    if (chunk.ready_event != nullptr) {
      check_cuda(cudaEventDestroy(chunk.ready_event), "cudaEventDestroy for undelivered prefetched chunk");
    }
  }
  producer_done_ = false;
  stop_requested_ = false;
  producer_exception_ = nullptr;

  reader_.reset();
  decode_stream_.reset();
}

}  // namespace kernellake
