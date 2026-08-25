// Coverage for GpuExecutionCoordinator's KERNELLAKE_WITH_CUDA=OFF variant
// (gpu_execution_coordinator_stub.cpp) -- never constructed by any test
// before this file, in either build mode. This is a Flight SQL server type
// (server.engine.backend == "gpu" wiring, see the class's own doc comment
// in gpu_execution_coordinator.hpp), so like flight_sql_server_test.cpp/
// auth_middleware_test.cpp it only builds/links under KERNELLAKE_BUILD_SERVER
// (see tests/unit/CMakeLists.txt) -- e.g. the `server-dev` preset, not
// `cpu-dev` alone.
//
// The KERNELLAKE_WITH_CUDA=ON variant (gpu_execution_coordinator_gpu.cpp)
// needs a real GPU/CUDA toolkit -- see the second half of this file (guarded
// by `#else` below) for real coverage of it, added alongside the semaphore-
// based bounded-concurrency rewrite (opt #2, see docs/GPU_OPTIMIZATIONS.md).
// src/server/CMakeLists.txt selects between the two variants via
// KERNELLAKE_WITH_CUDA, so this file is split the same way: the stub half
// asserts on its documented fail-fast behavior, which would be a wrong
// assertion to compile (and, worse, silently never build) against the real
// GPU variant, and vice versa.
#include <gtest/gtest.h>

#include "kernellake/common/config.hpp"
#include "kernellake/common/errors.hpp"
#include "kernellake/server/gpu_execution_coordinator.hpp"

#ifndef KERNELLAKE_WITH_CUDA

namespace kernellake {
namespace {

TEST(GpuExecutionCoordinatorStub, ConstructorThrowsConfigurationErrorFailingFast) {
  EngineConfig config = default_config();
  config.engine.backend = "gpu";
  EXPECT_THROW((void)(GpuExecutionCoordinator(config)), ConfigurationError);
}

TEST(GpuExecutionCoordinatorStub, ConstructorErrorMessageExplainsHowToGetRealGpuSupport) {
  EngineConfig config = default_config();
  config.engine.backend = "gpu";
  try {
    (void)(GpuExecutionCoordinator(config));
    FAIL() << "expected GpuExecutionCoordinator construction to throw in a KERNELLAKE_WITH_CUDA=OFF build";
  } catch (const ConfigurationError& e) {
    const std::string message = e.what();
    EXPECT_NE(message.find("KERNELLAKE_WITH_CUDA"), std::string::npos) << message;
    EXPECT_NE(message.find("cpu"), std::string::npos) << message;
  }
}

}  // namespace
}  // namespace kernellake

#else  // KERNELLAKE_WITH_CUDA

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <filesystem>
#include <thread>
#include <vector>

#include "kernellake/api/query_engine.hpp"

namespace kernellake {
namespace {

namespace fs = std::filesystem;

// Regression coverage for the opt #2 rewrite (see
// docs/GPU_OPTIMIZATIONS.md): GpuExecutionCoordinator used to serialize
// every call through a single std::mutex over one shared RmmEnvironment
// whose statistics_resource_adaptor push/pop *stack* was not safe for
// concurrent callers (confirmed from RMM's own source -- one shared,
// non-thread-local stack; two threads racing push/pop could pop each
// other's frame). It's now a bounded semaphore
// (EngineSection::max_concurrent_gpu_queries) over a shared, thread-safe
// limiter with a fresh, independent QueryMemoryTracker per call
// (RmmEnvironment::make_query_tracker()) -- this test exercises real
// concurrent execution through that path and checks the two things the old
// design could get wrong: query *results* getting mixed up between
// concurrently-running queries, and *peak_gpu_memory_bytes* reporting
// getting corrupted by another query's counters.
class GpuExecutionCoordinatorConcurrencyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_gpu_execution_coordinator_concurrency_test_" +
                    std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::create_directories(dir_);
    path_ = (dir_ / "regions.parquet").string();

    arrow::StringBuilder region_builder;
    arrow::DoubleBuilder amount_builder;
    // Disjoint, easily-distinguishable totals per region -- if two
    // concurrently-running queries' results (or memory accounting) ever
    // got crossed, a region-filtered SUM landing on the *other* region's
    // total is an immediate, unambiguous signal, not something that could
    // pass by coincidence.
    const std::vector<std::string> regions = {"A", "A", "A", "B", "B", "B"};
    const std::vector<double> amounts = {10.0, 20.0, 5.0, 100.0, 7.0, 3.0};
    for (std::size_t i = 0; i < regions.size(); ++i) {
      ASSERT_TRUE(region_builder.Append(regions[i]).ok());
      ASSERT_TRUE(amount_builder.Append(amounts[i]).ok());
    }
    std::shared_ptr<arrow::Array> region_array, amount_array;
    ASSERT_TRUE(region_builder.Finish(&region_array).ok());
    ASSERT_TRUE(amount_builder.Finish(&amount_array).ok());
    const auto schema = arrow::schema(
        {arrow::field("region", arrow::utf8(), false), arrow::field("amount", arrow::float64(), false)});
    const auto table = arrow::Table::Make(schema, {region_array, amount_array});
    auto sink = arrow::io::FileOutputStream::Open(path_).ValueOrDie();
    const arrow::Status status =
        parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, /*chunk_size=*/3);
    ASSERT_TRUE(status.ok()) << status.ToString();
  }

  void TearDown() override { fs::remove_all(dir_); }

  fs::path dir_;
  std::string path_;
};

TEST_F(GpuExecutionCoordinatorConcurrencyTest,
       ConcurrentQueriesDoNotMixResultsOrMemoryAccountingAcrossThreads) {
  EngineConfig config = default_config();
  config.engine.backend = "gpu";
  // Small on purpose: with 4 threads below, this forces real queueing
  // through the semaphore (at least one thread must wait for another to
  // finish), not just "4 threads that all happened to run one at a time
  // anyway" -- exercises the actual bounded-concurrency path, not just
  // that construction doesn't throw.
  config.engine.max_concurrent_gpu_queries = 2;

  GpuExecutionCoordinator coordinator(config);
  QueryEngine engine(config);

  struct ThreadResult {
    QueryResult result;
    double expected_total = 0.0;
  };

  constexpr int kThreads = 4;
  std::vector<std::thread> threads;
  std::vector<ThreadResult> results(kThreads);
  threads.reserve(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    // Even threads query region A (expected 35.0), odd threads query
    // region B (expected 110.0) -- a real mix of concurrently-running,
    // genuinely different queries, not N copies of the same one.
    const bool is_region_a = (i % 2) == 0;
    const std::string region = is_region_a ? "A" : "B";
    results[i].expected_total = is_region_a ? 35.0 : 110.0;
    threads.emplace_back([&, i, region] {
      const std::string sql =
          "SELECT SUM(amount) AS total FROM read_parquet('" + path_ + "') WHERE region = '" + region + "'";
      const PhysicalPlanPtr physical = engine.explain(sql);
      results[i].result = coordinator.execute(engine, physical);
    });
  }
  for (std::thread& t : threads) t.join();

  for (int i = 0; i < kThreads; ++i) {
    const QueryResult& result = results[i].result;
    ASSERT_EQ(result.rows_returned, 1) << "thread " << i;
    ASSERT_EQ(result.batches.size(), 1u) << "thread " << i;
    const auto total_column =
        std::static_pointer_cast<arrow::DoubleArray>(result.batches.front()->GetColumnByName("total"));
    ASSERT_NE(total_column, nullptr) << "thread " << i;
    EXPECT_DOUBLE_EQ(total_column->Value(0), results[i].expected_total)
        << "thread " << i << " got another thread's result -- concurrent execution mixed up query state";

    // Every query here does the same-shaped work (scan + filter + SUM
    // over 3 matching rows), so a real per-query tracker should report a
    // sane, strictly positive peak for every one of them independently --
    // the old shared push/pop stack could instead report zero, a
    // negative/underflowed value, or another thread's peak here under
    // concurrency.
    ASSERT_TRUE(result.peak_gpu_memory_bytes.has_value()) << "thread " << i;
    EXPECT_GT(*result.peak_gpu_memory_bytes, 0) << "thread " << i;
  }
}

// Regression coverage for the Tier 1 multi-device rewrite
// (docs/MULTI_GPU_SCALING.md): GpuExecutionCoordinator now owns one
// RmmEnvironment/semaphore pair per visible CUDA device
// (cudaGetDeviceCount()) and round-robins execute() calls across them via
// an ever-growing atomic counter modulo device count, rather than every
// call always targeting the single process-wide instance this used to be.
// Real hardware for this project's CI/dev boxes has exactly one visible
// GPU, so this can't directly confirm two *different* physical devices
// each ran a query -- what it does confirm, and what a single-device box
// can't get right by accident, is that the round-robin index arithmetic
// stays in bounds and every call still gets a correct, independent result
// well past a full wrap-around of the counter (more calls than any
// plausible device count), on top of the existing concurrency test above
// already covering the semaphore/RmmEnvironment-per-call isolation itself.
TEST_F(GpuExecutionCoordinatorConcurrencyTest, RoundRobinDeviceSelectionStaysCorrectAcrossManyCalls) {
  EngineConfig config = default_config();
  config.engine.backend = "gpu";

  GpuExecutionCoordinator coordinator(config);
  QueryEngine engine(config);

  const std::string sql = "SELECT SUM(amount) AS total FROM read_parquet('" + path_ + "') WHERE region = 'A'";
  const PhysicalPlanPtr physical = engine.explain(sql);

  // Comfortably more than any real single-node GPU count (see
  // kMaxTrackedGpuDevices's own comment, kernellake/memory/gpu_memory_metrics.hpp)
  // so this exercises at least one full wrap of the round-robin counter
  // even on an 8- or 16-GPU box, not just a 1-GPU dev machine.
  constexpr int kCalls = 20;
  for (int i = 0; i < kCalls; ++i) {
    const QueryResult result = coordinator.execute(engine, physical);
    ASSERT_EQ(result.rows_returned, 1) << "call " << i;
    ASSERT_EQ(result.batches.size(), 1u) << "call " << i;
    const auto total_column =
        std::static_pointer_cast<arrow::DoubleArray>(result.batches.front()->GetColumnByName("total"));
    ASSERT_NE(total_column, nullptr) << "call " << i;
    EXPECT_DOUBLE_EQ(total_column->Value(0), 35.0) << "call " << i;
  }
}

}  // namespace
}  // namespace kernellake

#endif  // KERNELLAKE_WITH_CUDA
