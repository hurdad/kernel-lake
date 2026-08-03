// Deterministic, no-network coverage for kernellake::observability's real
// (KERNELLAKE_ENABLE_OTEL=ON) implementation: wires in-memory span/metric
// exporters and a small custom in-memory log exporter (see
// query_tracing_test_support.hpp) in place of the real OTLP/gRPC ones, so
// this proves QuerySpan::finish/finish_error and the spdlog bridge actually
// populate real OTel data, not just that the code compiles.
#include <gtest/gtest.h>

#include <opentelemetry/sdk/metrics/data/point_data.h>
#include <spdlog/spdlog.h>

#include "kernellake/api/query_engine.hpp"
#include "kernellake/common/errors.hpp"
#include "kernellake/observability/query_tracing.hpp"
#include "kernellake/observability/query_tracing_test_support.hpp"

namespace kernellake {
namespace {

class QueryTracingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    observability::init_for_testing("kernellake-test", spans_, metrics_, log_exporter_);
  }

  void TearDown() override { observability::shutdown(); }

  std::shared_ptr<opentelemetry::exporter::memory::InMemorySpanData> spans_;
  std::shared_ptr<opentelemetry::exporter::memory::SimpleAggregateInMemoryMetricData> metrics_;
  observability::TestLogRecordExporter* log_exporter_ = nullptr;
};

TEST_F(QueryTracingTest, FinishRecordsSpanAndHistogram) {
  QueryResult result;
  result.rows_returned = 42;
  result.elapsed_wall_seconds = 0.125;

  {
    observability::QuerySpan span = observability::start_query_span("kernellake.query");
    span.finish(result, "SELECT 1", "cpu");
  }
  observability::shutdown();  // ForceFlush()es synchronously.

  const auto finished = spans_->GetSpans();
  ASSERT_EQ(finished.size(), 1u);
  EXPECT_EQ(finished[0]->GetName(), "kernellake.query");
  EXPECT_EQ(finished[0]->GetStatus(), opentelemetry::trace::StatusCode::kOk);

  const auto& points = metrics_->Get("kernellake-test", "kernellake.query.duration_seconds");
  ASSERT_FALSE(points.empty());
  const auto& point_type = points.begin()->second;
  const auto& histogram =
      opentelemetry::nostd::get<opentelemetry::sdk::metrics::HistogramPointData>(point_type);
  EXPECT_EQ(histogram.count_, 1u);
  EXPECT_DOUBLE_EQ(opentelemetry::nostd::get<double>(histogram.sum_), 0.125);
}

TEST_F(QueryTracingTest, FinishErrorSetsErrorStatus) {
  {
    observability::QuerySpan span = observability::start_query_span("kernellake.query");
    span.finish_error(ExecutionError("boom"), "SELECT 1", "cpu");
  }
  observability::shutdown();

  const auto finished = spans_->GetSpans();
  ASSERT_EQ(finished.size(), 1u);
  EXPECT_EQ(finished[0]->GetStatus(), opentelemetry::trace::StatusCode::kError);

  // No histogram observation on the error path.
  const auto& points = metrics_->Get("kernellake-test", "kernellake.query.duration_seconds");
  EXPECT_TRUE(points.empty());
}

TEST_F(QueryTracingTest, SpdlogCallsAreBridgedToOtelLogs) {
  spdlog::info("hello from kernellake");
  observability::shutdown();

  ASSERT_FALSE(log_exporter_->records.empty());
}

}  // namespace
}  // namespace kernellake
