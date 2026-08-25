// Real-OTel-SDK coverage for InstrumentedOperator's per-operator span tree
// (operator_builder.cpp): proves a real query against the real GPU pipeline
// produces a span *tree* shaped like the physical plan -- not one flat span
// -- and specifically that a two-child operator (HashJoinOperator) parents
// both children under itself as siblings rather than nesting the second
// under the first (the bug ExecutionContext::current_span's scoped
// attach/detach exists to avoid -- see that field's own comment). Also
// proves ParquetScanOperator's real decode time (background-thread
// decode/compute overlap -- see that class) reaches its own span as the
// kernellake.operator.resource_seconds attribute. Mirrors
// gpu_memory_metrics_otel_test.cpp's in-memory-exporter pattern.
#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <filesystem>
#include <unordered_map>
#include <vector>

#include "kernellake/api/query_engine.hpp"
#include "kernellake/observability/query_tracing.hpp"
#include "kernellake/observability/query_tracing_test_support.hpp"

namespace kernellake {
namespace {

namespace fs = std::filesystem;

std::string span_id_to_string(const opentelemetry::trace::SpanId& id) {
  char buffer[2 * opentelemetry::trace::SpanId::kSize];
  id.ToLowerBase16(buffer);
  return std::string(buffer, sizeof(buffer));
}

class QueryPlanTracingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    observability::init_for_testing("kernellake-query-plan-tracing-test", spans_, metrics_, log_exporter_);

    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_query_plan_tracing_test_" +
                    std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::create_directories(dir_);
    orders_path_ = (dir_ / "orders.parquet").string();
    customers_path_ = (dir_ / "customers.parquet").string();

    {
      arrow::Int64Builder order_id_builder;
      arrow::Int64Builder customer_id_builder;
      for (std::int64_t i = 0; i < 5; ++i) {
        ASSERT_TRUE(order_id_builder.Append(i).ok());
        ASSERT_TRUE(customer_id_builder.Append(i % 3).ok());
      }
      std::shared_ptr<arrow::Array> order_id_array, customer_id_array;
      ASSERT_TRUE(order_id_builder.Finish(&order_id_array).ok());
      ASSERT_TRUE(customer_id_builder.Finish(&customer_id_array).ok());
      const auto schema = arrow::schema({arrow::field("order_id", arrow::int64(), false),
                                         arrow::field("customer_id", arrow::int64(), false)});
      const auto table = arrow::Table::Make(schema, {order_id_array, customer_id_array});
      auto sink = arrow::io::FileOutputStream::Open(orders_path_).ValueOrDie();
      ASSERT_TRUE(
          parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, /*chunk_size=*/5).ok());
    }
    {
      arrow::Int64Builder customer_id_builder;
      for (std::int64_t i = 0; i < 3; ++i) ASSERT_TRUE(customer_id_builder.Append(i).ok());
      std::shared_ptr<arrow::Array> customer_id_array;
      ASSERT_TRUE(customer_id_builder.Finish(&customer_id_array).ok());
      const auto schema = arrow::schema({arrow::field("customer_id", arrow::int64(), false)});
      const auto table = arrow::Table::Make(schema, {customer_id_array});
      auto sink = arrow::io::FileOutputStream::Open(customers_path_).ValueOrDie();
      ASSERT_TRUE(
          parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, /*chunk_size=*/3).ok());
    }
  }

  void TearDown() override {
    observability::shutdown();
    fs::remove_all(dir_);
  }

  fs::path dir_;
  std::string orders_path_;
  std::string customers_path_;
  std::shared_ptr<opentelemetry::exporter::memory::InMemorySpanData> spans_;
  std::shared_ptr<opentelemetry::exporter::memory::SimpleAggregateInMemoryMetricData> metrics_;
  observability::TestLogRecordExporter* log_exporter_ = nullptr;
};

TEST_F(QueryPlanTracingTest, JoinProducesRealSpanTreeWithSiblingScansAndDecodeTime) {
  QueryEngine engine{default_config()};
  {
    observability::QuerySpan query_span = observability::start_query_span("kernellake.query");
    const QueryResult result = engine.execute("SELECT COUNT(*) AS n FROM read_parquet('" + orders_path_ +
                                              "') o JOIN read_parquet('" + customers_path_ +
                                              "') c ON o.customer_id = c.customer_id");
    query_span.finish(result, "join query", "gpu");
  }
  observability::shutdown();  // ForceFlush()es synchronously.

  const auto finished = spans_->GetSpans();

  // At minimum: the whole-query span, HashJoin, two ParquetScan (one per
  // side), and ArrowResult -- a real tree, not one flat span.
  ASSERT_GE(finished.size(), 5u);

  const opentelemetry::sdk::trace::SpanData* query_span_data = nullptr;
  const opentelemetry::sdk::trace::SpanData* hash_join_span = nullptr;
  std::vector<const opentelemetry::sdk::trace::SpanData*> scan_spans;

  for (const auto& span : finished) {
    if (span->GetName() == "kernellake.query") query_span_data = span.get();
    if (span->GetName() == "HashJoin") hash_join_span = span.get();
    if (span->GetName() == "ParquetScan") scan_spans.push_back(span.get());
  }

  ASSERT_NE(query_span_data, nullptr);
  ASSERT_NE(hash_join_span, nullptr);
  ASSERT_EQ(scan_spans.size(), 2u);

  // HashJoin nests somewhere under the whole-query span -- not necessarily
  // as its *direct* child (this query's real plan is
  // ArrowResult -> ScalarAggregate -> HashJoin -> [scan, scan], so
  // HashJoin's own immediate parent is ScalarAggregate's span), but
  // walking the parent chain must reach it within the tree's actual depth.
  std::unordered_map<std::string, opentelemetry::trace::SpanId> parent_by_id;
  for (const auto& span : finished) {
    parent_by_id.emplace(span_id_to_string(span->GetSpanId()), span->GetParentSpanId());
  }
  opentelemetry::trace::SpanId current = hash_join_span->GetSpanId();
  bool reached_query_span = false;
  for (int hop = 0; hop < 10 && !reached_query_span; ++hop) {
    const auto it = parent_by_id.find(span_id_to_string(current));
    if (it == parent_by_id.end()) break;
    current = it->second;
    if (current == query_span_data->GetSpanId()) reached_query_span = true;
  }
  EXPECT_TRUE(reached_query_span) << "HashJoin span's parent chain never reached the whole-query span";

  // Each scan's own immediate parent is now a "BatchSizeLimit" span (see
  // operator_builder.cpp: EngineSection::batch_rows wraps every
  // ParquetScanNode's own operator in a BatchSizeLimitOperator), not
  // HashJoin directly -- but those two BatchSizeLimit spans must
  // themselves parent directly to HashJoin as true siblings, not nested
  // under each other. This is the specific bug
  // ExecutionContext::current_span's scoped attach/detach (restoring the
  // previous parent right after each child's own recursive open() call
  // returns, not holding it attached for this operator's whole lifetime)
  // exists to avoid: naively holding a "current span" attached across an
  // operator's entire open()-to-close() lifetime would make the second
  // child (whichever side HashJoinOperator::open() opens second) wrongly
  // nest under the first child's span instead.
  // Three BatchSizeLimit spans exist in this tree in total (one per scan's
  // own EngineSection::batch_rows wrapper, plus one more just below
  // ArrowResult for EngineSection::result_batch_rows) -- only the two
  // wrapping a scan are relevant to this check, found by walking each
  // scan's own immediate parent rather than assuming a fixed total count.
  std::unordered_map<std::string, const opentelemetry::sdk::trace::SpanData*> span_by_id;
  for (const auto& span : finished) {
    span_by_id.emplace(span_id_to_string(span->GetSpanId()), span.get());
  }
  std::vector<const opentelemetry::sdk::trace::SpanData*> scan_batch_limit_spans;
  for (const auto* scan : scan_spans) {
    const auto it = span_by_id.find(span_id_to_string(scan->GetParentSpanId()));
    ASSERT_NE(it, span_by_id.end()) << "ParquetScan span " << span_id_to_string(scan->GetSpanId())
                                    << " has no matching parent span";
    EXPECT_EQ(it->second->GetName(), "BatchSizeLimit")
        << "ParquetScan span " << span_id_to_string(scan->GetSpanId()) << " parented to \""
        << it->second->GetName() << "\", expected \"BatchSizeLimit\"";
    scan_batch_limit_spans.push_back(it->second);
  }
  for (const auto* limit : scan_batch_limit_spans) {
    EXPECT_TRUE(limit->GetParentSpanId() == hash_join_span->GetSpanId())
        << "BatchSizeLimit span " << span_id_to_string(limit->GetSpanId()) << " parented to "
        << span_id_to_string(limit->GetParentSpanId()) << ", expected HashJoin ("
        << span_id_to_string(hash_join_span->GetSpanId()) << ")";
  }
  ASSERT_EQ(scan_batch_limit_spans.size(), 2u);
  EXPECT_FALSE(scan_batch_limit_spans[0]->GetSpanId() == scan_batch_limit_spans[1]->GetSpanId());
  EXPECT_FALSE(scan_spans[0]->GetSpanId() == scan_spans[1]->GetSpanId());

  // Real decode time (not just span nesting) reaches at least one scan's
  // span as a numeric attribute -- see ParquetScanOperator::resource_seconds().
  bool found_resource_seconds = false;
  for (const auto* scan : scan_spans) {
    const auto& attributes = scan->GetAttributes();
    const auto it = attributes.find("kernellake.operator.resource_seconds");
    if (it != attributes.end()) {
      found_resource_seconds = true;
      EXPECT_GE(opentelemetry::nostd::get<double>(it->second), 0.0);
    }
  }
  EXPECT_TRUE(found_resource_seconds) << "no ParquetScan span carried kernellake.operator.resource_seconds";
}

}  // namespace
}  // namespace kernellake
