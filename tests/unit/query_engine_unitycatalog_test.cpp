#include <arrow/api.h>
#include <arrow/io/file.h>
#include <fmt/format.h>
#include <gtest/gtest.h>
#include <parquet/arrow/writer.h>

#include <filesystem>
#include <string>
#include <vector>

#include "kernellake/api/query_engine.hpp"
#include "kernellake/common/errors.hpp"
#include "kernellake/planner/logical_plan.hpp"
#include "loopback_http_server.hpp"

// This is the capstone test for the read_unity_catalog(...) SQL surface: a
// real QueryEngine, constructed the same way any caller (CLI, Flight SQL
// server) would, planning, physically resolving, and executing a genuine
// `SELECT ... FROM read_unity_catalog('instance.catalog.schema.table')`
// query end to end -- SQL text -> parser preprocessing -> binder -> logical
// plan -> physical plan -> real Acero CPU execution -- against a fake
// Unity Catalog REST API and real Parquet data fixtures.
//
// Covers only the PARQUET dispatch target (storage_location is a local
// directory, not "s3://...", so UnityCatalogSourceResolver never fetches
// temporary credentials -- see that class's own comment): this is the
// shape every other dispatch target (DELTA, ICEBERG) shares once
// UnityCatalogClient::get_table() has returned, and both of those targets'
// own resolution logic (resolve_delta_table(), resolve_iceberg_table()) is
// already covered end to end by query_engine_delta_test.cpp/
// query_engine_iceberg_test.cpp -- this file exists purely to prove the
// Unity Catalog auth/lookup/dispatch wiring itself, not to re-test paths
// already proven elsewhere. A real S3-backed temporary-credential round
// trip is not covered here (no live AWS or MinIO in this test environment
// -- see docs/ROADMAP.md).
namespace kernellake {
namespace {

namespace fs = std::filesystem;

class QueryEngineUnityCatalogTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_query_engine_unitycatalog_test_" +
                    std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_" +
                    ::testing::UnitTest::GetInstance()->current_test_info()->name());
    fs::create_directories(dir_);
  }

  void TearDown() override { fs::remove_all(dir_); }

  void write_data_file(const std::string& file_name, int64_t id_start, int64_t count) {
    arrow::Int64Builder id_builder;
    arrow::DoubleBuilder amount_builder;
    for (int64_t i = 0; i < count; ++i) {
      ASSERT_TRUE(id_builder.Append(id_start + i).ok());
      ASSERT_TRUE(amount_builder.Append(static_cast<double>(id_start + i)).ok());
    }
    std::shared_ptr<arrow::Array> id_array;
    std::shared_ptr<arrow::Array> amount_array;
    ASSERT_TRUE(id_builder.Finish(&id_array).ok());
    ASSERT_TRUE(amount_builder.Finish(&amount_array).ok());
    const auto schema = arrow::schema(
        {arrow::field("id", arrow::int64(), false), arrow::field("amount", arrow::float64(), true)});
    const auto table = arrow::Table::Make(schema, {id_array, amount_array});
    auto sink_result = arrow::io::FileOutputStream::Open((dir_ / file_name).string());
    ASSERT_TRUE(sink_result.ok()) << sink_result.status().ToString();
    const arrow::Status write_status =
        parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *sink_result, /*chunk_size=*/count);
    ASSERT_TRUE(write_status.ok()) << write_status.ToString();
  }

  std::string table_info_json(const std::string& data_source_format) const {
    return fmt::format(R"json({{
      "table_id": "table-uuid-1",
      "table_type": "EXTERNAL",
      "data_source_format": "{}",
      "storage_location": "{}",
      "columns": [
        {{"name": "id", "type_name": "LONG", "nullable": false, "position": 0}},
        {{"name": "amount", "type_name": "DOUBLE", "nullable": true, "position": 1}}
      ]
    }})json",
                      data_source_format, dir_.string());
  }

  EngineConfig config_with_instance(const std::string& uc_url) {
    EngineConfig config = default_config();
    UnityCatalogInstanceSection instance;
    instance.uc_url = uc_url;
    config.unity_catalog.instances["prod"] = instance;
    return config;
  }

  fs::path dir_;
};

TEST_F(QueryEngineUnityCatalogTest, ExplainLogicalBindsAgainstUnityCatalogSchema) {
  write_data_file("data-0.parquet", 0, 5);
  LoopbackHttpServer server;
  server.respond({http_ok_json(table_info_json("PARQUET"))});  // explain_logical() makes one request.

  QueryEngine engine(config_with_instance(server.base_url()));
  const LogicalPlanPtr plan = engine.explain_logical(
      "SELECT id, amount FROM read_unity_catalog('prod.main.db.orders') WHERE id < 100");
  server.join();

  ASSERT_EQ(plan->output_schema().field_count(), 2u);
  EXPECT_EQ(plan->output_schema().field(0).name, "id");
  EXPECT_EQ(plan->output_schema().field(1).name, "amount");
}

TEST_F(QueryEngineUnityCatalogTest, ExplainProducesFullPhysicalPlanOverRealDataFiles) {
  write_data_file("data-0.parquet", 0, 5);
  write_data_file("data-1.parquet", 5, 5);
  const std::string response = http_ok_json(table_info_json("PARQUET"));
  LoopbackHttpServer server;
  // explain() re-resolves independently at physical-planning time (see
  // physical_planner.cpp's own comment) -- two requests total.
  server.respond({response, response});

  QueryEngine engine(config_with_instance(server.base_url()));
  const PhysicalPlanPtr plan = engine.explain("SELECT id FROM read_unity_catalog('prod.main.db.orders')");
  server.join();

  const std::string text = explain_text(*plan);
  EXPECT_NE(text.find("ArrowResult"), std::string::npos);
  EXPECT_NE(text.find("ParquetScan"), std::string::npos);
}

TEST_F(QueryEngineUnityCatalogTest, ExecuteRunsQueryEndToEndOverRealDataFiles) {
  write_data_file("data-0.parquet", 0, 5);
  write_data_file("data-1.parquet", 5, 3);
  const std::string response = http_ok_json(table_info_json("PARQUET"));
  LoopbackHttpServer server;
  server.respond({response, response});

  EngineConfig config = config_with_instance(server.base_url());
  config.engine.backend = "cpu";
  QueryEngine engine(config);

  const QueryResult result =
      engine.execute("SELECT SUM(amount) AS total FROM read_unity_catalog('prod.main.db.orders')");
  server.join();

  ASSERT_EQ(result.rows_returned, 1);
  const auto total_column =
      std::static_pointer_cast<arrow::DoubleArray>(result.batches.front()->GetColumnByName("total"));
  ASSERT_NE(total_column, nullptr);
  // ids 0..4 and 5..7: sum(0..7) = 28.
  EXPECT_DOUBLE_EQ(total_column->Value(0), 28.0);
}

TEST_F(QueryEngineUnityCatalogTest, ExplainLogicalThrowsOnUnknownColumn) {
  write_data_file("data-0.parquet", 0, 5);
  LoopbackHttpServer server;
  server.respond({http_ok_json(table_info_json("PARQUET"))});

  QueryEngine engine(config_with_instance(server.base_url()));
  EXPECT_THROW(
      (void)(engine.explain_logical("SELECT nonexistent FROM read_unity_catalog('prod.main.db.orders')")),
      BindingError);
  server.join();
}

TEST_F(QueryEngineUnityCatalogTest, ExplainLogicalThrowsOnUnknownInstanceName) {
  QueryEngine engine(default_config());  // no "prod" instance configured
  EXPECT_THROW(
      (void)(engine.explain_logical("SELECT id FROM read_unity_catalog('prod.main.db.orders')")),
      ConfigurationError);
}

TEST_F(QueryEngineUnityCatalogTest, ExplainLogicalThrowsWhenDeltaGrpcEndpointIsNotConfigured) {
  LoopbackHttpServer server;
  server.respond({http_ok_json(table_info_json("DELTA"))});

  QueryEngine engine(config_with_instance(server.base_url()));  // no delta.grpc_endpoint configured
  EXPECT_THROW((void)(engine.explain_logical("SELECT id FROM read_unity_catalog('prod.main.db.orders')")),
               ConfigurationError);
  server.join();
}

TEST_F(QueryEngineUnityCatalogTest, ExplainLogicalThrowsOnUnsupportedDataSourceFormat) {
  LoopbackHttpServer server;
  server.respond({http_ok_json(table_info_json("CSV"))});

  QueryEngine engine(config_with_instance(server.base_url()));
  EXPECT_THROW((void)(engine.explain_logical("SELECT id FROM read_unity_catalog('prod.main.db.orders')")),
               StorageError);
  server.join();
}

}  // namespace
}  // namespace kernellake
