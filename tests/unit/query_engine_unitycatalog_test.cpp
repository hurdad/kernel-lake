#include <arpa/inet.h>
#include <arrow/api.h>
#include <arrow/filesystem/s3fs.h>
#include <arrow/io/file.h>
#include <fmt/format.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <parquet/arrow/writer.h>
#include <sys/socket.h>
#include <unistd.h>

#include <filesystem>
#include <string>
#include <vector>

#include "kernellake/api/query_engine.hpp"
#include "kernellake/common/errors.hpp"
#include "kernellake/planner/logical_plan.hpp"
#include "loopback_http_server.hpp"

namespace {
// This is the first (and, as of writing, only) test file in this suite to
// construct a real arrow::fs::S3FileSystem (via UnityCatalogSourceResolver's
// vended-credentials S3ObjectStore, against a real local MinIO -- see the
// Minio*-prefixed tests below). Arrow's own EnsureS3Initialized() docs
// require a matching FinalizeS3() "before the end of the program" or the
// AWS SDK's global teardown corrupts the heap at process exit (confirmed
// for real: this exact test binary SIGABRTs after gtest itself reports
// every test passing, until this environment was added) -- s3_object_store.cpp
// itself deliberately never calls FinalizeS3() (matching http_client.cpp's
// identical curl_global_cleanup()-never-called reasoning, correct for a
// short-lived CLI process where the OS reclaims everything at exit anyway),
// so a long-lived test binary that actually exercises real S3 I/O needs to
// do this once, itself, after every test has finished.
class S3FinalizeEnvironment final : public ::testing::Environment {
 public:
  void TearDown() override {
    if (arrow::fs::IsS3Initialized()) {
      (void)arrow::fs::FinalizeS3();
    }
  }
};
const ::testing::Environment* const kS3FinalizeEnvironment =
    ::testing::AddGlobalTestEnvironment(new S3FinalizeEnvironment());
}  // namespace

// This is the capstone test for the read_unity_catalog(...) SQL surface: a
// real QueryEngine, constructed the same way any caller (CLI, Flight SQL
// server) would, planning, physically resolving, and executing a genuine
// `SELECT ... FROM read_unity_catalog('instance.catalog.schema.table')`
// query end to end -- SQL text -> parser preprocessing -> binder -> logical
// plan -> physical plan -> real Acero CPU execution -- against a fake
// Unity Catalog REST API and real Parquet data fixtures.
//
// Covers the PARQUET dispatch target both ways: a local directory
// storage_location (UnityCatalogSourceResolver never fetches temporary
// credentials for one -- see that class's own comment) for the always-run
// tests below, and a real "s3://..." one against a real local MinIO for
// the two Minio*-prefixed tests (skipped if MinIO isn't reachable -- see
// minio_reachable() below). Either way, the DELTA/ICEBERG dispatch
// targets' own resolution logic (resolve_delta_table(),
// resolve_iceberg_table()) is already covered end to end by
// query_engine_delta_test.cpp/query_engine_iceberg_test.cpp -- this file
// exists purely to prove the Unity Catalog auth/lookup/dispatch wiring
// itself, not to re-test paths already proven elsewhere.
//
// Also verified directly against a real unitycatalog/unitycatalog OSS
// server (not captured as an automated test -- that server's own real
// temporary-table-credentials vending is AWS-IAM-role/STS-based, which
// MinIO doesn't support without much deeper setup; see docs/ROADMAP.md).
// That live-server verification found a real architectural gap the two
// Minio*-prefixed tests below pin down precisely: vended credentials are
// correctly used for schema discovery and physical planning (both go
// through TableSourceResolver::resolve(), which UnityCatalogSourceResolver
// builds a temporary, vended-credentialed S3ObjectStore for), but the
// *actual scan execution* reads through QueryEngine's own long-lived,
// statically-configured ObjectStore instead -- the physical plan carries
// file paths, not a reference to which store resolved them. So `explain`/
// `explain_logical` (resolve-only) succeed against a bucket only the
// vended credentials can reach; a real `execute()` fails once it tries to
// actually read the data. See docs/ROADMAP.md's Unity Catalog entry.
namespace kernellake {
namespace {

namespace fs = std::filesystem;

// True if something is listening on 127.0.0.1:9000 -- this project's own
// benchmarks/local/docker-compose.yml MinIO service, if brought up
// (`docker compose up -d minio minio-init` from that directory). The
// MinIO-backed test below is skipped, not failed, when this is false: a
// real network dependency this test suite's other files never require.
bool minio_reachable() {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return false;
  timeval timeout{1, 0};
  ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(9000);
  ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
  const bool ok = ::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
  ::close(fd);
  return ok;
}

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

  // Real fixture data uploaded once via `mc cp` to a real local MinIO --
  // see the Minio*-prefixed tests' own header comments -- 5 rows: id 1-5,
  // amount 10-50.
  static std::string minio_table_info() {
    return http_ok_json(R"json({
      "table_id": "table-uuid-minio",
      "table_type": "EXTERNAL",
      "data_source_format": "PARQUET",
      "storage_location": "s3://kernellake-uc-test/orders",
      "columns": [
        {"name": "id", "type_name": "LONG", "nullable": false, "position": 0},
        {"name": "amount", "type_name": "DOUBLE", "nullable": true, "position": 1}
      ]
    })json");
  }

  // Stands in for Unity Catalog's real temporary-table-credentials
  // response shape (verified against real docs and, separately, this
  // project's own live-server testing) -- MinIO's own static root
  // credentials playing the vended-credential role, since
  // S3ObjectStore's vended-credentials constructor has no way to tell
  // the difference (see its own comment in s3_object_store.hpp).
  static std::string minio_temp_credentials() {
    return http_ok_json(R"json({
      "aws_temp_credentials": {
        "access_key_id": "minioadmin",
        "secret_access_key": "minioadmin",
        "session_token": ""
      }
    })json");
  }

  // MinIO, not real AWS: region is arbitrary (MinIO ignores it, but
  // Arrow's S3 client requires a non-empty value), endpoint_override/
  // scheme point at the local MinIO container -- the same fields
  // benchmarks/local/config/kernellake-server.yaml already documents for
  // this exact reason.
  EngineConfig config_with_minio_instance(const std::string& uc_url) {
    EngineConfig config = config_with_instance(uc_url);
    config.storage.s3.options.region = "us-east-1";
    config.storage.s3.options.endpoint_override = "127.0.0.1:9000";
    config.storage.s3.options.scheme = "http";
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

// The one piece query_engine_unitycatalog_test.cpp's other tests can't
// cover: a real S3-vended-credentials round trip. UnityCatalogClient's own
// parsing of a real "aws_temp_credentials" response is already verified in
// unity_catalog_client_test.cpp; what's new here is that
// UnityCatalogSourceResolver actually applies those credentials to a real
// S3ObjectStore and reads real bytes over the network -- MinIO standing in
// for both Unity Catalog's vended-credential source (its own
// storage_location is real MinIO data uploaded independently of this
// test, not through Unity Catalog at all) and for AWS S3 itself. Requires
// `docker compose up -d minio minio-init` from benchmarks/local/ first,
// and the fixture bucket/object this test reads
// (kernellake-uc-test/orders/data-0.parquet, 5 rows: id 1-5, amount
// 10-50) uploaded once via `mc cp` -- not created by this test, since
// SetUp()/TearDown() only manage the local dir_ fixture every other test
// here uses.
//
// `explain()` only (not `explain_logical()`): it re-resolves independently
// at physical-planning time (see QueryEngine::explain()'s own comment),
// so this exercises UnityCatalogSourceResolver building a *second*,
// independent vended-credentialed S3ObjectStore/real AWS-SDK S3 client
// within the same process -- a real, live-MinIO-backed regression test
// for the fresh-per-call design already documented in that class's own
// comment, not just a single-shot check.
TEST_F(QueryEngineUnityCatalogTest, MinioBackedExplainProducesPhysicalPlanWithVendedCredentials) {
  if (!minio_reachable()) {
    GTEST_SKIP() << "MinIO not reachable at 127.0.0.1:9000 -- run `docker compose up -d minio minio-init` "
                    "from benchmarks/local/ first";
  }

  LoopbackHttpServer server;
  server.respond({minio_table_info(), minio_temp_credentials(), minio_table_info(), minio_temp_credentials()});

  QueryEngine engine(config_with_minio_instance(server.base_url()));
  const PhysicalPlanPtr plan =
      engine.explain("SELECT SUM(amount) AS total FROM read_unity_catalog('prod.main.db.orders')");
  server.join();

  const std::string text = explain_text(*plan);
  EXPECT_NE(text.find("ScalarAggregate"), std::string::npos);
  EXPECT_NE(text.find("ParquetScan"), std::string::npos);
}

// Pins down the real gap this session's live-server verification found
// (see this file's own header comment): a real `execute()` currently
// fails once it reaches actual scan execution, because that step reads
// through QueryEngine's own long-lived, statically-configured
// ObjectStore -- never through the resolver's temporary, vended-
// credentialed one, which only `resolve()` (schema discovery, physical
// planning -- see the test above) ever sees. This is a real limitation of
// this vertical slice, not desired behavior: this test exists so it fails
// loudly (not silently) once someone threads per-query object-store
// credentials through to execution and this stops throwing -- at which
// point this test should be replaced with one asserting a correct
// SUM(amount) = 150.0 result, the same shape as
// ExecuteRunsQueryEndToEndOverRealDataFiles above.
TEST_F(QueryEngineUnityCatalogTest, MinioBackedExecuteFailsBecauseScanExecutionBypassesVendedCredentials) {
  if (!minio_reachable()) {
    GTEST_SKIP() << "MinIO not reachable at 127.0.0.1:9000 -- run `docker compose up -d minio minio-init` "
                    "from benchmarks/local/ first";
  }

  LoopbackHttpServer server;
  server.respond({minio_table_info(), minio_temp_credentials(), minio_table_info(), minio_temp_credentials()});

  EngineConfig config = config_with_minio_instance(server.base_url());
  config.engine.backend = "cpu";
  QueryEngine engine(config);

  EXPECT_THROW(
      (void)(engine.execute("SELECT SUM(amount) AS total FROM read_unity_catalog('prod.main.db.orders')")),
      StorageError);
  server.join();
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
