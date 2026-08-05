// End-to-end test for kernellake-server's Arrow Flight SQL surface
// (KernelLakeFlightSqlServer): starts a real server on an OS-assigned
// ephemeral port, connects a real arrow::flight::sql::FlightSqlClient over
// gRPC, and runs a query through the full RPC round trip -- proving
// GetFlightInfoStatement/DoGetStatement work together over the wire, not
// just that they compile. CPU backend only (no GPU needed); reuses the
// exact fixture data from tests/unit/query_engine_test.cpp's
// QueryEngineTest so the expected grouped sums (region A: 10, region B: 35)
// are already-established values, not new magic numbers.
#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/flight/api.h>
#include <arrow/flight/sql/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <atomic>
#include <filesystem>
#include <map>
#include <thread>
#include <vector>

#include "kernellake/common/config.hpp"
#include "kernellake/server/flight_sql_server.hpp"

namespace kernellake {
namespace {

namespace fs = std::filesystem;
namespace flight = arrow::flight;
namespace flight_sql = arrow::flight::sql;

EngineConfig cpu_backend_server_config(std::uint32_t max_pending_results = 1024) {
  EngineConfig config = default_config();
  config.engine.backend = "cpu";
  config.server.host = "127.0.0.1";
  config.server.port = 0;  // OS-assigned ephemeral port, read back after Init().
  config.server.max_pending_results = max_pending_results;
  return config;
}

class FlightSqlServerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           fs::path("kernellake_flight_sql_server_test_" +
                    std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::create_directories(dir_);
    path_ = (dir_ / "sales.parquet").string();

    arrow::Int64Builder id_builder;
    arrow::DoubleBuilder amount_builder;
    arrow::StringBuilder region_builder;
    for (int64_t i = 0; i < 10; ++i) {
      ASSERT_TRUE(id_builder.Append(i).ok());
      ASSERT_TRUE(amount_builder.Append(static_cast<double>(i)).ok());
      ASSERT_TRUE(region_builder.Append(i < 5 ? "A" : "B").ok());
    }
    std::shared_ptr<arrow::Array> id_array, amount_array, region_array;
    ASSERT_TRUE(id_builder.Finish(&id_array).ok());
    ASSERT_TRUE(amount_builder.Finish(&amount_array).ok());
    ASSERT_TRUE(region_builder.Finish(&region_array).ok());
    const auto schema = arrow::schema({arrow::field("id", arrow::int64(), false),
                                       arrow::field("amount", arrow::float64(), false),
                                       arrow::field("region", arrow::utf8(), false)});
    const auto table = arrow::Table::Make(schema, {id_array, amount_array, region_array});
    auto sink = arrow::io::FileOutputStream::Open(path_).ValueOrDie();
    const arrow::Status write_status =
        parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, /*chunk_size=*/5);
    ASSERT_TRUE(write_status.ok()) << write_status.ToString();

    server_ = std::make_unique<KernelLakeFlightSqlServer>(cpu_backend_server_config());
    auto location = flight::Location::ForGrpcTcp("127.0.0.1", 0);
    ASSERT_TRUE(location.ok()) << location.status().ToString();
    const flight::FlightServerOptions options(*location);
    const arrow::Status init_status = server_->Init(options);
    ASSERT_TRUE(init_status.ok()) << init_status.ToString();

    serve_thread_ = std::thread([this] {
      const arrow::Status status = server_->Serve();
      EXPECT_TRUE(status.ok() || status.IsCancelled()) << status.ToString();
    });

    auto client_location = flight::Location::ForGrpcTcp("127.0.0.1", server_->port());
    ASSERT_TRUE(client_location.ok()) << client_location.status().ToString();
    auto flight_client = flight::FlightClient::Connect(*client_location);
    ASSERT_TRUE(flight_client.ok()) << flight_client.status().ToString();
    client_ = std::make_unique<flight_sql::FlightSqlClient>(std::move(*flight_client));
  }

  void TearDown() override {
    client_.reset();
    if (server_) {
      ASSERT_TRUE(server_->Shutdown().ok());
    }
    if (serve_thread_.joinable()) serve_thread_.join();
    fs::remove_all(dir_);
  }

  fs::path dir_;
  std::string path_;
  std::unique_ptr<KernelLakeFlightSqlServer> server_;
  std::thread serve_thread_;
  std::unique_ptr<flight_sql::FlightSqlClient> client_;
};

TEST_F(FlightSqlServerTest, ExecutesGroupedAggregateOverRealRpcRoundTrip) {
  const flight::FlightCallOptions call_options;
  auto info = client_->Execute(
      call_options, "SELECT region, SUM(amount) AS total FROM read_parquet('" + path_ + "') GROUP BY region");
  ASSERT_TRUE(info.ok()) << info.status().ToString();
  ASSERT_EQ((*info)->endpoints().size(), 1u);

  auto reader = client_->DoGet(call_options, (*info)->endpoints()[0].ticket);
  ASSERT_TRUE(reader.ok()) << reader.status().ToString();

  auto table = (*reader)->ToTable();
  ASSERT_TRUE(table.ok()) << table.status().ToString();
  ASSERT_EQ((*table)->num_rows(), 2);

  auto combined = (*table)->CombineChunksToBatch();
  ASSERT_TRUE(combined.ok()) << combined.status().ToString();
  const auto& region_column = *std::static_pointer_cast<arrow::StringArray>((*combined)->column(0));
  const auto& total_column = *std::static_pointer_cast<arrow::DoubleArray>((*combined)->column(1));

  std::map<std::string, double> totals_by_region;
  for (int64_t i = 0; i < region_column.length(); ++i) {
    totals_by_region[region_column.GetString(i)] = total_column.Value(i);
  }
  EXPECT_DOUBLE_EQ(totals_by_region.at("A"), 10.0);
  EXPECT_DOUBLE_EQ(totals_by_region.at("B"), 35.0);
}

TEST_F(FlightSqlServerTest, InvalidSqlReturnsInvalidStatusNotACrash) {
  const flight::FlightCallOptions call_options;
  auto info = client_->Execute(call_options, "SELECT this is not valid sql");
  ASSERT_FALSE(info.ok());
  EXPECT_TRUE(info.status().IsInvalid()) << info.status().ToString();
}

// Regression test for the unbounded results_ registry: a client that calls
// Execute() repeatedly without ever draining via DoGet() (simulated here by
// simply never calling it) must eventually be rejected, not buffer results
// forever. Uses its own tiny server/fixture (rather than FlightSqlServerTest
// above) so it can configure max_pending_results down to 1.
TEST(FlightSqlServerPendingResultsCapTest, RejectsNewStatementsOncePendingResultsCapIsReached) {
  const fs::path dir = fs::temp_directory_path() / "kernellake_pending_results_cap_test";
  fs::create_directories(dir);
  const std::string path = (dir / "sales.parquet").string();

  arrow::Int64Builder id_builder;
  ASSERT_TRUE(id_builder.Append(1).ok());
  std::shared_ptr<arrow::Array> id_array;
  ASSERT_TRUE(id_builder.Finish(&id_array).ok());
  const auto schema = arrow::schema({arrow::field("id", arrow::int64(), false)});
  const auto table = arrow::Table::Make(schema, {id_array});
  auto sink = arrow::io::FileOutputStream::Open(path).ValueOrDie();
  ASSERT_TRUE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, /*chunk_size=*/1).ok());

  auto server =
      std::make_unique<KernelLakeFlightSqlServer>(cpu_backend_server_config(/*max_pending_results=*/1));
  auto location = flight::Location::ForGrpcTcp("127.0.0.1", 0);
  ASSERT_TRUE(location.ok()) << location.status().ToString();
  const flight::FlightServerOptions options(*location);
  ASSERT_TRUE(server->Init(options).ok());

  std::thread serve_thread([&server] {
    const arrow::Status status = server->Serve();
    EXPECT_TRUE(status.ok() || status.IsCancelled()) << status.ToString();
  });

  auto client_location = flight::Location::ForGrpcTcp("127.0.0.1", server->port());
  ASSERT_TRUE(client_location.ok()) << client_location.status().ToString();
  auto flight_client = flight::FlightClient::Connect(*client_location);
  ASSERT_TRUE(flight_client.ok()) << flight_client.status().ToString();
  flight_sql::FlightSqlClient client(std::move(*flight_client));

  const flight::FlightCallOptions call_options;
  const std::string sql = "SELECT id FROM read_parquet('" + path + "')";

  // First call fills the cap (1 buffered result, never fetched via DoGet).
  auto first = client.Execute(call_options, sql);
  ASSERT_TRUE(first.ok()) << first.status().ToString();

  // Second call must be rejected -- the cap is already full.
  auto second = client.Execute(call_options, sql);
  ASSERT_FALSE(second.ok());
  EXPECT_TRUE(second.status().IsOutOfMemory()) << second.status().ToString();

  ASSERT_TRUE(server->Shutdown().ok());
  serve_thread.join();
  fs::remove_all(dir);
}

// Diagnostic: GetFlightInfoStatement's max_pending_results cap is enforced
// via a check-then-act pattern -- lock, read results_.size(), unlock, run
// the (potentially slow) query, lock again, insert -- with the actual query
// execution happening *outside* the lock in between. Concurrent callers
// that all pass the check before any of them has inserted yet could all
// proceed to insert, growing results_ past the configured cap. This test
// fires many concurrent Execute() calls at a server configured with a
// small cap and counts how many succeed, to check empirically whether the
// cap is really enforced under concurrency or only under sequential calls
// (as RejectsNewStatementsOncePendingResultsCapIsReached above already
// covers).
TEST(FlightSqlServerPendingResultsCapTest, CapIsEnforcedUnderConcurrentCallers) {
  const fs::path dir = fs::temp_directory_path() / "kernellake_pending_results_cap_concurrency_test";
  fs::create_directories(dir);
  const std::string path = (dir / "sales.parquet").string();

  arrow::Int64Builder id_builder;
  for (int64_t i = 0; i < 5000; ++i) ASSERT_TRUE(id_builder.Append(i).ok());
  std::shared_ptr<arrow::Array> id_array;
  ASSERT_TRUE(id_builder.Finish(&id_array).ok());
  const auto schema = arrow::schema({arrow::field("id", arrow::int64(), false)});
  const auto table = arrow::Table::Make(schema, {id_array});
  auto sink = arrow::io::FileOutputStream::Open(path).ValueOrDie();
  ASSERT_TRUE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, /*chunk_size=*/100).ok());

  constexpr std::uint32_t kCap = 2;
  auto server = std::make_unique<KernelLakeFlightSqlServer>(cpu_backend_server_config(kCap));
  auto location = flight::Location::ForGrpcTcp("127.0.0.1", 0);
  ASSERT_TRUE(location.ok()) << location.status().ToString();
  const flight::FlightServerOptions options(*location);
  ASSERT_TRUE(server->Init(options).ok());

  std::thread serve_thread([&server] {
    const arrow::Status status = server->Serve();
    EXPECT_TRUE(status.ok() || status.IsCancelled()) << status.ToString();
  });

  const std::string sql = "SELECT id FROM read_parquet('" + path + "') ORDER BY id DESC";

  constexpr int kConcurrentCallers = 20;
  std::vector<std::thread> callers;
  std::atomic<int> succeeded{0};
  callers.reserve(kConcurrentCallers);
  for (int i = 0; i < kConcurrentCallers; ++i) {
    callers.emplace_back([&] {
      auto client_location = flight::Location::ForGrpcTcp("127.0.0.1", server->port());
      ASSERT_TRUE(client_location.ok()) << client_location.status().ToString();
      auto flight_client = flight::FlightClient::Connect(*client_location);
      ASSERT_TRUE(flight_client.ok()) << flight_client.status().ToString();
      flight_sql::FlightSqlClient client(std::move(*flight_client));
      const flight::FlightCallOptions call_options;
      auto info = client.Execute(call_options, sql);
      if (info.ok()) ++succeeded;
    });
  }
  for (std::thread& t : callers) t.join();

  ASSERT_TRUE(server->Shutdown().ok());
  serve_thread.join();
  fs::remove_all(dir);

  // None of these calls ever drain via DoGet(), so every successful one
  // left a permanent entry in results_ -- with the cap actually enforced,
  // at most kCap of the kConcurrentCallers calls should have succeeded.
  EXPECT_LE(succeeded.load(), static_cast<int>(kCap))
      << succeeded.load() << " of " << kConcurrentCallers
      << " concurrent Execute() calls succeeded against a max_pending_results cap of " << kCap
      << " -- the cap is not actually enforced under concurrent load.";
}

}  // namespace
}  // namespace kernellake
