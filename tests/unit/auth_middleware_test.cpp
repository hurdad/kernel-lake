// End-to-end test for BearerTokenMiddlewareFactory: starts a real
// KernelLakeFlightSqlServer with the middleware registered exactly as
// main.cpp wires it up (see build_server_options()/add_auth_middleware()
// there), then drives it with a real arrow::flight::sql::FlightSqlClient
// over gRPC -- proving the "authorization: Bearer <token>" header is
// actually enforced on the wire, not just that StartCall() compiles.
// Mirrors flight_sql_server_test.cpp's own fixture shape.
#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/flight/api.h>
#include <arrow/flight/sql/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <filesystem>
#include <memory>
#include <string>
#include <thread>

#include "kernellake/common/config.hpp"
#include "kernellake/server/auth_middleware.hpp"
#include "kernellake/server/flight_sql_server.hpp"

namespace kernellake {
namespace {

namespace fs = std::filesystem;
namespace flight = arrow::flight;
namespace flight_sql = arrow::flight::sql;

constexpr const char* kToken = "correct-token";

class AuthMiddlewareTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() / "kernellake_auth_middleware_test";
    fs::create_directories(dir_);
    path_ = (dir_ / "sales.parquet").string();

    arrow::Int64Builder id_builder;
    for (int64_t i = 0; i < 3; ++i) {
      ASSERT_TRUE(id_builder.Append(i).ok());
    }
    std::shared_ptr<arrow::Array> id_array;
    ASSERT_TRUE(id_builder.Finish(&id_array).ok());
    const auto schema = arrow::schema({arrow::field("id", arrow::int64(), false)});
    const auto table = arrow::Table::Make(schema, {id_array});
    auto sink = arrow::io::FileOutputStream::Open(path_).ValueOrDie();
    const arrow::Status write_status =
        parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, /*chunk_size=*/3);
    ASSERT_TRUE(write_status.ok()) << write_status.ToString();

    ServerConfig config = default_server_config();
    config.engine_config.engine.backend = "cpu";
    config.server.host = "127.0.0.1";
    config.server.port = 0;  // OS-assigned ephemeral port, read back after Init().

    server_ = std::make_unique<KernelLakeFlightSqlServer>(config);
    auto location = flight::Location::ForGrpcTcp("127.0.0.1", 0);
    ASSERT_TRUE(location.ok()) << location.status().ToString();
    flight::FlightServerOptions options(*location);
    options.middleware.emplace_back("auth", std::make_shared<BearerTokenMiddlewareFactory>(kToken));
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

TEST_F(AuthMiddlewareTest, RejectsCallWithNoAuthorizationHeader) {
  const flight::FlightCallOptions call_options;
  auto info = client_->Execute(call_options, "SELECT id FROM read_parquet('" + path_ + "')");
  ASSERT_FALSE(info.ok());
  EXPECT_TRUE(info.status().IsIOError() ||
              info.status().ToString().find("Unauthenticated") != std::string::npos)
      << info.status().ToString();
}

TEST_F(AuthMiddlewareTest, RejectsCallWithWrongToken) {
  flight::FlightCallOptions call_options;
  call_options.headers.emplace_back("authorization", "Bearer wrong-token");
  auto info = client_->Execute(call_options, "SELECT id FROM read_parquet('" + path_ + "')");
  ASSERT_FALSE(info.ok());
}

TEST_F(AuthMiddlewareTest, AcceptsCallWithCorrectToken) {
  flight::FlightCallOptions call_options;
  call_options.headers.emplace_back("authorization", std::string("Bearer ") + kToken);
  auto info = client_->Execute(call_options, "SELECT id FROM read_parquet('" + path_ + "')");
  ASSERT_TRUE(info.ok()) << info.status().ToString();

  auto reader = client_->DoGet(call_options, (*info)->endpoints()[0].ticket);
  ASSERT_TRUE(reader.ok()) << reader.status().ToString();
  auto table = (*reader)->ToTable();
  ASSERT_TRUE(table.ok()) << table.status().ToString();
  EXPECT_EQ((*table)->num_rows(), 3);
}

}  // namespace
}  // namespace kernellake
