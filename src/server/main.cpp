#include <arrow/flight/server.h>
#include <arrow/flight/types.h>
#include <spdlog/spdlog.h>

#include <cstdio>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "kernellake/common/config.hpp"
#include "kernellake/common/errors.hpp"
#include "kernellake/common/logging.hpp"
#include "kernellake/server/flight_sql_server.hpp"

namespace {

void print_usage() {
  std::printf(
      "kernellake-server - Arrow Flight SQL server for KernelLake\n\n"
      "Usage:\n"
      "  kernellake-server [--config <path>]\n\n"
      "Listens on server.host:server.port from the config file (default "
      "0.0.0.0:31337) and serves SQL queries via Arrow Flight SQL, running them\n"
      "through the same QueryEngine the `kernellake query` CLI command uses.\n"
      "engine.backend (\"gpu\" or \"cpu\") selects the execution backend, exactly as\n"
      "it does for the CLI.\n");
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string_view> args(argv + 1, argv + argc);

  if (!args.empty() && (args[0] == "--help" || args[0] == "-h")) {
    print_usage();
    return 0;
  }

  std::string config_path = "config/kernellake.yaml";
  bool explicit_config = false;
  if (!args.empty() && args[0] == "--config") {
    if (args.size() < 2) {
      std::fprintf(stderr, "kernellake-server: --config requires a path argument\n");
      return 1;
    }
    config_path = args[1];
    explicit_config = true;
  }

  kernellake::EngineConfig config;
  try {
    if (explicit_config || std::filesystem::exists(config_path)) {
      config = kernellake::load_config_file(config_path);
    } else {
      config = kernellake::default_config();
    }
    kernellake::validate_config(config);
    kernellake::init_logging(config.logging);
  } catch (const kernellake::ConfigurationError& e) {
    std::fprintf(stderr, "kernellake-server: configuration error: %s\n", e.what());
    return 1;
  }

  // Constructing the server (and, for backend == "gpu", the RmmEnvironment
  // it owns via GpuExecutionCoordinator) happens here rather than inside
  // the try/catch below that guards Init()/Serve() -- keeping construction
  // failures (e.g. "gpu" backend requested against a CPU-only build) in the
  // same reporting path as bind/serve failures.
  std::unique_ptr<kernellake::KernelLakeFlightSqlServer> server;
  try {
    server = std::make_unique<kernellake::KernelLakeFlightSqlServer>(config);

    auto location_result = arrow::flight::Location::ForGrpcTcp(config.server.host, config.server.port);
    if (!location_result.ok()) {
      std::fprintf(stderr, "kernellake-server: invalid server.host/server.port: %s\n",
                   location_result.status().ToString().c_str());
      return 1;
    }
    const arrow::flight::FlightServerOptions options(*location_result);

    const arrow::Status init_status = server->Init(options);
    if (!init_status.ok()) {
      std::fprintf(stderr, "kernellake-server: failed to start on %s:%d: %s\n", config.server.host.c_str(),
                   config.server.port, init_status.ToString().c_str());
      return 1;
    }

    spdlog::info("kernellake-server listening on {}:{} (backend={})", config.server.host, server->port(),
                 config.engine.backend);

    const arrow::Status serve_status = server->Serve();
    if (!serve_status.ok()) {
      std::fprintf(stderr, "kernellake-server: %s\n", serve_status.ToString().c_str());
      return 1;
    }
  } catch (const kernellake::KernelLakeError& e) {
    std::fprintf(stderr, "kernellake-server: %s\n", e.what());
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "kernellake-server: unexpected error: %s\n", e.what());
    return 1;
  }

  return 0;
}
