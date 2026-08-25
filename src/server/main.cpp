#include <arrow/filesystem/s3fs.h>
#include <arrow/flight/server.h>
#include <arrow/flight/types.h>
#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "kernellake/common/config.hpp"
#include "kernellake/common/errors.hpp"
#include "kernellake/common/logging.hpp"
#include "kernellake/observability/query_tracing.hpp"
#include "kernellake/server/auth_middleware.hpp"
#include "kernellake/server/flight_sql_server.hpp"

namespace {

// Mirrors delta_txn_client.cpp's identical helper for the outbound-TLS
// case: read a PEM file whole into memory for handoff to Arrow
// Flight/gRPC, which take cert/key material as in-memory strings rather
// than paths.
std::string read_pem_file(const std::string& path, const char* config_key) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw kernellake::ConfigurationError(fmt::format("couldn't open {} '{}'", config_key, path));
  }
  std::ostringstream contents;
  contents << stream.rdbuf();
  return contents.str();
}

// Populates the TLS-related fields of FlightServerOptions from
// config.server when server.use_tls is set; a no-op (default-constructed,
// i.e. plaintext) otherwise.
arrow::flight::FlightServerOptions build_server_options(const arrow::flight::Location& location,
                                                        const kernellake::ServerSection& config) {
  arrow::flight::FlightServerOptions options(location);
  if (!config.use_tls) {
    return options;
  }
  arrow::flight::CertKeyPair cert_key;
  cert_key.pem_cert = read_pem_file(config.tls_cert_path, "server.tls_cert_path");
  cert_key.pem_key = read_pem_file(config.tls_key_path, "server.tls_key_path");
  options.tls_certificates.push_back(std::move(cert_key));
  options.verify_client = config.require_client_cert;
  if (config.require_client_cert) {
    options.root_certificates =
        read_pem_file(config.tls_client_ca_cert_path, "server.tls_client_ca_cert_path");
  }
  return options;
}

// Adds the bearer-token middleware from kernellake/server/auth_middleware.hpp
// to `options.middleware` when server.auth_enabled is set; a no-op
// otherwise. Called after build_server_options() populates the TLS fields
// above, since it mutates the same FlightServerOptions rather than building
// it from scratch.
void add_auth_middleware(arrow::flight::FlightServerOptions& options,
                         const kernellake::ServerSection& config) {
  if (!config.auth_enabled) {
    return;
  }
  options.middleware.emplace_back(
      "auth", std::make_shared<kernellake::BearerTokenMiddlewareFactory>(config.auth_token));
}

void print_usage() {
  std::printf(
      "kernellake-server - Arrow Flight SQL server for KernelLake\n\n"
      "Usage:\n"
      "  kernellake-server [--config <path>]\n\n"
      "Listens on server.host:server.port from the config file (default "
      "0.0.0.0:31337) and serves SQL queries via Arrow Flight SQL, running them\n"
      "through the same QueryEngine the `kernellake query` CLI command uses.\n"
      "engine.backend (\"gpu\" or \"cpu\") selects the execution backend, exactly as\n"
      "it does for the CLI.\n\n"
      "Set server.use_tls (plus server.tls_cert_path/tls_key_path) in the config\n"
      "file to serve over TLS; server.require_client_cert (plus\n"
      "server.tls_client_ca_cert_path) additionally enables mTLS.\n");
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string_view> args(argv + 1, argv + argc);

  if (!args.empty() && (args[0] == "--help" || args[0] == "-h")) {
    print_usage();
    return 0;
  }

  std::string config_path = "config/kernellake-server.yaml";
  bool explicit_config = false;
  if (!args.empty() && args[0] == "--config") {
    if (args.size() < 2) {
      std::fprintf(stderr, "kernellake-server: --config requires a path argument\n");
      return 1;
    }
    config_path = args[1];
    explicit_config = true;
  }

  kernellake::ServerConfig config;
  try {
    if (explicit_config || std::filesystem::exists(config_path)) {
      config = kernellake::load_server_config_file(config_path);
    } else {
      config = kernellake::default_server_config();
    }
    kernellake::validate_server_config(config);
    kernellake::init_logging(config.engine_config.logging);
    kernellake::observability::init(config.engine_config.observability);
  } catch (const kernellake::ConfigurationError& e) {
    std::fprintf(stderr, "kernellake-server: configuration error: %s\n", e.what());
    return 1;
  }

  // Ensures observability::shutdown() (flushing any buffered spans/metrics/
  // logs) runs on every return path below -- a no-op if
  // observability::init() above was itself a no-op (disabled, or
  // KERNELLAKE_ENABLE_OTEL=OFF).
  struct ObservabilityShutdownGuard {
    ~ObservabilityShutdownGuard() { kernellake::observability::shutdown(); }
  } observability_shutdown_guard;

  // See src/cli/main.cpp's identical guard: S3ObjectStore lazily calls
  // arrow::fs::EnsureS3Initialized() the first time an "s3://" URI is
  // opened, and FinalizeS3() must run once before exit if that happened, or
  // the AWS SDK segfaults at static-destruction time.
  struct S3ShutdownGuard {
    ~S3ShutdownGuard() {
      if (arrow::fs::IsS3Initialized()) {
        const arrow::Status status = arrow::fs::FinalizeS3();
        if (!status.ok()) {
          spdlog::warn("arrow::fs::FinalizeS3() failed: {}", status.ToString());
        }
      }
    }
  } s3_shutdown_guard;

  // Constructing the server (and, for backend == "gpu", the RmmEnvironment
  // it owns via GpuExecutionCoordinator) happens here rather than inside
  // the try/catch below that guards Init()/Serve() -- keeping construction
  // failures (e.g. "gpu" backend requested against a CPU-only build) in the
  // same reporting path as bind/serve failures.
  std::unique_ptr<kernellake::KernelLakeFlightSqlServer> server;
  try {
    server = std::make_unique<kernellake::KernelLakeFlightSqlServer>(config);

    auto location_result = config.server.use_tls
                               ? arrow::flight::Location::ForGrpcTls(config.server.host, config.server.port)
                               : arrow::flight::Location::ForGrpcTcp(config.server.host, config.server.port);
    if (!location_result.ok()) {
      std::fprintf(stderr, "kernellake-server: invalid server.host/server.port: %s\n",
                   location_result.status().ToString().c_str());
      return 1;
    }
    arrow::flight::FlightServerOptions options = build_server_options(*location_result, config.server);
    add_auth_middleware(options, config.server);

    const arrow::Status init_status = server->Init(options);
    if (!init_status.ok()) {
      std::fprintf(stderr, "kernellake-server: failed to start on %s:%d: %s\n", config.server.host.c_str(),
                   config.server.port, init_status.ToString().c_str());
      return 1;
    }

    spdlog::info("kernellake-server listening on {}:{} (backend={}, tls={})", config.server.host,
                 server->port(), config.engine_config.engine.backend, config.server.use_tls);

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
