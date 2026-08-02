#pragma once

#include <arrow/flight/sql/server.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "kernellake/api/query_engine.hpp"
#include "kernellake/common/config.hpp"
#include "kernellake/server/gpu_execution_coordinator.hpp"

namespace kernellake {

// A Flight SQL server wired to a real kernellake::QueryEngine: it runs
// arriving SQL statements through the same parse/bind/plan/execute path
// the CLI's `kernellake query` command uses, over either backend
// (config.engine.backend, exactly as `--backend` already selects for the
// CLI), and streams results back as Arrow RecordBatches over gRPC.
//
// Phase 1 scope (see docs/ROADMAP.md): implements just the two RPCs a
// plain `FlightSqlClient::Execute(sql)` + `DoGet(ticket)` round trip needs
// -- GetFlightInfoStatement and DoGetStatement. Every other FlightSqlServerBase
// RPC (prepared statements, catalogs/schemas/tables listing, SqlInfo, ...)
// keeps its base-class default (NotImplemented) behavior; not scoped here.
//
// Query results are executed *eagerly* inside GetFlightInfoStatement (the
// first RPC) and buffered in an in-process registry keyed by an opaque
// handle, which DoGetStatement (the second RPC) looks up and streams from.
// This is a deliberate Phase 1 simplification: it avoids needing a live
// cursor kept open across two separate gRPC calls that may not even land on
// the same connection, at the cost of buffering the whole result in host
// memory between the two calls -- not production-grade streaming, and not
// pretended to be.
class KernelLakeFlightSqlServer : public arrow::flight::sql::FlightSqlServerBase {
 public:
  explicit KernelLakeFlightSqlServer(EngineConfig config);
  ~KernelLakeFlightSqlServer() override;

  arrow::Result<std::unique_ptr<arrow::flight::FlightInfo>> GetFlightInfoStatement(
      const arrow::flight::ServerCallContext& context, const arrow::flight::sql::StatementQuery& command,
      const arrow::flight::FlightDescriptor& descriptor) override;

  arrow::Result<std::unique_ptr<arrow::flight::FlightDataStream>> DoGetStatement(
      const arrow::flight::ServerCallContext& context,
      const arrow::flight::sql::StatementQueryTicket& command) override;

 private:
  EngineConfig config_;
  QueryEngine engine_;
  // Only constructed for config_.engine.backend == "gpu" (see
  // GpuExecutionCoordinator's own doc comment); null for "cpu".
  std::unique_ptr<GpuExecutionCoordinator> gpu_coordinator_;

  std::mutex results_mutex_;
  std::unordered_map<std::string, QueryResult> results_;
  std::uint64_t next_handle_ = 0;
};

}  // namespace kernellake
