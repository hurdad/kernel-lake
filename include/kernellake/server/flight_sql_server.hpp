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
// Phase 1 scope (see docs/ROADMAP.md): implements the RPCs a plain
// `FlightSqlClient::Execute(sql)` + `DoGet(ticket)` round trip needs
// (GetFlightInfoStatement/DoGetStatement) plus prepared-statement support
// (CreatePreparedStatement/ClosePreparedStatement/
// GetFlightInfoPreparedStatement) -- the latter added because the JDBC
// Flight SQL driver (used by DBeaver and most JDBC-based BI tools) routes
// *every* query through CreatePreparedStatement internally, even a plain
// `Statement.executeQuery()`; without it, no query at all could run from
// such a client (confirmed against a real JDBC driver). Every other
// FlightSqlServerBase RPC (catalogs/schemas/tables listing, SqlInfo,
// transactions, bulk ingest, ...) keeps its base-class default
// (NotImplemented) behavior; not scoped here.
//
// KernelLake has no parameter-binding support (no `?` placeholders in the
// SQL grammar), so a "prepared statement" here is just a named, pre-bound
// physical plan: CreatePreparedStatement runs parse/bind/plan (via
// QueryEngine::explain(), same as GetFlightInfoStatement, just without
// executing yet) and stores the resulting PhysicalPlanPtr, so a genuine
// syntax/binding error surfaces at prepare time like a real prepared
// statement, and the later execution skips redundant parse/bind/plan work.
// GetFlightInfoPreparedStatement then executes that stored plan through
// the exact same eager-execute-and-buffer path GetFlightInfoStatement
// uses, so DoGetStatement (unchanged) serves both kinds of query results
// from the same results_ registry -- there's no need for a separate
// DoGetPreparedStatement, since GetFlightInfoPreparedStatement hands back
// a ticket built the same way GetFlightInfoStatement's is.
//
// Query results are executed *eagerly* inside GetFlightInfoStatement/
// GetFlightInfoPreparedStatement and buffered in an in-process registry
// keyed by an opaque handle, which DoGetStatement looks up and streams
// from. This is a deliberate Phase 1 simplification: it avoids needing a
// live cursor kept open across two separate gRPC calls that may not even
// land on the same connection, at the cost of buffering the whole result
// in host memory between the two calls -- not production-grade
// streaming, and not pretended to be.
class KernelLakeFlightSqlServer : public arrow::flight::sql::FlightSqlServerBase {
 public:
  explicit KernelLakeFlightSqlServer(const EngineConfig& config);
  ~KernelLakeFlightSqlServer() override;

  arrow::Result<std::unique_ptr<arrow::flight::FlightInfo>> GetFlightInfoStatement(
      const arrow::flight::ServerCallContext& context, const arrow::flight::sql::StatementQuery& command,
      const arrow::flight::FlightDescriptor& descriptor) override;

  arrow::Result<std::unique_ptr<arrow::flight::FlightDataStream>> DoGetStatement(
      const arrow::flight::ServerCallContext& context,
      const arrow::flight::sql::StatementQueryTicket& command) override;

  arrow::Result<arrow::flight::sql::ActionCreatePreparedStatementResult> CreatePreparedStatement(
      const arrow::flight::ServerCallContext& context,
      const arrow::flight::sql::ActionCreatePreparedStatementRequest& request) override;

  arrow::Status ClosePreparedStatement(
      const arrow::flight::ServerCallContext& context,
      const arrow::flight::sql::ActionClosePreparedStatementRequest& request) override;

  arrow::Result<std::unique_ptr<arrow::flight::FlightInfo>> GetFlightInfoPreparedStatement(
      const arrow::flight::ServerCallContext& context,
      const arrow::flight::sql::PreparedStatementQuery& command,
      const arrow::flight::FlightDescriptor& descriptor) override;

 private:
  struct PreparedStatementEntry {
    std::string sql;  // kept only for observability::QuerySpan's sql attribute at execute time.
    PhysicalPlanPtr physical;
  };

  // Shared by GetFlightInfoStatement (which explains `sql` itself first)
  // and GetFlightInfoPreparedStatement (which already has `physical`/`sql`
  // from a prior CreatePreparedStatement call): runs the cap-check-and-
  // reserve, executes on the configured backend, buffers the result in
  // results_, and builds the returned FlightInfo/ticket.
  arrow::Result<std::unique_ptr<arrow::flight::FlightInfo>> ExecuteAndBuffer(
      const PhysicalPlanPtr& physical, std::string_view sql, const arrow::flight::FlightDescriptor& descriptor);

  EngineConfig config_;
  QueryEngine engine_;
  // Only constructed for config_.engine.backend == "gpu" (see
  // GpuExecutionCoordinator's own doc comment); null for "cpu".
  std::unique_ptr<GpuExecutionCoordinator> gpu_coordinator_;

  std::mutex results_mutex_;
  std::unordered_map<std::string, QueryResult> results_;
  // Reservations for in-flight queries that have passed the
  // max_pending_results check but haven't landed in results_ yet --
  // incremented in the same critical section as the check itself, so the
  // check-and-reserve is atomic. Without this, two concurrent
  // GetFlightInfoStatement calls could both observe results_.size() below
  // the cap (since neither has inserted yet), both proceed to execute
  // their query, and both insert -- defeating the cap under concurrent
  // callers even though it holds under sequential ones. See
  // FlightSqlServerPendingResultsCapTest.CapIsEnforcedUnderConcurrentCallers.
  std::uint32_t pending_count_ = 0;
  std::uint64_t next_handle_ = 0;
  // A client that calls CreatePreparedStatement repeatedly without ever
  // calling ClosePreparedStatement (a leaked JDBC PreparedStatement, or a
  // buggy/malicious client) must not be able to grow this map without
  // bound either -- same rationale as max_pending_results above, just for
  // un-executed prepared plans instead of buffered results. Reuses
  // max_pending_results as the cap rather than adding a second config
  // field for what's the same underlying concern (bounding how much a
  // non-cooperating client can make this server hold on its behalf).
  std::unordered_map<std::string, PreparedStatementEntry> prepared_;
};

}  // namespace kernellake
