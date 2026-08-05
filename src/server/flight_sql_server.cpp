#include "kernellake/server/flight_sql_server.hpp"

#include <arrow/flight/types.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/type.h>
#include <fmt/format.h>

#include <utility>

#include "kernellake/common/errors.hpp"
#include "kernellake/observability/query_tracing.hpp"
#include "kernellake/planner/physical_plan.hpp"
#include "kernellake/types/arrow_adapter.hpp"

namespace kernellake {

namespace {

namespace flight = arrow::flight;
namespace flight_sql = arrow::flight::sql;

// Maps KernelLakeError subclasses (kernellake/common/errors.hpp) onto the
// closest-matching generic arrow::Status factory. Arrow's own gRPC Flight
// transport translates these StatusCodes to gRPC status codes automatically
// (e.g. StatusCode::Invalid -> INVALID_ARGUMENT); FlightStatusCode
// (arrow::flight::MakeFlightError) only needs to be used for the handful of
// RPC-specific conditions (Unauthenticated, TimedOut, ...) that have no
// generic arrow::Status equivalent, none of which apply to the errors
// QueryEngine itself can throw.
arrow::Status ToFlightStatus(const KernelLakeError& error) {
  if (dynamic_cast<const SqlError*>(&error) || dynamic_cast<const BindingError*>(&error) ||
      dynamic_cast<const PlanningError*>(&error) || dynamic_cast<const OptimizationError*>(&error)) {
    return arrow::Status::Invalid(error.what());
  }
  if (dynamic_cast<const StorageError*>(&error)) {
    return arrow::Status::IOError(error.what());
  }
  if (dynamic_cast<const OutOfMemoryError*>(&error)) {
    return arrow::Status::OutOfMemory(error.what());
  }
  if (dynamic_cast<const ExecutionError*>(&error) || dynamic_cast<const CudaError*>(&error)) {
    return arrow::Status::ExecutionError(error.what());
  }
  // ConfigurationError/BenchmarkError aren't expected to surface from a
  // per-request call (configuration errors happen at server startup,
  // benchmark errors don't apply here) -- fall through to a generic error
  // rather than asserting, since a server should never crash on a request.
  return arrow::Status::UnknownError(error.what());
}

}  // namespace

KernelLakeFlightSqlServer::KernelLakeFlightSqlServer(const EngineConfig& config)
    : config_(config), engine_(config) {
  if (config_.engine.backend == "gpu") {
    gpu_coordinator_ = std::make_unique<GpuExecutionCoordinator>(config_);
  }
}

KernelLakeFlightSqlServer::~KernelLakeFlightSqlServer() = default;

arrow::Result<std::unique_ptr<flight::FlightInfo>> KernelLakeFlightSqlServer::ExecuteAndBuffer(
    const PhysicalPlanPtr& physical, std::string_view sql, const flight::FlightDescriptor& descriptor) {
  QueryResult result;
  observability::QuerySpan span =
      observability::start_query_span("kernellake.flight_sql.get_flight_info_statement");
  bool reserved = false;
  try {
    // Reject before doing any work if the buffered-but-unfetched result
    // registry is already full (see ServerSection::max_pending_results'
    // own comment) -- a client that never calls DoGetStatement to drain it
    // must not be able to grow this map without bound. The reservation
    // (pending_count_) is incremented in the SAME critical section as the
    // check, not just checked and inserted later once the (potentially
    // slow) query finishes: two concurrent callers that both see
    // results_.size() below the cap before either has inserted yet would
    // otherwise both proceed to execute and both insert, defeating the cap
    // under concurrent load even though it holds sequentially -- confirmed
    // via a concurrent-caller stress test before this fix (20 of 20 calls
    // succeeded against a cap of 2).
    {
      const std::lock_guard<std::mutex> lock(results_mutex_);
      if (results_.size() + pending_count_ >= config_.server.max_pending_results) {
        throw OutOfMemoryError(fmt::format(
            "too many buffered query results awaiting fetch (limit: {}); fetch or abandon existing "
            "results before issuing more statements",
            config_.server.max_pending_results));
      }
      ++pending_count_;
      reserved = true;
    }
    result = config_.engine.backend == "cpu" ? engine_.execute_cpu(physical)
                                             : gpu_coordinator_->execute(engine_, physical);
    span.finish(result, sql, config_.engine.backend);
  } catch (const KernelLakeError& e) {
    span.finish_error(e, sql, config_.engine.backend);
    if (reserved) {
      const std::lock_guard<std::mutex> lock(results_mutex_);
      --pending_count_;
    }
    return ToFlightStatus(e);
  } catch (const std::exception& e) {
    span.finish_error(e, sql, config_.engine.backend);
    if (reserved) {
      const std::lock_guard<std::mutex> lock(results_mutex_);
      --pending_count_;
    }
    return arrow::Status::UnknownError(e.what());
  }

  std::int64_t total_records = 0;
  for (const auto& batch : result.batches) {
    total_records += batch->num_rows();
  }

  std::string handle;
  std::shared_ptr<arrow::Schema> schema = result.schema;
  {
    const std::lock_guard<std::mutex> lock(results_mutex_);
    --pending_count_;
    handle = "q-" + std::to_string(next_handle_++);
    results_.emplace(handle, std::move(result));
  }

  ARROW_ASSIGN_OR_RAISE(std::string ticket_string, flight_sql::CreateStatementQueryTicket(handle));
  const flight::FlightEndpoint endpoint{flight::Ticket{std::move(ticket_string)}, {}, std::nullopt, ""};

  ARROW_ASSIGN_OR_RAISE(flight::FlightInfo info, flight::FlightInfo::Make(*schema, descriptor, {endpoint},
                                                                          total_records, /*total_bytes=*/-1));
  return std::make_unique<flight::FlightInfo>(std::move(info));
}

arrow::Result<std::unique_ptr<flight::FlightInfo>> KernelLakeFlightSqlServer::GetFlightInfoStatement(
    const flight::ServerCallContext& /*context*/, const flight_sql::StatementQuery& command,
    const flight::FlightDescriptor& descriptor) {
  PhysicalPlanPtr physical;
  try {
    physical = engine_.explain(command.query);
  } catch (const KernelLakeError& e) {
    return ToFlightStatus(e);
  } catch (const std::exception& e) {
    return arrow::Status::UnknownError(e.what());
  }
  return ExecuteAndBuffer(physical, command.query, descriptor);
}

arrow::Result<flight_sql::ActionCreatePreparedStatementResult> KernelLakeFlightSqlServer::CreatePreparedStatement(
    const flight::ServerCallContext& /*context*/,
    const flight_sql::ActionCreatePreparedStatementRequest& request) {
  PhysicalPlanPtr physical;
  try {
    physical = engine_.explain(request.query);
  } catch (const KernelLakeError& e) {
    return ToFlightStatus(e);
  } catch (const std::exception& e) {
    return arrow::Status::UnknownError(e.what());
  }

  std::string handle;
  {
    const std::lock_guard<std::mutex> lock(results_mutex_);
    // Same class of guard as GetFlightInfoStatement's results_ cap (see its
    // own comment): a client that never calls ClosePreparedStatement must
    // not be able to grow prepared_ without bound.
    if (prepared_.size() >= config_.server.max_pending_results) {
      return arrow::Status::OutOfMemory(fmt::format(
          "too many prepared statements awaiting close (limit: {}); close existing prepared "
          "statements before creating more",
          config_.server.max_pending_results));
    }
    handle = "p-" + std::to_string(next_handle_++);
    prepared_.emplace(handle, PreparedStatementEntry{request.query, physical});
  }

  flight_sql::ActionCreatePreparedStatementResult result;
  result.dataset_schema = to_arrow_schema(physical->output_schema());
  result.parameter_schema = nullptr;  // KernelLake's SQL grammar has no bound-parameter ("?") support.
  result.prepared_statement_handle = std::move(handle);
  return result;
}

arrow::Status KernelLakeFlightSqlServer::ClosePreparedStatement(
    const flight::ServerCallContext& /*context*/,
    const flight_sql::ActionClosePreparedStatementRequest& request) {
  const std::lock_guard<std::mutex> lock(results_mutex_);
  prepared_.erase(request.prepared_statement_handle);
  return arrow::Status::OK();
}

arrow::Result<std::unique_ptr<flight::FlightInfo>> KernelLakeFlightSqlServer::GetFlightInfoPreparedStatement(
    const flight::ServerCallContext& /*context*/, const flight_sql::PreparedStatementQuery& command,
    const flight::FlightDescriptor& descriptor) {
  PhysicalPlanPtr physical;
  std::string sql;
  {
    const std::lock_guard<std::mutex> lock(results_mutex_);
    const auto it = prepared_.find(command.prepared_statement_handle);
    if (it == prepared_.end()) {
      return arrow::Status::KeyError("no prepared statement for handle '" +
                                     command.prepared_statement_handle +
                                     "' (already closed, or the server restarted)");
    }
    physical = it->second.physical;
    sql = it->second.sql;
  }
  return ExecuteAndBuffer(physical, sql, descriptor);
}

arrow::Result<std::unique_ptr<flight::FlightDataStream>> KernelLakeFlightSqlServer::DoGetStatement(
    const flight::ServerCallContext& /*context*/, const flight_sql::StatementQueryTicket& command) {
  QueryResult result;
  {
    const std::lock_guard<std::mutex> lock(results_mutex_);
    const auto it = results_.find(command.statement_handle);
    if (it == results_.end()) {
      return arrow::Status::KeyError("no query result for statement handle '" + command.statement_handle +
                                     "' (already consumed, or the server restarted)");
    }
    result = std::move(it->second);
    results_.erase(it);
  }

  ARROW_ASSIGN_OR_RAISE(std::shared_ptr<arrow::RecordBatchReader> reader,
                        arrow::RecordBatchReader::Make(std::move(result.batches), result.schema));
  return std::make_unique<flight::RecordBatchStream>(reader);
}

}  // namespace kernellake
