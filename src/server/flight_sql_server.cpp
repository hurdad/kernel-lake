#include "kernellake/server/flight_sql_server.hpp"

#include <arrow/flight/types.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/type.h>

#include <utility>

#include "kernellake/common/errors.hpp"
#include "kernellake/planner/physical_plan.hpp"

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

KernelLakeFlightSqlServer::KernelLakeFlightSqlServer(EngineConfig config) : config_(config), engine_(config) {
  if (config_.engine.backend == "gpu") {
    gpu_coordinator_ = std::make_unique<GpuExecutionCoordinator>(config_);
  }
}

KernelLakeFlightSqlServer::~KernelLakeFlightSqlServer() = default;

arrow::Result<std::unique_ptr<flight::FlightInfo>> KernelLakeFlightSqlServer::GetFlightInfoStatement(
    const flight::ServerCallContext& /*context*/, const flight_sql::StatementQuery& command,
    const flight::FlightDescriptor& descriptor) {
  QueryResult result;
  try {
    const PhysicalPlanPtr physical = engine_.explain(command.query);
    result = config_.engine.backend == "cpu" ? engine_.execute_cpu(physical)
                                             : gpu_coordinator_->execute(engine_, physical);
  } catch (const KernelLakeError& e) {
    return ToFlightStatus(e);
  } catch (const std::exception& e) {
    return arrow::Status::UnknownError(e.what());
  }

  std::int64_t total_records = 0;
  for (const auto& batch : result.batches) total_records += batch->num_rows();

  std::string handle;
  std::shared_ptr<arrow::Schema> schema = result.schema;
  {
    const std::lock_guard<std::mutex> lock(results_mutex_);
    handle = "q-" + std::to_string(next_handle_++);
    results_.emplace(handle, std::move(result));
  }

  ARROW_ASSIGN_OR_RAISE(std::string ticket_string, flight_sql::CreateStatementQueryTicket(handle));
  const flight::FlightEndpoint endpoint{flight::Ticket{std::move(ticket_string)}, {}, std::nullopt, ""};

  ARROW_ASSIGN_OR_RAISE(flight::FlightInfo info, flight::FlightInfo::Make(*schema, descriptor, {endpoint},
                                                                          total_records, /*total_bytes=*/-1));
  return std::make_unique<flight::FlightInfo>(std::move(info));
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
