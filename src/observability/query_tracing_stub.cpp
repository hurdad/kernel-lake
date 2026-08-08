// Provides a no-op kernellake::observability implementation for builds with
// KERNELLAKE_ENABLE_OTEL=OFF (the default). Mutually exclusive with
// query_tracing_otel.cpp -- see that file's comment. Deliberately includes
// no opentelemetry-cpp header, so this file plus query_tracing.hpp give
// call sites a genuinely zero-footprint no-op.
#include "kernellake/observability/query_tracing.hpp"

#include <spdlog/spdlog.h>

#include "kernellake/common/config.hpp"

namespace kernellake::observability {

// Never constructed; exists only so unique_ptr<Impl>'s special members
// below have a complete type to compile against.
struct QuerySpan::Impl {};

QuerySpan::QuerySpan(QuerySpan&&) noexcept = default;
QuerySpan& QuerySpan::operator=(QuerySpan&&) noexcept = default;
QuerySpan::~QuerySpan() = default;

void init(const ObservabilitySection& config) {
  if (config.enabled) {
    spdlog::warn(
        "observability.enabled is true, but this build was compiled with "
        "-DKERNELLAKE_ENABLE_OTEL=OFF (the default); tracing/metrics/logs export stays "
        "disabled. Rebuild with -DKERNELLAKE_ENABLE_OTEL=ON (requires opentelemetry-cpp-dev) "
        "to enable OTLP/gRPC export.");
  }
}

void shutdown() {}

QuerySpan start_query_span(std::string_view /*operation_name*/) {
  return QuerySpan();
}

void QuerySpan::finish(const QueryResult& /*result*/, std::string_view /*sql*/,
                       std::string_view /*backend*/) {}

void QuerySpan::finish_error(const std::exception& /*e*/, std::string_view /*sql*/,
                             std::string_view /*backend*/) {}

// Never constructed; exists only so unique_ptr<Impl>'s special members
// below have a complete type to compile against.
struct ClientSpan::Impl {};

ClientSpan::ClientSpan(ClientSpan&&) noexcept = default;
ClientSpan& ClientSpan::operator=(ClientSpan&&) noexcept = default;
ClientSpan::~ClientSpan() = default;

ClientSpan start_client_span(std::string_view /*operation_name*/) {
  return ClientSpan();
}

ClientSpan start_client_span(std::string_view /*operation_name*/, const ClientSpan& /*parent*/) {
  return ClientSpan();
}

void ClientSpan::inject(const std::function<void(std::string_view, std::string_view)>& /*setter*/) const {}

void ClientSpan::set_attribute(std::string_view /*key*/, double /*value*/) {}
void ClientSpan::set_attribute(std::string_view /*key*/, std::int64_t /*value*/) {}

void ClientSpan::finish_ok() {}

void ClientSpan::finish_error(std::string_view /*error_message*/) {}

}  // namespace kernellake::observability
