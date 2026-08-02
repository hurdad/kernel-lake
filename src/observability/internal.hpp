#pragma once
// Private, src/-local header shared only between query_tracing_otel.cpp and
// query_tracing_test_support.cpp -- never installed, never included from
// include/kernellake/. Exists because init_for_testing() (test-only) had to
// move into its own translation unit: it references
// opentelemetry-cpp::in_memory_span_exporter/in_memory_metric_exporter,
// which only tests/unit/CMakeLists.txt links -- keeping it in the same
// query_tracing_otel.cpp.o as init() pulled that whole object file (and its
// then-undefined in-memory-exporter symbols) into the CLI/server binaries
// too, since static archives link at object-file granularity, not per
// symbol (confirmed by an actual link failure, not assumed).
#include <opentelemetry/logs/provider.h>
#include <opentelemetry/metrics/provider.h>
#include <opentelemetry/nostd/shared_ptr.h>
#include <opentelemetry/nostd/unique_ptr.h>
#include <opentelemetry/sdk/resource/resource.h>
#include <opentelemetry/trace/provider.h>
#include <spdlog/sinks/base_sink.h>

#include <memory>
#include <mutex>
#include <string>
#include <string_view>

namespace kernellake::observability::detail {

// Note: no `otlp` alias here (opentelemetry::exporter::otlp) -- only
// query_tracing_otel.cpp touches the OTLP/gRPC exporter types;
// query_tracing_test_support.cpp uses the in-memory exporters instead, so
// keeping that alias (and its header) out of this shared file avoids
// coupling test-support code to an otlp-specific include.
namespace trace_sdk = opentelemetry::sdk::trace;
namespace metrics_sdk = opentelemetry::sdk::metrics;
namespace logs_sdk = opentelemetry::sdk::logs;
namespace resource_sdk = opentelemetry::sdk::resource;
namespace trace_api = opentelemetry::trace;
namespace metrics_api = opentelemetry::metrics;
namespace logs_api = opentelemetry::logs;

inline opentelemetry::nostd::string_view to_otel(std::string_view s) {
  return opentelemetry::nostd::string_view(s.data(), s.size());
}

// nostd::shared_ptr<Base>(std::move(some_unique_ptr<Derived>)) is ambiguous
// in this ABI version (three equally-viable candidate constructors: direct
// std::unique_ptr<T>, OTel's own nostd::unique_ptr<T>, and std::shared_ptr<T>
// -- confirmed by an actual build error, not assumed) -- going through an
// exact-type nostd::shared_ptr<Derived> first, then letting that upcast via
// nostd::shared_ptr's own templated shared_ptr<U>&& constructor, resolves
// unambiguously.
template <class Base, class Derived>
opentelemetry::nostd::shared_ptr<Base> to_provider(std::unique_ptr<Derived> provider) {
  opentelemetry::nostd::shared_ptr<Derived> derived_provider(std::move(provider));
  return derived_provider;
}

// Bridges every existing spdlog::info/warn/error call in this codebase into
// OTel's Logs signal -- no existing call site needs to change. Pushed onto
// spdlog::default_logger()->sinks() by init()/init_for_testing(), alongside
// whatever console sink init_logging() already configured (console output
// is unaffected; logs are just also exported).
class OtelSpdlogSink : public spdlog::sinks::base_sink<std::mutex> {
 public:
  explicit OtelSpdlogSink(std::string logger_name) : logger_name_(std::move(logger_name)) {}

 protected:
  void sink_it_(const spdlog::details::log_msg& msg) override {
    logs_api::Provider::GetLoggerProvider()
        ->GetLogger(logger_name_)
        ->EmitLogRecord(to_severity(msg.level), to_otel(std::string_view(msg.payload.data(), msg.payload.size())),
                        opentelemetry::common::SystemTimestamp(msg.time));
  }

  void flush_() override {}

 private:
  static logs_api::Severity to_severity(spdlog::level::level_enum level) {
    switch (level) {
      case spdlog::level::trace:
        return logs_api::Severity::kTrace;
      case spdlog::level::debug:
        return logs_api::Severity::kDebug;
      case spdlog::level::info:
        return logs_api::Severity::kInfo;
      case spdlog::level::warn:
        return logs_api::Severity::kWarn;
      case spdlog::level::err:
        return logs_api::Severity::kError;
      case spdlog::level::critical:
        return logs_api::Severity::kFatal;
      default:
        return logs_api::Severity::kInfo;
    }
  }

  std::string logger_name_;
};

inline resource_sdk::Resource build_resource(const std::string& service_name) {
  return resource_sdk::Resource::Create({{"service.name", service_name}});
}

// Defined once in query_tracing_otel.cpp; read/written by both it and
// query_tracing_test_support.cpp (a test only ever links one binary at a
// time, so there is exactly one definition per process either way).
extern bool g_enabled;
extern std::string g_tracer_name;
extern opentelemetry::nostd::unique_ptr<metrics_api::Histogram<double>> g_query_duration_histogram;

}  // namespace kernellake::observability::detail
