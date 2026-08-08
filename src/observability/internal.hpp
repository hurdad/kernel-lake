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
#include <opentelemetry/sdk/metrics/aggregation/aggregation_config.h>
#include <opentelemetry/sdk/metrics/instruments.h>
#include <opentelemetry/sdk/metrics/meter_provider.h>
#include <opentelemetry/sdk/metrics/view/instrument_selector_factory.h>
#include <opentelemetry/sdk/metrics/view/meter_selector_factory.h>
#include <opentelemetry/sdk/metrics/view/view_factory.h>
#include <opentelemetry/sdk/resource/resource.h>
#include <opentelemetry/trace/provider.h>
#include <spdlog/sinks/base_sink.h>

#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

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
        ->EmitLogRecord(to_severity(msg.level),
                        to_otel(std::string_view(msg.payload.data(), msg.payload.size())),
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

// Registers an explicit-bucket-boundary View for kernellake.query.duration_
// seconds, replacing the OTel SDK's own default histogram boundaries ([0,
// 5, 10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000]).
// Those defaults are tuned for an unscaled/millisecond-ish value range --
// this metric records values in *seconds* (QueryResult::
// elapsed_wall_seconds), so every real query observed so far (sub-second to
// a few seconds, even at real TPC-H scale factors) fell into the single
// "<=5" bucket, making p50/p95/p99 (histogram_quantile() in Grafana) and
// any heatmap built from this histogram meaningless -- confirmed for real
// against benchmarks/local/'s own Prometheus (`le="0.0"` count=0,
// `le="5.0"` count=8, identical through `le="10000.0"`, for query latencies
// that were actually all under 0.35s). Must be registered on the *SDK*
// MeterProvider (via its own AddView(), only available on the concrete
// type, not the type-erased API-layer nostd::shared_ptr<metrics_api::
// MeterProvider> Provider::SetMeterProvider() takes) before any instrument
// is created against it -- callers must call this before their own
// CreateDoubleHistogram("kernellake.query.duration_seconds", ...) call.
inline void add_query_duration_histogram_view(metrics_sdk::MeterProvider& provider,
                                              const std::string& service_name) {
  auto instrument_selector = metrics_sdk::InstrumentSelectorFactory::Create(
      metrics_sdk::InstrumentType::kHistogram, "kernellake.query.duration_seconds", "s");
  auto meter_selector = metrics_sdk::MeterSelectorFactory::Create(service_name, "", "");
  auto aggregation_config = std::make_shared<metrics_sdk::HistogramAggregationConfig>();
  aggregation_config->boundaries_ = {0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
                                     1,     2.5,   5,    10,    30,   60,  120};
  auto view =
      metrics_sdk::ViewFactory::Create("kernellake.query.duration_seconds", "whole-query execution duration",
                                       metrics_sdk::AggregationType::kHistogram, aggregation_config);
  provider.AddView(std::move(instrument_selector), std::move(meter_selector), std::move(view));
}

// Defined once in query_tracing_otel.cpp; read/written by both it and
// query_tracing_test_support.cpp (a test only ever links one binary at a
// time, so there is exactly one definition per process either way).
extern bool g_enabled;
extern std::string g_tracer_name;
extern opentelemetry::nostd::unique_ptr<metrics_api::Histogram<double>> g_query_duration_histogram;

}  // namespace kernellake::observability::detail
