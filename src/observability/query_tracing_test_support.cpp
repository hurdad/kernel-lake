// Test-only seam -- see query_tracing_test_support.hpp. Lives in its own
// translation unit, separate from query_tracing_otel.cpp -- see
// internal.hpp's comment for why (references
// opentelemetry-cpp::in_memory_span_exporter/in_memory_metric_exporter,
// which only tests/unit/CMakeLists.txt links; a static archive pulls in
// whichever .o member actually resolves a referenced symbol, so keeping
// this in a separate .o from init() keeps the CLI/server -- which only
// reference init() -- from ever needing those test-only libraries at all).
#include "kernellake/observability/query_tracing_test_support.hpp"

#include <opentelemetry/exporters/memory/in_memory_metric_exporter_factory.h>
#include <opentelemetry/exporters/memory/in_memory_span_exporter_factory.h>
#include <opentelemetry/sdk/logs/logger_provider_factory.h>
#include <opentelemetry/sdk/logs/read_write_log_record.h>
#include <opentelemetry/sdk/logs/simple_log_record_processor_factory.h>
#include <opentelemetry/sdk/metrics/export/periodic_exporting_metric_reader_factory.h>
#include <opentelemetry/sdk/metrics/meter_provider.h>
#include <opentelemetry/sdk/metrics/meter_provider_factory.h>
#include <opentelemetry/sdk/trace/simple_processor_factory.h>
#include <opentelemetry/sdk/trace/tracer_provider_factory.h>
#include <spdlog/spdlog.h>

#include <string>

#include "internal.hpp"

namespace kernellake::observability {

using namespace detail;  // NOLINT(google-build-using-namespace) -- private impl namespace, one file pair only.

std::unique_ptr<logs_sdk::Recordable> TestLogRecordExporter::MakeRecordable() noexcept {
  return std::make_unique<logs_sdk::ReadWriteLogRecord>();
}

opentelemetry::sdk::common::ExportResult TestLogRecordExporter::Export(
    const opentelemetry::nostd::span<std::unique_ptr<logs_sdk::Recordable>>& new_records) noexcept {
  for (auto& record : new_records) records.push_back(std::move(record));
  return opentelemetry::sdk::common::ExportResult::kSuccess;
}

bool TestLogRecordExporter::ForceFlush(std::chrono::microseconds /*timeout*/) noexcept { return true; }

bool TestLogRecordExporter::Shutdown(std::chrono::microseconds /*timeout*/) noexcept { return true; }

// Mirrors init() (query_tracing_otel.cpp) but wires in-memory/test
// exporters instead of the real OTLP/gRPC ones.
void init_for_testing(const std::string& service_name,
                      std::shared_ptr<opentelemetry::exporter::memory::InMemorySpanData>& spans,
                      std::shared_ptr<opentelemetry::exporter::memory::SimpleAggregateInMemoryMetricData>& metrics,
                      TestLogRecordExporter*& log_exporter) {
  const resource_sdk::Resource resource = build_resource(service_name);
  g_tracer_name = service_name;

  // InMemorySpanExporterFactory::Create takes `spans` as an out-parameter --
  // it constructs the InMemorySpanData itself and assigns it here, unlike
  // the metric exporter below (which takes an already-constructed
  // InMemoryMetricData as input).
  auto span_exporter = opentelemetry::exporter::memory::InMemorySpanExporterFactory::Create(spans);
  auto span_processor = trace_sdk::SimpleSpanProcessorFactory::Create(std::move(span_exporter));
  auto tracer_provider = trace_sdk::TracerProviderFactory::Create(std::move(span_processor), resource);
  trace_api::Provider::SetTracerProvider(to_provider<trace_api::TracerProvider>(std::move(tracer_provider)));

  metrics = std::make_shared<opentelemetry::exporter::memory::SimpleAggregateInMemoryMetricData>();
  auto metric_exporter = opentelemetry::exporter::memory::InMemoryMetricExporterFactory::Create(metrics);
  metrics_sdk::PeriodicExportingMetricReaderOptions reader_options;
  reader_options.export_interval_millis = std::chrono::milliseconds(1000);
  reader_options.export_timeout_millis = std::chrono::milliseconds(500);
  auto reader = metrics_sdk::PeriodicExportingMetricReaderFactory::Create(std::move(metric_exporter), reader_options);
  auto meter_provider = metrics_sdk::MeterProviderFactory::Create();
  meter_provider->AddMetricReader(std::shared_ptr<metrics_sdk::MetricReader>(std::move(reader)));
  metrics_api::Provider::SetMeterProvider(to_provider<metrics_api::MeterProvider>(std::move(meter_provider)));
  auto meter = metrics_api::Provider::GetMeterProvider()->GetMeter(service_name);
  g_query_duration_histogram =
      meter->CreateDoubleHistogram("kernellake.query.duration_seconds", "whole-query execution duration", "s");

  auto owned_log_exporter = std::make_unique<TestLogRecordExporter>();
  log_exporter = owned_log_exporter.get();
  auto log_processor = logs_sdk::SimpleLogRecordProcessorFactory::Create(std::move(owned_log_exporter));
  auto logger_provider = logs_sdk::LoggerProviderFactory::Create(std::move(log_processor), resource);
  logs_api::Provider::SetLoggerProvider(to_provider<logs_api::LoggerProvider>(std::move(logger_provider)));

  spdlog::default_logger()->sinks().push_back(std::make_shared<OtelSpdlogSink>(service_name));

  g_enabled = true;
}

}  // namespace kernellake::observability
