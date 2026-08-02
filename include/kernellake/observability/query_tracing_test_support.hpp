#pragma once
// Test-only seam for injecting opentelemetry-cpp's in-memory span/metric
// exporters and a small custom in-memory log exporter in place of the real
// OTLP/gRPC ones. Included only by tests/unit/query_tracing_test.cpp (a
// KERNELLAKE_ENABLE_OTEL-only test file) and by query_tracing_otel.cpp
// itself (which implements init_for_testing()). Deliberately kept out of
// query_tracing.hpp so production call sites never see an
// opentelemetry-cpp type, even in an ENABLE_OTEL build.
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <opentelemetry/exporters/memory/in_memory_metric_data.h>
#include <opentelemetry/exporters/memory/in_memory_span_data.h>
#include <opentelemetry/nostd/span.h>
#include <opentelemetry/sdk/common/exporter_utils.h>
#include <opentelemetry/sdk/logs/exporter.h>
#include <opentelemetry/sdk/logs/recordable.h>

namespace kernellake::observability {

// A minimal LogRecordExporter that just keeps exported records in memory --
// no in-memory log exporter ships in opentelemetry-cpp-dev 1.23.0, unlike
// spans/metrics.
class TestLogRecordExporter : public opentelemetry::sdk::logs::LogRecordExporter {
 public:
  std::unique_ptr<opentelemetry::sdk::logs::Recordable> MakeRecordable() noexcept override;
  opentelemetry::sdk::common::ExportResult Export(
      const opentelemetry::nostd::span<std::unique_ptr<opentelemetry::sdk::logs::Recordable>>& records) noexcept
      override;
  bool ForceFlush(std::chrono::microseconds timeout) noexcept override;
  bool Shutdown(std::chrono::microseconds timeout) noexcept override;

  std::vector<std::unique_ptr<opentelemetry::sdk::logs::Recordable>> records;
};

// Same Resource/provider construction as init(), but wired to
// InMemorySpanExporter/InMemoryMetricExporter (SimpleSpanProcessor instead
// of init()'s BatchSpanProcessor, so spans are visible synchronously on
// Span::End()) and a TestLogRecordExporter, instead of the real OTLP/gRPC
// ones. `metrics` uses SimpleAggregateInMemoryMetricData. `log_exporter` is
// a non-owning pointer into the LoggerProvider's own processor chain, valid
// for the process lifetime (test-only, never torn down mid-test).
void init_for_testing(const std::string& service_name,
                      std::shared_ptr<opentelemetry::exporter::memory::InMemorySpanData>& spans,
                      std::shared_ptr<opentelemetry::exporter::memory::SimpleAggregateInMemoryMetricData>& metrics,
                      TestLogRecordExporter*& log_exporter);

}  // namespace kernellake::observability
