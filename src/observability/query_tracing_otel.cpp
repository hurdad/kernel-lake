// Provides the real kernellake::observability implementation for builds
// with KERNELLAKE_ENABLE_OTEL=ON. Mutually exclusive with
// query_tracing_stub.cpp -- see that file's comment.
//
// init_for_testing() and TestLogRecordExporter live in their own
// translation unit, query_tracing_test_support.cpp -- see internal.hpp's
// comment for why (they reference opentelemetry-cpp::in_memory_span_exporter
// /in_memory_metric_exporter, which only tests/unit/CMakeLists.txt links;
// keeping them in this file pulled that whole object file, and its
// then-undefined in-memory-exporter symbols, into the CLI/server binaries
// too).
#include "kernellake/observability/query_tracing.hpp"

#include <opentelemetry/context/context.h>
#include <opentelemetry/context/propagation/text_map_propagator.h>
#include <opentelemetry/context/runtime_context.h>
#include <opentelemetry/exporters/otlp/otlp_grpc_exporter_factory.h>
#include <opentelemetry/exporters/otlp/otlp_grpc_exporter_options.h>
#include <opentelemetry/exporters/otlp/otlp_grpc_log_record_exporter_factory.h>
#include <opentelemetry/exporters/otlp/otlp_grpc_log_record_exporter_options.h>
#include <opentelemetry/exporters/otlp/otlp_grpc_metric_exporter_factory.h>
#include <opentelemetry/exporters/otlp/otlp_grpc_metric_exporter_options.h>
#include <opentelemetry/exporters/otlp/otlp_http_exporter_factory.h>
#include <opentelemetry/exporters/otlp/otlp_http_exporter_options.h>
#include <opentelemetry/exporters/otlp/otlp_http_log_record_exporter_factory.h>
#include <opentelemetry/exporters/otlp/otlp_http_log_record_exporter_options.h>
#include <opentelemetry/exporters/otlp/otlp_http_metric_exporter_factory.h>
#include <opentelemetry/exporters/otlp/otlp_http_metric_exporter_options.h>
#include <opentelemetry/sdk/logs/batch_log_record_processor_factory.h>
#include <opentelemetry/sdk/logs/logger_provider.h>
#include <opentelemetry/sdk/logs/logger_provider_factory.h>
#include <opentelemetry/sdk/logs/simple_log_record_processor_factory.h>
#include <opentelemetry/sdk/metrics/export/periodic_exporting_metric_reader_factory.h>
#include <opentelemetry/sdk/metrics/meter_provider.h>
#include <opentelemetry/sdk/metrics/meter_provider_factory.h>
#include <opentelemetry/sdk/trace/batch_span_processor_factory.h>
#include <opentelemetry/sdk/trace/samplers/always_off_factory.h>
#include <opentelemetry/sdk/trace/samplers/always_on_factory.h>
#include <opentelemetry/sdk/trace/samplers/parent_factory.h>
#include <opentelemetry/sdk/trace/simple_processor_factory.h>
#include <opentelemetry/sdk/trace/tracer_provider.h>
#include <opentelemetry/sdk/trace/tracer_provider_factory.h>
#include <opentelemetry/trace/context.h>
#include <opentelemetry/trace/propagation/http_trace_context.h>
#include <spdlog/spdlog.h>

#include <string>

#include "kernellake/api/query_engine.hpp"
#include "kernellake/common/config.hpp"
#include "internal.hpp"

namespace kernellake::observability {

using namespace detail;  // NOLINT(google-build-using-namespace) -- private impl namespace, one file pair
                         // only.

namespace {

// Only this file touches the OTLP/gRPC exporter option types (test support
// uses in-memory exporters instead), so this alias stays local rather than
// living in the shared internal.hpp.
namespace otlp = opentelemetry::exporter::otlp;

std::unique_ptr<trace_sdk::SpanProcessor> build_span_processor(
    const TraceExportConfig& config, std::unique_ptr<trace_sdk::SpanExporter> exporter) {
  if (config.processor == "simple") {
    return trace_sdk::SimpleSpanProcessorFactory::Create(std::move(exporter));
  }
  trace_sdk::BatchSpanProcessorOptions options;
  options.max_queue_size = config.batch.max_queue_size;
  options.max_export_batch_size = config.batch.max_export_batch_size;
  options.schedule_delay_millis = std::chrono::milliseconds(config.batch.schedule_delay_ms);
  return trace_sdk::BatchSpanProcessorFactory::Create(std::move(exporter), options);
}

std::unique_ptr<logs_sdk::LogRecordProcessor> build_log_processor(
    const LogExportConfig& config, std::unique_ptr<logs_sdk::LogRecordExporter> exporter) {
  if (config.processor == "simple") {
    return logs_sdk::SimpleLogRecordProcessorFactory::Create(std::move(exporter));
  }
  logs_sdk::BatchLogRecordProcessorOptions options;
  options.max_queue_size = config.batch.max_queue_size;
  options.max_export_batch_size = config.batch.max_export_batch_size;
  options.schedule_delay_millis = std::chrono::milliseconds(config.batch.schedule_delay_ms);
  return logs_sdk::BatchLogRecordProcessorFactory::Create(std::move(exporter), options);
}

otlp::OtlpGrpcClientOptions build_grpc_client_options(const ObservabilitySection& config) {
  otlp::OtlpGrpcClientOptions options;
  options.endpoint = config.otlp_endpoint;
  options.use_ssl_credentials = config.use_tls;
  options.ssl_credentials_cacert_path = config.tls_ca_cert_path;
  return options;
}

// The three OTLP/HTTP *ExporterOptions structs (trace/metric/log) aren't
// related by inheritance (unlike the gRPC ones' shared OtlpGrpcClientOptions
// base), but share the same field names -- verified by inspecting each
// installed otlp_http_*_options.h header, not assumed. TLS for HTTP is
// selected by otlp_endpoint's own "http://"/"https://" scheme, not a
// separate use_tls flag (see ObservabilitySection's own comment); client-
// cert mTLS (tls_client_cert_path/tls_client_key_path) *is* available here,
// unlike gRPC.
//
// `options.url` is the exact, full endpoint -- unlike gRPC's single
// endpoint (one port, three services multiplexed on it), OTLP/HTTP is
// plain REST and needs a distinct path per signal (the spec's default
// "general" endpoint convention: <base>/v1/traces, <base>/v1/metrics,
// <base>/v1/logs). It is not auto-appended by the exporter -- confirmed by
// an actual live request against a real collector (Jaeger) returning 404
// for the bare base URL and 200 once "/v1/traces" was appended -- so
// `path_suffix` does that here, letting kernellake's own config expose one
// shared `otlp_endpoint` (the base) rather than three separate per-signal
// endpoint fields.
template <class HttpOptions>
void fill_http_options(HttpOptions& options, const ObservabilitySection& config,
                       std::string_view path_suffix) {
  options.url = config.otlp_endpoint + std::string(path_suffix);
  options.ssl_ca_cert_path = config.tls_ca_cert_path;
  options.ssl_client_cert_path = config.tls_client_cert_path;
  options.ssl_client_key_path = config.tls_client_key_path;
}

std::unique_ptr<trace_sdk::SpanExporter> build_trace_exporter(const ObservabilitySection& config) {
  if (config.otlp_protocol == "http") {
    otlp::OtlpHttpExporterOptions options;
    fill_http_options(options, config, "/v1/traces");
    return otlp::OtlpHttpExporterFactory::Create(options);
  }
  otlp::OtlpGrpcExporterOptions options;
  static_cast<otlp::OtlpGrpcClientOptions&>(options) = build_grpc_client_options(config);
  return otlp::OtlpGrpcExporterFactory::Create(options);
}

std::unique_ptr<metrics_sdk::PushMetricExporter> build_metric_exporter(const ObservabilitySection& config) {
  if (config.otlp_protocol == "http") {
    otlp::OtlpHttpMetricExporterOptions options;
    fill_http_options(options, config, "/v1/metrics");
    return otlp::OtlpHttpMetricExporterFactory::Create(options);
  }
  otlp::OtlpGrpcMetricExporterOptions options;
  static_cast<otlp::OtlpGrpcClientOptions&>(options) = build_grpc_client_options(config);
  return otlp::OtlpGrpcMetricExporterFactory::Create(options);
}

std::unique_ptr<logs_sdk::LogRecordExporter> build_log_exporter(const ObservabilitySection& config) {
  if (config.otlp_protocol == "http") {
    otlp::OtlpHttpLogRecordExporterOptions options;
    fill_http_options(options, config, "/v1/logs");
    return otlp::OtlpHttpLogRecordExporterFactory::Create(options);
  }
  otlp::OtlpGrpcLogRecordExporterOptions options;
  static_cast<otlp::OtlpGrpcClientOptions&>(options) = build_grpc_client_options(config);
  return otlp::OtlpGrpcLogRecordExporterFactory::Create(options);
}

std::unique_ptr<trace_sdk::Sampler> build_sampler(const std::string& sampler_name) {
  if (sampler_name == "always") {
    return trace_sdk::AlwaysOnSamplerFactory::Create();
  }
  if (sampler_name == "never") {
    return trace_sdk::AlwaysOffSamplerFactory::Create();
  }
  // "default": ParentBased(AlwaysOn) -- sample unless an incoming parent
  // context says not to; matches the OTel spec's own recommended default
  // root-sampling behavior.
  return trace_sdk::ParentBasedSamplerFactory::Create(
      std::shared_ptr<trace_sdk::Sampler>(trace_sdk::AlwaysOnSamplerFactory::Create()));
}

}  // namespace

namespace detail {
bool g_enabled = false;
std::string g_tracer_name;
opentelemetry::nostd::unique_ptr<metrics_api::Histogram<double>> g_query_duration_histogram;
}  // namespace detail

void init(const ObservabilitySection& config) {
  if (!config.enabled) {
    return;
  }

  const resource_sdk::Resource resource = build_resource(config.service_name);
  g_tracer_name = config.service_name;

  // Traces.
  {
    auto exporter = build_trace_exporter(config);
    auto processor = build_span_processor(config.tracing, std::move(exporter));
    auto sampler = build_sampler(config.tracing.sampler);
    auto provider =
        trace_sdk::TracerProviderFactory::Create(std::move(processor), resource, std::move(sampler));
    trace_api::Provider::SetTracerProvider(to_provider<trace_api::TracerProvider>(std::move(provider)));
  }

  // Metrics.
  {
    auto exporter = build_metric_exporter(config);

    metrics_sdk::PeriodicExportingMetricReaderOptions reader_options;
    reader_options.export_interval_millis = std::chrono::milliseconds(config.metrics.export_interval_ms);
    reader_options.export_timeout_millis = std::chrono::milliseconds(config.metrics.export_timeout_ms);
    auto reader =
        metrics_sdk::PeriodicExportingMetricReaderFactory::Create(std::move(exporter), reader_options);

    auto provider = metrics_sdk::MeterProviderFactory::Create();
    provider->AddMetricReader(std::shared_ptr<metrics_sdk::MetricReader>(std::move(reader)));
    add_query_duration_histogram_view(*provider, config.service_name);
    metrics_api::Provider::SetMeterProvider(to_provider<metrics_api::MeterProvider>(std::move(provider)));

    auto meter = metrics_api::Provider::GetMeterProvider()->GetMeter(config.service_name);
    g_query_duration_histogram = meter->CreateDoubleHistogram("kernellake.query.duration_seconds",
                                                              "whole-query execution duration", "s");
  }

  // Logs.
  {
    auto exporter = build_log_exporter(config);
    auto processor = build_log_processor(config.logs, std::move(exporter));
    auto provider = logs_sdk::LoggerProviderFactory::Create(std::move(processor), resource);
    logs_api::Provider::SetLoggerProvider(to_provider<logs_api::LoggerProvider>(std::move(provider)));
  }

  // Bridge every existing spdlog call into the logs signal just registered
  // above -- see OtelSpdlogSink's own comment.
  spdlog::default_logger()->sinks().push_back(std::make_shared<OtelSpdlogSink>(config.service_name));

  g_enabled = true;
}

void shutdown() {
  if (!g_enabled) {
    return;
  }

  if (auto* provider =
          dynamic_cast<trace_sdk::TracerProvider*>(trace_api::Provider::GetTracerProvider().get())) {
    provider->ForceFlush();
    provider->Shutdown();
  }
  if (auto* provider =
          dynamic_cast<metrics_sdk::MeterProvider*>(metrics_api::Provider::GetMeterProvider().get())) {
    provider->ForceFlush();
    provider->Shutdown();
  }
  if (auto* provider =
          dynamic_cast<logs_sdk::LoggerProvider*>(logs_api::Provider::GetLoggerProvider().get())) {
    provider->ForceFlush();
    provider->Shutdown();
  }

  g_enabled = false;
}

struct QuerySpan::Impl {
  opentelemetry::nostd::shared_ptr<trace_api::Span> span;
  // Keeps this span "current" (opentelemetry::context::RuntimeContext::
  // GetCurrent()) for as long as the QuerySpan itself is alive, so a
  // ClientSpan started anywhere underneath a query's execution (e.g.
  // DeltaTxnClient's gRPC calls) picks it up as its parent by default --
  // see StartSpanOptions's own doc comment: "If the parent field is not
  // set, the newly created Span will inherit the parent of the currently
  // active Span." Detaches automatically (Token's destructor) whenever
  // this Impl is destroyed, restoring whatever context was active before
  // this query started.
  opentelemetry::nostd::unique_ptr<opentelemetry::context::Token> context_token;
};

QuerySpan::QuerySpan(QuerySpan&&) noexcept = default;
QuerySpan& QuerySpan::operator=(QuerySpan&&) noexcept = default;

QuerySpan::~QuerySpan() {
  if (impl_ && impl_->span && impl_->span->IsRecording()) {
    impl_->span->End();
  }
}

QuerySpan start_query_span(std::string_view operation_name) {
  if (!g_enabled) {
    return QuerySpan();
  }

  QuerySpan span;
  span.impl_ = std::make_unique<QuerySpan::Impl>();
  span.impl_->span =
      trace_api::Provider::GetTracerProvider()->GetTracer(g_tracer_name)->StartSpan(to_otel(operation_name));
  opentelemetry::context::Context ctx = opentelemetry::context::RuntimeContext::GetCurrent();
  ctx = trace_api::SetSpan(ctx, span.impl_->span);
  span.impl_->context_token = opentelemetry::context::RuntimeContext::Attach(ctx);
  return span;
}

namespace {

// SQL text can be arbitrarily long; cap the span attribute so one enormous
// generated query doesn't balloon export payload size.
constexpr std::size_t kMaxSqlAttributeLength = 4096;

void set_common_attributes(trace_api::Span& span, std::string_view sql, std::string_view backend) {
  span.SetAttribute("kernellake.backend", to_otel(backend));
  span.SetAttribute("kernellake.sql", to_otel(sql.substr(0, kMaxSqlAttributeLength)));
}

}  // namespace

void QuerySpan::finish(const QueryResult& result, std::string_view sql, std::string_view backend) {
  if (!impl_) {
    return;
  }
  trace_api::Span& span = *impl_->span;

  set_common_attributes(span, sql, backend);
  if (result.rows_returned) {
    span.SetAttribute("kernellake.rows_returned", *result.rows_returned);
  }
  if (result.rows_scanned) {
    span.SetAttribute("kernellake.rows_scanned", *result.rows_scanned);
  }
  if (result.files_considered) {
    span.SetAttribute("kernellake.files_considered", *result.files_considered);
  }
  if (result.files_scanned) {
    span.SetAttribute("kernellake.files_scanned", *result.files_scanned);
  }
  if (result.row_groups_considered) {
    span.SetAttribute("kernellake.row_groups_considered", *result.row_groups_considered);
  }
  if (result.row_groups_scanned) {
    span.SetAttribute("kernellake.row_groups_scanned", *result.row_groups_scanned);
  }
  if (result.compressed_bytes_read) {
    span.SetAttribute("kernellake.compressed_bytes_read", *result.compressed_bytes_read);
  }
  if (result.parquet_decoding_seconds) {
    span.SetAttribute("kernellake.parquet_decoding_seconds", *result.parquet_decoding_seconds);
  }
  if (result.gpu_execution_seconds) {
    span.SetAttribute("kernellake.gpu_execution_seconds", *result.gpu_execution_seconds);
  }
  if (result.cpu_execution_seconds) {
    span.SetAttribute("kernellake.cpu_execution_seconds", *result.cpu_execution_seconds);
  }
  if (result.host_to_device_seconds) {
    span.SetAttribute("kernellake.host_to_device_seconds", *result.host_to_device_seconds);
  }
  if (result.device_to_host_seconds) {
    span.SetAttribute("kernellake.device_to_host_seconds", *result.device_to_host_seconds);
  }
  if (result.peak_gpu_memory_bytes) {
    span.SetAttribute("kernellake.peak_gpu_memory_bytes", *result.peak_gpu_memory_bytes);
  }
  if (result.elapsed_wall_seconds) {
    span.SetAttribute("kernellake.elapsed_wall_seconds", *result.elapsed_wall_seconds);
  }

  span.SetStatus(trace_api::StatusCode::kOk);
  span.End();

  if (result.elapsed_wall_seconds && g_query_duration_histogram) {
    // This ABI version has no 2-arg Record(value, attributes) convenience
    // overload (only gated in behind OPENTELEMETRY_ABI_VERSION_NO >= 2,
    // which this build isn't using -- confirmed by an actual build error
    // listing only the 3-arg Context-taking candidates) -- pass an empty
    // Context explicitly.
    g_query_duration_histogram->Record(*result.elapsed_wall_seconds,
                                       {{"kernellake.backend", std::string(backend)}},
                                       opentelemetry::context::Context{});
  }
}

void QuerySpan::finish_error(const std::exception& e, std::string_view sql, std::string_view backend) {
  if (!impl_) {
    return;
  }
  trace_api::Span& span = *impl_->span;

  set_common_attributes(span, sql, backend);
  span.AddEvent("exception", {{"exception.message", std::string(e.what())}});
  span.SetStatus(trace_api::StatusCode::kError, e.what());
  span.End();
}

struct ClientSpan::Impl {
  opentelemetry::nostd::shared_ptr<trace_api::Span> span;
  // Same purpose as QuerySpan::Impl::context_token: keeps this span
  // "current" for as long as this ClientSpan is alive, so any further
  // nested ClientSpan (there are none today, but the whole point of using
  // RuntimeContext rather than a bespoke parent-tracking mechanism is that
  // this generalizes for free) parents correctly too.
  opentelemetry::nostd::unique_ptr<opentelemetry::context::Token> context_token;
};

ClientSpan::ClientSpan(ClientSpan&&) noexcept = default;
ClientSpan& ClientSpan::operator=(ClientSpan&&) noexcept = default;

ClientSpan::~ClientSpan() {
  if (impl_ && impl_->span && impl_->span->IsRecording()) {
    impl_->span->End();
  }
}

ClientSpan start_client_span(std::string_view operation_name) {
  if (!g_enabled) {
    return ClientSpan();
  }

  ClientSpan span;
  span.impl_ = std::make_unique<ClientSpan::Impl>();
  span.impl_->span =
      trace_api::Provider::GetTracerProvider()->GetTracer(g_tracer_name)->StartSpan(to_otel(operation_name));
  opentelemetry::context::Context ctx = opentelemetry::context::RuntimeContext::GetCurrent();
  ctx = trace_api::SetSpan(ctx, span.impl_->span);
  span.impl_->context_token = opentelemetry::context::RuntimeContext::Attach(ctx);
  return span;
}

ClientSpan start_client_span(std::string_view operation_name, const ClientSpan& parent) {
  if (!g_enabled || !parent.impl_) {
    return ClientSpan();
  }

  // Explicit parent via StartSpanOptions, not RuntimeContext::Attach --
  // this deliberately does *not* touch the thread-local "current span"
  // stack (see ExecutionContext::current_span's own comment for why
  // InstrumentedOperator needs that: correct nesting for an operator with
  // more than one child opened sequentially requires each child to name
  // its exact parent rather than inherit whatever's currently attached,
  // and background-thread work, e.g. ParquetScanOperator's decode thread,
  // never automatically picks up thread-local context from the thread that
  // spawned it anyway).
  trace_api::StartSpanOptions options;
  options.parent = parent.impl_->span->GetContext();

  ClientSpan span;
  span.impl_ = std::make_unique<ClientSpan::Impl>();
  span.impl_->span = trace_api::Provider::GetTracerProvider()
                         ->GetTracer(g_tracer_name)
                         ->StartSpan(to_otel(operation_name), options);
  return span;
}

void ClientSpan::set_attribute(std::string_view key, double value) {
  if (!impl_) {
    return;
  }
  impl_->span->SetAttribute(to_otel(key), value);
}

void ClientSpan::set_attribute(std::string_view key, std::int64_t value) {
  if (!impl_) {
    return;
  }
  impl_->span->SetAttribute(to_otel(key), value);
}

namespace {

// Adapts ClientSpan::inject()'s std::function setter to OTel's
// TextMapCarrier interface (what HttpTraceContext::Inject() needs) --
// Get()/Keys() are never actually called by an *injecting* propagator
// (both only matter for Extract(), delta-txn-service's own side of this,
// see that project's telemetry/trace_context.rs), so they're trivial
// stubs rather than real accessors into anything.
class SetterTextMapCarrier : public opentelemetry::context::propagation::TextMapCarrier {
 public:
  explicit SetterTextMapCarrier(const std::function<void(std::string_view, std::string_view)>& setter)
      : setter_(setter) {}

  opentelemetry::nostd::string_view Get(opentelemetry::nostd::string_view /*key*/) const noexcept override {
    return "";
  }

  void Set(opentelemetry::nostd::string_view key, opentelemetry::nostd::string_view value) noexcept override {
    setter_(std::string_view(key.data(), key.size()), std::string_view(value.data(), value.size()));
  }

 private:
  const std::function<void(std::string_view, std::string_view)>& setter_;
};

}  // namespace

void ClientSpan::inject(const std::function<void(std::string_view, std::string_view)>& setter) const {
  if (!impl_) {
    return;
  }
  SetterTextMapCarrier carrier(setter);
  opentelemetry::context::Context ctx;
  ctx = trace_api::SetSpan(ctx, impl_->span);
  trace_api::propagation::HttpTraceContext{}.Inject(carrier, ctx);
}

void ClientSpan::finish_ok() {
  if (!impl_) {
    return;
  }
  impl_->span->SetStatus(trace_api::StatusCode::kOk);
  impl_->span->End();
}

void ClientSpan::finish_error(std::string_view error_message) {
  if (!impl_) {
    return;
  }
  impl_->span->AddEvent("exception", {{"exception.message", std::string(error_message)}});
  impl_->span->SetStatus(trace_api::StatusCode::kError, to_otel(error_message));
  impl_->span->End();
}

}  // namespace kernellake::observability
