#pragma once

#include <exception>
#include <functional>
#include <memory>
#include <string_view>

namespace kernellake {

struct ObservabilitySection;
struct QueryResult;

namespace observability {

// Initializes the process-wide OTel TracerProvider/MeterProvider/
// LoggerProvider from `config`, registers them as the global providers, and
// attaches a bridging sink to spdlog's default logger so every existing
// spdlog::info/warn/error call in this codebase is also emitted as an OTel
// log record -- no existing call site needs to change. Call once, near the
// top of main(), right after init_logging(config.logging) (so the bridge
// sink is attached to an already-configured default logger, not replacing
// its own level/pattern setup). No-op -- touches no network, constructs no
// OTel object, attaches no sink -- when config.enabled is false, or when
// this binary was built with KERNELLAKE_ENABLE_OTEL=OFF (the stub logs one
// spdlog::warn if config.enabled was true, so the operator sees why).
void init(const ObservabilitySection& config);

// Flushes and shuts down whatever init() registered (tracer, meter, logger
// providers), forcing buffered spans/metrics/logs to export (or drop after
// a bounded timeout) before process exit. Safe to call even if init() was
// never called or was a no-op.
void shutdown();

// RAII handle for one whole-query span. Exactly one of finish()/
// finish_error() should be called before destruction; the destructor ends
// the span unfinished (status Unset) as a safety net if neither was.
class QuerySpan {
 public:
  QuerySpan(QuerySpan&&) noexcept;
  QuerySpan& operator=(QuerySpan&&) noexcept;
  QuerySpan(const QuerySpan&) = delete;
  QuerySpan& operator=(const QuerySpan&) = delete;
  ~QuerySpan();

  // Populates span attributes from result's *measured* fields only (each
  // std::optional that is actually set -- QueryResult's own "documented
  // null value, never an invented measurement" rule extends here), sets
  // status Ok, ends the span, and records one
  // kernellake.query.duration_seconds histogram observation from
  // result.elapsed_wall_seconds when present.
  void finish(const QueryResult& result, std::string_view sql, std::string_view backend);

  // Records `e` on the span as an exception event, sets status Error with
  // e.what() as the description, and ends the span. No histogram
  // observation (there is no elapsed_wall_seconds to record).
  void finish_error(const std::exception& e, std::string_view sql, std::string_view backend);

 private:
  friend QuerySpan start_query_span(std::string_view);
  QuerySpan() = default;
  struct Impl;
  std::unique_ptr<Impl> impl_;  // null when disabled or not built with otel
};

// Starts a whole-query span named `operation_name` (e.g. "kernellake.query"
// for the CLI, "kernellake.flight_sql.get_flight_info_statement" for the
// server). Always returns a valid QuerySpan, even when tracing is
// disabled/not built -- finish()/finish_error() are then no-ops, so call
// sites never need an `if (otel_enabled)` branch.
[[nodiscard]] QuerySpan start_query_span(std::string_view operation_name);

// RAII handle for one outbound-call span (e.g. one gRPC call to
// delta-txn-service) -- a *child* span, not a whole-query span like
// QuerySpan. Unlike QuerySpan, this exposes inject(): encoding this span's
// own context as W3C trace-context headers so the receiving service can
// link its own spans as this one's children, completing the distributed
// trace across the process boundary rather than leaving it as two
// disconnected trees. Exactly one of finish_ok()/finish_error() should be
// called before destruction; the destructor ends the span unfinished
// (status Unset) as a safety net if neither was.
class ClientSpan {
 public:
  ClientSpan(ClientSpan&&) noexcept;
  ClientSpan& operator=(ClientSpan&&) noexcept;
  ClientSpan(const ClientSpan&) = delete;
  ClientSpan& operator=(const ClientSpan&) = delete;
  ~ClientSpan();

  // Calls `setter(key, value)` once per W3C trace-context header this
  // span's context encodes as -- currently just "traceparent" ("tracestate"
  // only when the propagator actually has one to send). No-op (setter never
  // called) when disabled/not built, so callers never need an
  // `if (otel_enabled)` branch here either -- e.g.
  //   span.inject([&](std::string_view k, std::string_view v) {
  //     context.AddMetadata(std::string(k), std::string(v));
  //   });
  void inject(const std::function<void(std::string_view, std::string_view)>& setter) const;

  // Sets status Ok and ends the span.
  void finish_ok();

  // Records `error_message` on the span as an exception event, sets status
  // Error with it as the description, and ends the span.
  void finish_error(std::string_view error_message);

 private:
  friend ClientSpan start_client_span(std::string_view);
  ClientSpan() = default;
  struct Impl;
  std::unique_ptr<Impl> impl_;  // null when disabled or not built with otel
};

// Starts a child span named `operation_name` (e.g.
// "delta_txn.ListActiveFiles"), parented to whatever span is currently
// active (see start_query_span() -- the whole-query span, typically).
// Always returns a valid ClientSpan, even when tracing is disabled/not
// built.
[[nodiscard]] ClientSpan start_client_span(std::string_view operation_name);

}  // namespace observability
}  // namespace kernellake
