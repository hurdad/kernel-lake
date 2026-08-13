// Coverage for the KERNELLAKE_ENABLE_OTEL=OFF (default) build's no-op
// observability implementation (query_tracing_stub.cpp), never exercised by
// any test before this file -- query_tracing_test.cpp only covers the
// KERNELLAKE_ENABLE_OTEL=ON variant (query_tracing_otel.cpp), and is itself
// only compiled into this suite when that flag is on (see
// tests/unit/CMakeLists.txt). This file is the mirror image, gated to only
// build under the stub (KERNELLAKE_ENABLE_OTEL unset/OFF).
#include <gtest/gtest.h>

#include <spdlog/spdlog.h>

#include "kernellake/api/query_engine.hpp"
#include "kernellake/common/config.hpp"
#include "kernellake/common/errors.hpp"
#include "kernellake/observability/query_tracing.hpp"

namespace kernellake {
namespace {

TEST(QueryTracingStub, InitWithObservabilityDisabledIsANoOp) {
  ObservabilitySection config;
  config.enabled = false;
  EXPECT_NO_THROW((void)(observability::init(config)));
  EXPECT_NO_THROW((void)(observability::shutdown()));
}

// The one real behavior this build's init() has: telling the operator
// their config.enabled = true request silently can't be honored, instead of
// either quietly doing nothing (surprising) or throwing (this isn't a
// configuration error -- the rest of KernelLake works fine without OTel).
TEST(QueryTracingStub, InitWithObservabilityEnabledLogsAWarningInsteadOfSilentlyNoOp) {
  // Independent of whatever level some earlier test in this same binary
  // left spdlog's global level at (e.g. logging_test.cpp's RejectsLevel*
  // cases never restore a default) -- spdlog::warn must actually be
  // visible for CaptureStdout() below to see it.
  spdlog::set_level(spdlog::level::info);

  ObservabilitySection config;
  config.enabled = true;

  testing::internal::CaptureStdout();
  EXPECT_NO_THROW((void)(observability::init(config)));
  const std::string output = testing::internal::GetCapturedStdout();

  EXPECT_NE(output.find("KERNELLAKE_ENABLE_OTEL"), std::string::npos) << output;
  EXPECT_NO_THROW((void)(observability::shutdown()));
}

TEST(QueryTracingStub, QuerySpanLifecycleIsANoOpAndDoesNotCrash) {
  QueryResult result;
  result.rows_returned = 7;

  observability::QuerySpan span = observability::start_query_span("kernellake.query");
  EXPECT_NO_THROW(span.finish(result, "SELECT 1", "cpu"));

  observability::QuerySpan error_span = observability::start_query_span("kernellake.query");
  EXPECT_NO_THROW(error_span.finish_error(ExecutionError("boom"), "SELECT 1", "cpu"));
}

TEST(QueryTracingStub, ClientSpanInjectNeverCallsItsSetter) {
  observability::ClientSpan span = observability::start_client_span("kernellake.client");
  bool setter_called = false;
  span.inject([&](std::string_view, std::string_view) { setter_called = true; });
  EXPECT_FALSE(setter_called);
  EXPECT_NO_THROW(span.finish_ok());
}

}  // namespace
}  // namespace kernellake
