#include <gtest/gtest.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include "kernellake/common/config.hpp"
#include "kernellake/memory/rmm_environment.hpp"

namespace kernellake {
namespace {

TEST(RmmEnvironment, ConstructsAndInstallsCurrentDeviceResource) {
  EngineConfig config = default_config();
  EXPECT_NO_THROW((void)({ RmmEnvironment env(config); }));
}

TEST(RmmEnvironment, TracksBytesAllocatedDuringQuery) {
  EngineConfig config = default_config();
  RmmEnvironment env(config);

  constexpr std::size_t kBytes = 1024 * 1024;  // 1 MiB
  const MemoryUsage usage = env.track_query([&] {
    rmm::device_buffer buffer(kBytes, rmm::cuda_stream_view{});
    EXPECT_EQ(buffer.size(), kBytes);
  });

  EXPECT_GE(usage.peak_bytes, static_cast<std::int64_t>(kBytes));
  // The buffer was freed before track_query returned, so current usage
  // within that bracket should have dropped back towards zero.
  EXPECT_LT(usage.current_bytes, static_cast<std::int64_t>(kBytes));
}

TEST(RmmEnvironment, RespectsConfiguredQueryMemoryLimit) {
  EngineConfig config = default_config();
  config.engine.query_memory_limit_bytes = 1024;  // Deliberately tiny.
  RmmEnvironment env(config);

  EXPECT_THROW((void)({
                 rmm::device_buffer buffer(64 * 1024 * 1024, rmm::cuda_stream_view{});  // 64 MiB > limit
               }),
               std::exception);
}

TEST(RmmEnvironment, QueryMemoryLimitBytesAccessorReflectsExplicitConfig) {
  EngineConfig config = default_config();
  config.engine.query_memory_limit_bytes = 12345;
  const RmmEnvironment env(config);

  // Not just "not zero" -- the accessor must report the exact value this
  // instance's limiting_resource_adaptor was actually built with, since
  // query_engine_execute_gpu.cpp's pass_read_limit_bytes sizing depends on
  // that exact agreement (see this accessor's own doc comment on why a
  // fresh resolve_query_memory_limit_bytes() call instead would silently
  // drift for a long-lived instance).
  EXPECT_EQ(env.query_memory_limit_bytes(), 12345u);
}

TEST(RmmEnvironment, QueryMemoryLimitBytesAccessorAutoDetectsWhenConfigIsZero) {
  EngineConfig config = default_config();
  config.engine.query_memory_limit_bytes = 0;  // default_config()'s own default; explicit here for clarity.
  const RmmEnvironment env(config);

  // A real value resolved from actual GPU VRAM, not the sentinel itself.
  EXPECT_GT(env.query_memory_limit_bytes(), 0u);
}

}  // namespace
}  // namespace kernellake
