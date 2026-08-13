#include <gtest/gtest.h>

#include <stdexcept>

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

// Regression coverage: default_config() leaves use_async_allocator at its
// own default (true), so every other test in this file only ever
// exercises build_base_resource()'s cuda_async_memory_resource branch --
// the non-async rmm::mr::pool_memory_resource construction path had zero
// coverage before this test.
TEST(RmmEnvironment, ConstructsWithNonAsyncPoolAllocator) {
  EngineConfig config = default_config();
  config.memory.use_async_allocator = false;
  RmmEnvironment env(config);

  constexpr std::size_t kBytes = 1024 * 1024;  // 1 MiB
  const MemoryUsage usage = env.track_query([&] {
    rmm::device_buffer buffer(kBytes, rmm::cuda_stream_view{});
    EXPECT_EQ(buffer.size(), kBytes);
  });
  EXPECT_GE(usage.peak_bytes, static_cast<std::int64_t>(kBytes));
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

// Regression coverage for track_query()'s own doc comment: query() throwing
// must still run pop_counters() (via the try/catch), or the push/pop stack
// goes permanently unbalanced, corrupting peak/current byte accounting for
// the rest of this RmmEnvironment's lifetime. Confirmed here by forcing a
// throw with no real allocation at all (the stack-balance bug wouldn't
// depend on whether real device memory was involved), then proving a
// *subsequent* track_query() call still reports sane byte counts --
// mirroring TracksBytesAllocatedDuringQuery's own assertions -- rather
// than inheriting corrupted state from the failed call before it.
TEST(RmmEnvironment, TrackQueryLeavesCountersBalancedAfterAnException) {
  EngineConfig config = default_config();
  RmmEnvironment env(config);

  EXPECT_THROW((void)(env.track_query([&] { throw std::runtime_error("simulated query failure"); })),
               std::runtime_error);

  constexpr std::size_t kBytes = 1024 * 1024;  // 1 MiB
  const MemoryUsage usage = env.track_query([&] {
    rmm::device_buffer buffer(kBytes, rmm::cuda_stream_view{});
    EXPECT_EQ(buffer.size(), kBytes);
  });
  EXPECT_GE(usage.peak_bytes, static_cast<std::int64_t>(kBytes));
  EXPECT_LT(usage.current_bytes, static_cast<std::int64_t>(kBytes));
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
