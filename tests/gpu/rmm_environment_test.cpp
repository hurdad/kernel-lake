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
  QueryMemoryTracker tracker = env.make_query_tracker();
  {
    rmm::device_buffer buffer(kBytes, rmm::cuda_stream_view{}, tracker.resource_ref());
    EXPECT_EQ(buffer.size(), kBytes);
    EXPECT_GE(tracker.current_usage().peak_bytes, static_cast<std::int64_t>(kBytes));
  }
}

TEST(RmmEnvironment, TracksBytesAllocatedDuringQuery) {
  EngineConfig config = default_config();
  RmmEnvironment env(config);

  constexpr std::size_t kBytes = 1024 * 1024;  // 1 MiB
  QueryMemoryTracker tracker = env.make_query_tracker();
  {
    rmm::device_buffer buffer(kBytes, rmm::cuda_stream_view{}, tracker.resource_ref());
    EXPECT_EQ(buffer.size(), kBytes);
    EXPECT_GE(tracker.current_usage().peak_bytes, static_cast<std::int64_t>(kBytes));
  }
  // The buffer was freed above (out of scope), so current usage now should
  // have dropped back towards zero.
  EXPECT_LT(tracker.current_usage().current_bytes, static_cast<std::int64_t>(kBytes));
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

// Regression coverage for make_query_tracker()'s own doc comment: unlike
// the old track_query(std::function)'s shared push/pop counter stack (one
// query throwing mid-flight could leave a *different*, later query's
// accounting corrupted -- the exact bug class this design change exists to
// rule out, see RmmEnvironment::make_query_tracker()'s own comment), each
// QueryMemoryTracker is a genuinely separate object. Confirmed here: a
// tracker whose associated work throws mid-flight has no way to affect a
// second, independently-constructed tracker's own counters -- there is no
// shared structure between them at all, not just careful exception
// handling around a shared one.
TEST(RmmEnvironment, AQueryTrackerThrowingDoesNotAffectAnotherTracker) {
  EngineConfig config = default_config();
  RmmEnvironment env(config);

  {
    QueryMemoryTracker failing_tracker = env.make_query_tracker();
    EXPECT_THROW(([&] {
                   rmm::device_buffer buffer(1024, rmm::cuda_stream_view{}, failing_tracker.resource_ref());
                   throw std::runtime_error("simulated query failure");
                 })(),
                 std::runtime_error);
  }

  constexpr std::size_t kBytes = 1024 * 1024;  // 1 MiB
  QueryMemoryTracker tracker = env.make_query_tracker();
  {
    rmm::device_buffer buffer(kBytes, rmm::cuda_stream_view{}, tracker.resource_ref());
    EXPECT_EQ(buffer.size(), kBytes);
    EXPECT_GE(tracker.current_usage().peak_bytes, static_cast<std::int64_t>(kBytes));
  }
  EXPECT_LT(tracker.current_usage().current_bytes, static_cast<std::int64_t>(kBytes));
}

// Regression coverage for the multi-device Tier 1 rewrite
// (docs/MULTI_GPU_SCALING.md): GpuExecutionCoordinator now constructs one
// RmmEnvironment per visible CUDA device, each from its own EngineConfig
// copy with device_id overridden -- query_engine_execute_gpu.cpp reads the
// target device back via this accessor rather than from
// config_.engine.device_id, so it must reflect whatever device_id the
// instance was actually constructed with.
TEST(RmmEnvironment, DeviceIdAccessorReflectsConstructionConfig) {
  EngineConfig config = default_config();
  config.engine.device_id = 0;
  const RmmEnvironment env(config);

  EXPECT_EQ(env.device_id(), 0);
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
