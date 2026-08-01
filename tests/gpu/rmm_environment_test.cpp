#include <gtest/gtest.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include "kernellake/common/config.hpp"
#include "kernellake/memory/rmm_environment.hpp"

namespace kernellake {
namespace {

TEST(RmmEnvironment, ConstructsAndInstallsCurrentDeviceResource) {
  EngineConfig config = default_config();
  EXPECT_NO_THROW({ RmmEnvironment env(config); });
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

  EXPECT_THROW(
      {
        rmm::device_buffer buffer(64 * 1024 * 1024, rmm::cuda_stream_view{});  // 64 MiB > limit
      },
      std::exception);
}

}  // namespace
}  // namespace kernellake
