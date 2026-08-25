// Real-OTel-SDK coverage for the kernellake.gpu.memory.* instruments (see
// gpu_memory_metrics_otel.cpp): proves the ObservableGauge/ObservableCounter
// callbacks actually report GpuMemoryMetricsRegistry's real values through
// opentelemetry-cpp's own metrics SDK, not just that the registry's own
// bookkeeping is correct (gpu_memory_metrics_test.cpp already covers that
// without any OTel dependency). Mirrors tests/unit/query_tracing_test.cpp's
// own in-memory-exporter pattern.
#include <gtest/gtest.h>

#include <opentelemetry/sdk/metrics/data/point_data.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include "kernellake/common/config.hpp"
#include "kernellake/memory/gpu_memory_metrics.hpp"
#include "kernellake/memory/gpu_memory_otel.hpp"
#include "kernellake/memory/rmm_environment.hpp"
#include "kernellake/observability/query_tracing.hpp"
#include "kernellake/observability/query_tracing_test_support.hpp"

namespace kernellake {
namespace {

using AttributeToPoint = opentelemetry::exporter::memory::SimpleAggregateInMemoryMetricData::AttributeToPoint;

// Finds the one entry in `points` tagged gpu.device.id == device_id --
// AttributeToPoint holds one entry per distinct attribute set, and every
// GPU test in this binary that has ever run in this process (all sharing
// device_id 0, the only real device this machine has) contributes its own
// device_id's entry, so a plain "assume one entry" lookup (as
// query_tracing_test.cpp's own histogram check does, for an instrument
// with no per-device attribute at all) isn't safe here.
template <class PointType>
const PointType& find_point_for_device(const AttributeToPoint& points, int device_id) {
  for (const auto& [attributes, point] : points) {
    const auto it = attributes.find("gpu.device.id");
    if (it != attributes.end() && opentelemetry::nostd::get<std::int64_t>(it->second) == device_id) {
      return opentelemetry::nostd::get<PointType>(point);
    }
  }
  ADD_FAILURE() << "no exported point found for gpu.device.id=" << device_id;
  static const PointType kEmpty{};
  return kEmpty;
}

class GpuMemoryMetricsOtelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    observability::init_for_testing("kernellake-gpu-memory-otel-test", spans_, metrics_, log_exporter_);
    // register_gpu_memory_otel_instruments() may already have run (e.g. a
    // GPU test that ran earlier in this binary constructed an
    // RmmEnvironment before this test's SetUp), binding these instruments
    // to whatever MeterProvider was current *then* -- reset and re-register
    // now, against the in-memory provider init_for_testing() just installed
    // above. See gpu_memory_otel.hpp's own comment on this seam.
    reset_gpu_memory_otel_instruments_for_testing();
    register_gpu_memory_otel_instruments();
  }

  void TearDown() override { observability::shutdown(); }

  std::shared_ptr<opentelemetry::exporter::memory::InMemorySpanData> spans_;
  std::shared_ptr<opentelemetry::exporter::memory::SimpleAggregateInMemoryMetricData> metrics_;
  observability::TestLogRecordExporter* log_exporter_ = nullptr;
};

TEST_F(GpuMemoryMetricsOtelTest, ObservableGaugeAndCounterCallbacksReportRealRegistryValues) {
  EngineConfig config = default_config();
  const int device_id = 0;

  const GpuMemoryMetricsSnapshot before = GpuMemoryMetricsRegistry::snapshot(device_id);

  {
    RmmEnvironment env(config);
    rmm::device_buffer buffer(1024 * 1024,
                              rmm::cuda_stream_view{});  // 1 MiB, kept alive past the flush below.

    observability::shutdown();  // ForceFlush()es synchronously -- see query_tracing_test.cpp's own comment.

    const auto& allocated_points = metrics_->Get("kernellake.gpu", "kernellake.gpu.memory.allocated");
    const auto& allocated =
        find_point_for_device<opentelemetry::sdk::metrics::LastValuePointData>(allocated_points, device_id);
    EXPECT_GE(opentelemetry::nostd::get<std::int64_t>(allocated.value_),
              static_cast<std::int64_t>(before.current_bytes) + 1024 * 1024);

    const auto& allocations_points = metrics_->Get("kernellake.gpu", "kernellake.gpu.memory.allocations");
    const auto& allocations =
        find_point_for_device<opentelemetry::sdk::metrics::SumPointData>(allocations_points, device_id);
    EXPECT_GT(opentelemetry::nostd::get<std::int64_t>(allocations.value_),
              static_cast<std::int64_t>(before.allocations));

    const auto& total_points = metrics_->Get("kernellake.gpu", "kernellake.gpu.memory.allocated_total");
    const auto& total =
        find_point_for_device<opentelemetry::sdk::metrics::SumPointData>(total_points, device_id);
    EXPECT_GE(opentelemetry::nostd::get<std::int64_t>(total.value_),
              static_cast<std::int64_t>(before.allocated_total_bytes) + 1024 * 1024);
  }
}

}  // namespace
}  // namespace kernellake
