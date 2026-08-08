// Provides a no-op register_gpu_memory_otel_instruments() for
// KERNELLAKE_ENABLE_OTEL=OFF builds (the default). Mutually exclusive with
// gpu_memory_metrics_otel.cpp -- see src/memory/CMakeLists.txt. Deliberately
// includes no opentelemetry-cpp header, matching
// query_tracing_stub.cpp's own reasoning.
#include "kernellake/memory/gpu_memory_otel.hpp"

namespace kernellake {

void register_gpu_memory_otel_instruments() {}

void reset_gpu_memory_otel_instruments_for_testing() {}

}  // namespace kernellake
