// Provides a no-op register_nvme_cache_otel_instruments() for
// KERNELLAKE_ENABLE_OTEL=OFF builds (the default). Mutually exclusive with
// nvme_cache_metrics_otel.cpp -- see src/storage/CMakeLists.txt.
#include "kernellake/storage/nvme_cache_otel.hpp"

namespace kernellake {

void register_nvme_cache_otel_instruments(const ObjectStoreRegistry&) {}

}  // namespace kernellake
