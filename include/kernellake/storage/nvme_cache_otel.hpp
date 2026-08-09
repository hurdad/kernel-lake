#pragma once

namespace kernellake {

class ObjectStoreRegistry;

// Registers `registry`'s NVMe cache metrics (see NvmeObjectCache's own
// comment) as OTel instruments under "kernellake.storage.cache.*" --
// mirrors src/memory/gpu_memory_metrics_otel.cpp's split, selected by
// KERNELLAKE_ENABLE_OTEL alone (unlike that module, this needs no CUDA
// gate, since neither ObjectStoreRegistry nor NvmeObjectCache have a CUDA
// dependency). A no-op (nvme_cache_metrics_otel_stub.cpp) when
// KERNELLAKE_ENABLE_OTEL is off, and also a no-op at runtime if
// `registry`'s cache is disabled (nothing to export).
//
// Unlike gpu_memory_metrics_otel.cpp's registry-wide, instance-agnostic
// design (needed because RmmEnvironment is recreated per query in the CLI
// path), this ties instruments directly to one specific `registry`
// instance via the OTel callback's own `void*` user-data parameter --
// `registry` must outlive every future metrics export, which holds for
// kernellake-server's own long-lived QueryEngine but not for a short-lived
// per-request object. Call at most once per `registry` instance -- see
// QueryEngine::register_cache_otel_instruments()'s own comment for the one
// real call site (kernellake-server's constructor).
void register_nvme_cache_otel_instruments(const ObjectStoreRegistry& registry);

}  // namespace kernellake
