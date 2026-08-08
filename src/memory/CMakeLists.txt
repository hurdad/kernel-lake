# Only built when KERNELLAKE_WITH_CUDA is ON (see src/CMakeLists.txt).
#
# gpu_memory_metrics.cpp/gpu_memory_tracking_resource.hpp are CUDA/RMM-
# coupled (unlike GpuMemoryMetricsRegistry's own plain-atomics design,
# which has no such dependency in principle -- see that header's comment)
# only because this is where RmmEnvironment, the sole thing that
# constructs a TrackingMemoryResource, already lives.
#
# gpu_memory_metrics_otel.cpp/_stub.cpp mirror src/observability/
# CMakeLists.txt's own query_tracing_otel.cpp/query_tracing_stub.cpp split
# (selected by the same KERNELLAKE_ENABLE_OTEL flag) rather than living in
# kernellake_observability itself: that library has no CUDA dependency and
# must stay that way (built in every preset, including CPU-only ones), so
# the two-dimensional gate (needs both CUDA *and* OTel to be the real
# implementation) has to live in whichever module already requires CUDA --
# this one.
add_library(kernellake_memory STATIC
  rmm_environment.cpp
  gpu_memory_metrics.cpp
  $<IF:$<BOOL:${KERNELLAKE_ENABLE_OTEL}>,gpu_memory_metrics_otel.cpp,gpu_memory_metrics_otel_stub.cpp>
)

target_include_directories(kernellake_memory PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
)

target_compile_features(kernellake_memory PUBLIC cxx_std_20)

target_link_libraries(kernellake_memory PUBLIC
  kernellake_common
  rmm::rmm
)

if(KERNELLAKE_ENABLE_OTEL)
  # Metrics API/SDK only -- gpu_memory_metrics_otel.cpp reads the global
  # MeterProvider kernellake_observability's own init() already installed
  # (via opentelemetry::metrics::Provider), it never constructs one or
  # touches an OTLP exporter itself, so it needs none of the exporter
  # targets kernellake_observability links.
  target_link_libraries(kernellake_memory PRIVATE opentelemetry-cpp::metrics)
endif()
