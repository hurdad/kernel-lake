# nvme_cache_metrics_otel.cpp/_stub.cpp mirror src/observability/
# CMakeLists.txt's own query_tracing_otel.cpp/query_tracing_stub.cpp split
# (selected by KERNELLAKE_ENABLE_OTEL alone, no CUDA gate needed -- unlike
# src/memory/CMakeLists.txt's two-dimensional gate, neither
# ObjectStoreRegistry nor NvmeObjectCache have a CUDA dependency) rather
# than living in kernellake_observability itself, for the same reason that
# module gives: it must stay CUDA/OTel-optional-dependency-free and built
# in every preset.
add_library(kernellake_storage STATIC
  local_object_store.cpp
  file_discovery.cpp
  generic_fs_object_store.cpp
  s3_object_store.cpp
  gcs_object_store.cpp
  azure_object_store.cpp
  hdfs_object_store.cpp
  object_store_registry.cpp
  nvme_object_cache.cpp
  $<IF:$<BOOL:${KERNELLAKE_ENABLE_OTEL}>,nvme_cache_metrics_otel.cpp,nvme_cache_metrics_otel_stub.cpp>
)

target_include_directories(kernellake_storage PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
)

target_compile_features(kernellake_storage PUBLIC cxx_std_20)

target_link_libraries(kernellake_storage PUBLIC
  kernellake_common
  Arrow::arrow_static
)

if(KERNELLAKE_ENABLE_OTEL)
  # Metrics API/SDK only -- nvme_cache_metrics_otel.cpp reads the global
  # MeterProvider kernellake_observability's own init() already installed,
  # it never constructs one or touches an OTLP exporter itself.
  target_link_libraries(kernellake_storage PRIVATE opentelemetry-cpp::metrics)
endif()
