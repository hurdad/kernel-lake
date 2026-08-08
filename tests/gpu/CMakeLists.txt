# GPU-specific tests, clearly separated from tests/unit (which must never
# require a GPU). Only built when KERNELLAKE_WITH_CUDA is ON; see
# docs/ARCHITECTURE.md for the CPU/GPU build split and
# docs/ROADMAP.md for why "skip cleanly when no GPU" handling matters even
# though this build already requires CUDA to configure at all.
add_executable(kernellake_gpu_tests
  rmm_environment_test.cpp
  cuda_utils_test.cpp
  device_batch_test.cpp
  arrow_bridge_test.cpp
  expression_compiler_test.cpp
  filter_projection_operator_test.cpp
  scalar_aggregate_operator_test.cpp
  hash_aggregate_operator_test.cpp
  parquet_scan_operator_test.cpp
  limit_operator_test.cpp
  sort_operator_test.cpp
  query_engine_execute_test.cpp
  multi_batch_integration_test.cpp
  decimal_test.cpp
  hash_join_test.cpp
  gpu_memory_metrics_test.cpp
  # OTel-specific: asserts real ObservableGauge/ObservableCounter callback
  # values via kernellake_observability's own in-memory test seam -- see
  # that file's own comment. Only compiled/linked when this build actually
  # has OTel wired in, same condition tests/unit/CMakeLists.txt uses for
  # query_tracing_test.cpp.
  $<$<BOOL:${KERNELLAKE_ENABLE_OTEL}>:gpu_memory_metrics_otel_test.cpp>
)

target_link_libraries(kernellake_gpu_tests PRIVATE
  kernellake_common
  kernellake_types
  kernellake_memory
  kernellake_execution_gpu
  kernellake_api
  Parquet::parquet_static
  GTest::gtest
  GTest::gtest_main
)

if(KERNELLAKE_ENABLE_OTEL)
  # "exporters_in_memory" is a find_package component name, not a CMake
  # target -- these two are the actual targets it provides (mirrors
  # tests/unit/CMakeLists.txt's own comment on the same point).
  # in_memory_span_exporter is needed transitively by
  # observability::init_for_testing() itself (query_tracing_test_support.cpp),
  # not by anything gpu_memory_metrics_otel_test.cpp calls directly.
  target_link_libraries(kernellake_gpu_tests PRIVATE
    kernellake_observability
    opentelemetry-cpp::in_memory_span_exporter
    opentelemetry-cpp::in_memory_metric_exporter
  )
endif()

# See cmake/LinkArrowBundledCloudDeps.cmake.
kernellake_link_arrow_bundled_cloud_deps(kernellake_gpu_tests)

include(GoogleTest)
gtest_discover_tests(kernellake_gpu_tests)
