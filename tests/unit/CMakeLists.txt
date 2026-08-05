add_executable(kernellake_unit_tests
  common_errors_test.cpp
  config_test.cpp
  schema_test.cpp
  arrow_adapter_test.cpp
  expression_test.cpp
  sql_parser_test.cpp
  binder_test.cpp
  logical_plan_test.cpp
  optimizer_test.cpp
  storage_test.cpp
  gcs_object_store_test.cpp
  object_store_registry_test.cpp
  parquet_metadata_test.cpp
  parquet_pruning_test.cpp
  physical_planner_test.cpp
  query_engine_test.cpp
  query_engine_execute_cpu_test.cpp
  sample_data_generator_test.cpp
  $<$<BOOL:${KERNELLAKE_BUILD_SERVER}>:flight_sql_server_test.cpp>
  $<$<BOOL:${KERNELLAKE_ENABLE_OTEL}>:query_tracing_test.cpp>
)

target_link_libraries(kernellake_unit_tests PRIVATE
  kernellake_common
  kernellake_types
  kernellake_expression
  kernellake_sql
  kernellake_planner
  kernellake_optimizer
  kernellake_storage
  kernellake_io
  kernellake_api
  kernellake_generator
  GTest::gtest
  GTest::gtest_main
)

# See cmake/LinkArrowBundledCloudDeps.cmake for why this is needed: config.hpp
# unconditionally pulls in Arrow's GCS/Azure filesystem support, which needs
# extra Abseil symbols plus libxml2/libuuid that a plain link can't resolve.
kernellake_link_arrow_bundled_cloud_deps(kernellake_unit_tests)

if(KERNELLAKE_BUILD_SERVER)
  # See src/server/CMakeLists.txt's comment on why this needs a raw
  # -Wl,--start-group/--end-group wrap, not just a plain link.
  target_link_libraries(kernellake_unit_tests PRIVATE
    -Wl,--start-group
    kernellake_flight_sql_server
    ArrowFlightSql::arrow_flight_sql_static
    ArrowFlight::arrow_flight_static
    gRPC::grpc++
    -Wl,--end-group
  )
endif()

if(KERNELLAKE_ENABLE_OTEL)
  # "exporters_in_memory" is a find_package component name, not a CMake
  # target -- these two are the actual targets it provides.
  target_link_libraries(kernellake_unit_tests PRIVATE
    kernellake_observability
    opentelemetry-cpp::in_memory_span_exporter
    opentelemetry-cpp::in_memory_metric_exporter
  )
endif()

# kernellake_unit_tests is built (and run) under both the CPU-only `dev`
# preset and the GPU-enabled `gpu-dev` preset. QueryEngine::execute()'s
# actual behavior differs between the two (see query_engine_test.cpp), so
# tests need to know which one they were compiled against.
target_compile_definitions(kernellake_unit_tests PRIVATE
  $<$<BOOL:${KERNELLAKE_WITH_CUDA}>:KERNELLAKE_WITH_CUDA>
)

include(GoogleTest)
gtest_discover_tests(kernellake_unit_tests)
