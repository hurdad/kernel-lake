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
  azure_object_store_test.cpp
  s3_object_store_test.cpp
  hdfs_object_store_test.cpp
  generic_fs_object_store_test.cpp
  object_store_registry_test.cpp
  nvme_object_cache_test.cpp
  parquet_metadata_test.cpp
  parquet_pruning_test.cpp
  table_resolution_test.cpp
  rest_catalog_client_test.cpp
  unity_catalog_client_test.cpp
  unity_catalog_token_cache_test.cpp
  unity_catalog_source_resolver_test.cpp
  manifest_reader_test.cpp
  position_delete_reader_test.cpp
  schema_translation_test.cpp
  partition_pruning_test.cpp
  iceberg_table_resolution_test.cpp
  iceberg_source_resolver_test.cpp
  query_engine_iceberg_test.cpp
  delta_txn_client_test.cpp
  delta_schema_translation_test.cpp
  delta_table_resolution_test.cpp
  delta_source_resolver_test.cpp
  query_engine_delta_test.cpp
  query_engine_unitycatalog_test.cpp
  physical_planner_test.cpp
  query_engine_test.cpp
  query_engine_execute_cpu_test.cpp
  sample_data_generator_test.cpp
  date_util_test.cpp
  logging_test.cpp
  $<$<NOT:$<BOOL:${KERNELLAKE_ENABLE_OTEL}>>:query_tracing_stub_test.cpp>
  result_formatter_test.cpp
  query_command_test.cpp
  benchmark_tpch_command_test.cpp
  inspect_parquet_command_test.cpp
  $<$<BOOL:${KERNELLAKE_BUILD_SERVER}>:flight_sql_server_test.cpp>
  $<$<BOOL:${KERNELLAKE_BUILD_SERVER}>:auth_middleware_test.cpp>
  $<$<BOOL:${KERNELLAKE_BUILD_SERVER}>:gpu_execution_coordinator_test.cpp>
  $<$<BOOL:${KERNELLAKE_ENABLE_OTEL}>:query_tracing_test.cpp>
)

# generic_fs_object_store_test.cpp/hdfs_object_store_test.cpp reach directly
# into src/storage/generic_fs_object_store.hpp -- a private, non-installed
# header shared by the S3/GCS/Azure/HDFS backends (see its own doc comment)
# -- to unit test its detail:: functions (glob_match(), generic_fs_list(),
# strip_hdfs_authority(), ...) directly against a real
# arrow::fs::LocalFileSystem, rather than only indirectly through a backend
# that needs real cloud credentials to even construct.
target_include_directories(kernellake_unit_tests PRIVATE
  "${PROJECT_SOURCE_DIR}/src/storage"
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
  kernellake_iceberg
  kernellake_unitycatalog
  kernellake_delta
  kernellake_api
  kernellake_generator
  kernellake_cli
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
# tests need to know which one they were compiled against. Same idea for
# KERNELLAKE_ENABLE_OTEL: ClientSpan::inject() (kernellake/observability/
# query_tracing.hpp) only actually sends a traceparent header when this
# build has real OTel wired in (see delta_txn_client_test.cpp) -- a no-op
# under the stub build otherwise, which is itself worth asserting on, not
# just skipping.
target_compile_definitions(kernellake_unit_tests PRIVATE
  $<$<BOOL:${KERNELLAKE_WITH_CUDA}>:KERNELLAKE_WITH_CUDA>
  $<$<BOOL:${KERNELLAKE_ENABLE_OTEL}>:KERNELLAKE_ENABLE_OTEL>
)

include(GoogleTest)
gtest_discover_tests(kernellake_unit_tests)
