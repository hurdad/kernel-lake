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
  parquet_metadata_test.cpp
  physical_planner_test.cpp
  query_engine_test.cpp
  query_engine_execute_cpu_test.cpp
  sample_data_generator_test.cpp
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

# kernellake_unit_tests is built (and run) under both the CPU-only `dev`
# preset and the GPU-enabled `gpu-dev` preset. QueryEngine::execute()'s
# actual behavior differs between the two (see query_engine_test.cpp), so
# tests need to know which one they were compiled against.
target_compile_definitions(kernellake_unit_tests PRIVATE
  $<$<BOOL:${KERNELLAKE_WITH_CUDA}>:KERNELLAKE_WITH_CUDA>
)

include(GoogleTest)
gtest_discover_tests(kernellake_unit_tests)
