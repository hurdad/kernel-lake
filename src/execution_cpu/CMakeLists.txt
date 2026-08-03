# Always built (both `dev` and `gpu-dev` presets, no CUDA dependency) --
# the Apache Arrow Acero-based CPU execution backend. See
# docs/ARCHITECTURE.md's CPU backend section.
add_library(kernellake_execution_cpu STATIC
  expression_compiler_cpu.cpp
  acero_query_executor.cpp
)

target_include_directories(kernellake_execution_cpu PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
)

target_compile_features(kernellake_execution_cpu PUBLIC cxx_std_20)

# Arrow Compute's kernel functions (sum, hash_sum, cast, greater_equal,
# sort_indices, ...) are not self-registering at static-init time -- Arrow
# requires an explicit arrow::compute::Initialize() call to populate the
# global FunctionRegistry (see arrow/compute/initialize.h: "Register the
# compute kernel functions to be available on the global FunctionRegistry").
# Without it every Acero query fails at runtime with "No function registered
# with name: ...", regardless of which libraries are linked in. See
# acero_query_executor.cpp, which calls it once via std::call_once.
target_link_libraries(kernellake_execution_cpu PUBLIC
  kernellake_common
  kernellake_types
  kernellake_expression
  kernellake_planner
  kernellake_storage
  Arrow::arrow_static
  ArrowCompute::arrow_compute_static
  ArrowAcero::arrow_acero_static
  Parquet::parquet_static
)
