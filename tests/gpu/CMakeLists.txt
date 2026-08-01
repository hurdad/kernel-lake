# GPU-specific tests, clearly separated from tests/unit (which must never
# require a GPU). Only built when KERNELLAKE_WITH_CUDA is ON; see
# docs/architecture.md for the CPU/GPU build split and
# docs/roadmap.md for why "skip cleanly when no GPU" handling matters even
# though this build already requires CUDA to configure at all.
add_executable(kernellake_gpu_tests
  rmm_environment_test.cpp
  cuda_utils_test.cpp
  device_batch_test.cpp
  arrow_bridge_test.cpp
  expression_compiler_test.cpp
  filter_projection_operator_test.cpp
)

target_link_libraries(kernellake_gpu_tests PRIVATE
  kernellake_common
  kernellake_types
  kernellake_memory
  kernellake_execution
  GTest::gtest
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(kernellake_gpu_tests)
