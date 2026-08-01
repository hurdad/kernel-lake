# Only built when KERNELLAKE_WITH_CUDA is ON (see src/CMakeLists.txt).
add_library(kernellake_execution STATIC
  cuda_utils.cpp
  device_batch.cpp
  arrow_bridge.cpp
)

target_include_directories(kernellake_execution PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
)

target_compile_features(kernellake_execution PUBLIC cxx_std_20)

target_link_libraries(kernellake_execution PUBLIC
  kernellake_common
  kernellake_types
  CUDA::cudart
  rmm::rmm
  cudf::cudf
  Arrow::arrow_static
)
