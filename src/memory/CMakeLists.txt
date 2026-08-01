# Only built when KERNELLAKE_WITH_CUDA is ON (see src/CMakeLists.txt).
add_library(kernellake_memory STATIC
  rmm_environment.cpp
)

target_include_directories(kernellake_memory PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
)

target_compile_features(kernellake_memory PUBLIC cxx_std_20)

target_link_libraries(kernellake_memory PUBLIC
  kernellake_common
  rmm::rmm
)
