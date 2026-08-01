cmake_minimum_required(VERSION 3.26)

# CUDA is an optional language: it is only enabled when KERNELLAKE_WITH_CUDA is
# turned on, which additionally requires libcudf/RMM to build the GPU
# execution layer. See docs/architecture.md for the CPU/GPU build split.
option(KERNELLAKE_WITH_CUDA "Enable CUDA and GPU execution (requires CUDA toolkit, libcudf, RMM)" OFF)
option(KERNELLAKE_BUILD_TESTS "Build unit and integration tests" ON)
option(KERNELLAKE_BUILD_BENCHMARKS "Build microbenchmarks and TPC-H benchmarks" ON)
option(KERNELLAKE_ENABLE_S3 "Build the optional S3 object store implementation" OFF)

if(KERNELLAKE_WITH_CUDA)
  project(kernellake LANGUAGES CXX CUDA)
else()
  project(kernellake LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(KERNELLAKE_WITH_CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CTest)

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)

if(KERNELLAKE_BUILD_TESTS)
  find_package(GTest REQUIRED)
endif()

if(KERNELLAKE_BUILD_BENCHMARKS)
  find_package(benchmark REQUIRED)
endif()

if(KERNELLAKE_WITH_CUDA)
  find_package(CUDAToolkit REQUIRED)
  find_package(rmm CONFIG REQUIRED)
  find_package(cudf CONFIG REQUIRED)
endif()

add_subdirectory(src)

if(KERNELLAKE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(KERNELLAKE_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
