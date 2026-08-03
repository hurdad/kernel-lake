cmake_minimum_required(VERSION 3.29)

# CUDA is an optional language: it is only enabled when KERNELLAKE_WITH_CUDA is
# turned on, which additionally requires libcudf/RMM to build the GPU
# execution layer. See docs/ARCHITECTURE.md for the CPU/GPU build split.
option(KERNELLAKE_WITH_CUDA "Enable CUDA and GPU execution (requires CUDA toolkit, libcudf, RMM)" OFF)
option(KERNELLAKE_BUILD_TESTS "Build unit and integration tests" ON)
option(KERNELLAKE_BUILD_BENCHMARKS "Build microbenchmarks and TPC-H benchmarks" ON)
# Requires libarrow-flight-dev, libarrow-flight-sql-dev, and gRPC (see
# docs/ARCHITECTURE.md's "Ubuntu 26.04 baseline" section -- Arrow Flight SQL
# does not link on Ubuntu 24.04's system Abseil).
option(KERNELLAKE_BUILD_SERVER "Build the Arrow Flight SQL server (kernellake-server)" OFF)
# Requires opentelemetry-cpp-dev (apt-native on Ubuntu 26.04). OTLP/gRPC
# export only -- see docs/ARCHITECTURE.md.
option(KERNELLAKE_ENABLE_OTEL "Enable OpenTelemetry OTLP/gRPC tracing+metrics+logs export" OFF)

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
find_package(ArrowCompute REQUIRED)
find_package(ArrowAcero REQUIRED)
find_package(Parquet REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
# Needed by every executable linking kernellake_common (see
# tests/unit/CMakeLists.txt's own comment) to resolve
# libarrow_bundled_dependencies.a's bundled google-cloud-cpp (GCS) and Azure
# SDK C++ (Azure) code -- pulled in as soon as anything includes
# <arrow/filesystem/gcsfs.h>/<arrow/filesystem/azurefs.h>, which
# config.hpp now does unconditionally. google-cloud-cpp needs many more
# Abseil components than gRPC alone did; the Azure SDK's bundled XML
# parsing (used for its REST API responses) needs libxml2, and its request-
# ID generation needs libuuid -- neither was a dependency of this project
# before. Confirmed for real via actual link failures, not assumed from
# documentation.
find_package(absl CONFIG REQUIRED)
find_package(LibXml2 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(UUID REQUIRED IMPORTED_TARGET uuid)
include(LinkArrowBundledCloudDeps)

if(KERNELLAKE_BUILD_TESTS)
  find_package(GTest REQUIRED)
endif()

if(KERNELLAKE_BUILD_BENCHMARKS)
  find_package(benchmark REQUIRED)
endif()

if(KERNELLAKE_WITH_CUDA)
  find_package(CUDAToolkit REQUIRED)
  include(ThirdPartyRapids)
endif()

if(KERNELLAKE_BUILD_SERVER)
  find_package(ArrowFlight REQUIRED)
  find_package(ArrowFlightSql REQUIRED)
  find_package(gRPC REQUIRED)
endif()

if(KERNELLAKE_ENABLE_OTEL)
  # exporters_in_memory is only used by tests/unit/query_tracing_test.cpp,
  # but it's small and header-light enough to just always request alongside
  # the others rather than threading a KERNELLAKE_BUILD_TESTS condition
  # through here too. exporters_otlp_http backs observability.otlp_protocol
  # == "http" (see include/kernellake/common/config.hpp).
  find_package(opentelemetry-cpp CONFIG REQUIRED
    COMPONENTS sdk exporters_otlp_grpc exporters_otlp_http exporters_in_memory)
endif()

add_subdirectory(src)

if(KERNELLAKE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(KERNELLAKE_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
