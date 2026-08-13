# Split out from the `kernellake` executable so tests/unit/CMakeLists.txt can
# link the actual command/formatter logic (pure (args, config) -> int
# functions, or free functions over QueryResult) without dragging in main()
# -- src/cli/ otherwise had zero test coverage. main.cpp itself stays out of
# this library (it's the one file that isn't independently testable, being
# entry-point wiring/dispatch only).
add_library(kernellake_cli STATIC
  inspect_parquet_command.cpp
  explain_command.cpp
  query_command.cpp
  result_formatter.cpp
  generate_data_command.cpp
  benchmark_tpch_command.cpp
)

target_include_directories(kernellake_cli PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
)

target_compile_features(kernellake_cli PUBLIC cxx_std_20)

target_link_libraries(kernellake_cli PUBLIC
  kernellake_common
  kernellake_storage
  kernellake_io
  kernellake_api
  kernellake_generator
  kernellake_observability
  Arrow::arrow_static
  nlohmann_json::nlohmann_json
)

add_executable(kernellake main.cpp)

target_link_libraries(kernellake PRIVATE kernellake_cli)

# See cmake/LinkArrowBundledCloudDeps.cmake.
kernellake_link_arrow_bundled_cloud_deps(kernellake)
