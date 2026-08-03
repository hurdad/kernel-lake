add_executable(kernellake
  main.cpp
  inspect_parquet_command.cpp
  explain_command.cpp
  query_command.cpp
  result_formatter.cpp
  generate_data_command.cpp
  benchmark_tpch_command.cpp
)

target_link_libraries(kernellake PRIVATE
  kernellake_common
  kernellake_storage
  kernellake_io
  kernellake_api
  kernellake_generator
  kernellake_observability
  Arrow::arrow_static
  nlohmann_json::nlohmann_json
)

# See cmake/LinkArrowBundledCloudDeps.cmake.
kernellake_link_arrow_bundled_cloud_deps(kernellake)
