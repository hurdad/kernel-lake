add_library(kernellake_delta STATIC
  delta_txn_client.cpp
  schema_translation.cpp
  delta_table_resolution.cpp
  delta_source_resolver.cpp
)

target_include_directories(kernellake_delta PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
)

target_compile_features(kernellake_delta PUBLIC cxx_std_20)

target_link_libraries(kernellake_delta PUBLIC
  kernellake_common
  kernellake_observability
  kernellake_storage
  kernellake_types
  kernellake_expression
  kernellake_io
  nlohmann_json::nlohmann_json
  delta_txn_proto
)
