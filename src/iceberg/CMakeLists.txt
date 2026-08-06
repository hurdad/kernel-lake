add_library(kernellake_iceberg STATIC
  rest_catalog_client.cpp
  manifest_reader.cpp
  schema_translation.cpp
  partition_pruning.cpp
  iceberg_table_resolution.cpp
  iceberg_source_resolver.cpp
)

target_include_directories(kernellake_iceberg PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
)

target_compile_features(kernellake_iceberg PUBLIC cxx_std_20)

target_link_libraries(kernellake_iceberg PUBLIC
  kernellake_common
  kernellake_storage
  kernellake_types
  kernellake_expression
  kernellake_io
  PkgConfig::CURL
  nlohmann_json::nlohmann_json
  Avro::avro
  Arrow::arrow_static
  Parquet::parquet_static
)
