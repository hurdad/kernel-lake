add_library(kernellake_unitycatalog STATIC
  unity_catalog_client.cpp
  unity_catalog_source_resolver.cpp
)

target_include_directories(kernellake_unitycatalog PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
)

target_compile_features(kernellake_unitycatalog PUBLIC cxx_std_20)

target_link_libraries(kernellake_unitycatalog PUBLIC
  kernellake_common
  kernellake_storage
  kernellake_types
  kernellake_expression
  kernellake_io
  kernellake_iceberg
  kernellake_delta
  nlohmann_json::nlohmann_json
)
