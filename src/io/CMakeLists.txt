add_library(kernellake_io STATIC
  parquet_metadata.cpp
  parquet_pruning.cpp
  physical_planner.cpp
  table_resolution.cpp
)

target_include_directories(kernellake_io PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
)

target_compile_features(kernellake_io PUBLIC cxx_std_20)

target_link_libraries(kernellake_io PUBLIC
  kernellake_common
  kernellake_types
  kernellake_expression
  kernellake_storage
  kernellake_planner
  Parquet::parquet_static
  Arrow::arrow_static
)
