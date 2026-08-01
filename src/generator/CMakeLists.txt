add_library(kernellake_generator STATIC
  sample_data_generator.cpp
)

target_include_directories(kernellake_generator PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
)

target_compile_features(kernellake_generator PUBLIC cxx_std_20)

target_link_libraries(kernellake_generator PUBLIC
  kernellake_common
  Arrow::arrow_static
  Parquet::parquet_static
)
