add_library(kernellake_storage STATIC
  local_object_store.cpp
  file_discovery.cpp
  generic_fs_object_store.cpp
  s3_object_store.cpp
  gcs_object_store.cpp
  azure_object_store.cpp
  hdfs_object_store.cpp
  object_store_registry.cpp
)

target_include_directories(kernellake_storage PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
)

target_compile_features(kernellake_storage PUBLIC cxx_std_20)

target_link_libraries(kernellake_storage PUBLIC
  kernellake_common
  Arrow::arrow_static
)
