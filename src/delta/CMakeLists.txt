add_library(kernellake_delta STATIC
  delta_txn_client.cpp
)

target_include_directories(kernellake_delta PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
)

target_compile_features(kernellake_delta PUBLIC cxx_std_20)

target_link_libraries(kernellake_delta PUBLIC
  kernellake_common
  kernellake_observability
  delta_txn_proto
)
