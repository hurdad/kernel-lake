add_library(kernellake_api STATIC
  query_engine.cpp
)

target_include_directories(kernellake_api PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
)

target_compile_features(kernellake_api PUBLIC cxx_std_20)

target_link_libraries(kernellake_api PUBLIC
  kernellake_common
  kernellake_planner
  kernellake_optimizer
  kernellake_storage
  kernellake_io
  Arrow::arrow_static
)
