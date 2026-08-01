add_library(kernellake_optimizer STATIC
  optimizer.cpp
)

target_include_directories(kernellake_optimizer PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
)

target_compile_features(kernellake_optimizer PUBLIC cxx_std_20)

target_link_libraries(kernellake_optimizer PUBLIC
  kernellake_common
  kernellake_types
  kernellake_expression
  kernellake_planner
)
