add_library(kernellake_planner STATIC
  binder.cpp
  logical_plan.cpp
  logical_planner.cpp
  physical_plan.cpp
)

target_include_directories(kernellake_planner PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
)

target_compile_features(kernellake_planner PUBLIC cxx_std_20)

target_link_libraries(kernellake_planner PUBLIC
  kernellake_common
  kernellake_types
  kernellake_expression
  kernellake_sql
  kernellake_storage
  nlohmann_json::nlohmann_json
)
