add_library(kernellake_expression STATIC
  expression.cpp
)

target_include_directories(kernellake_expression PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
)

target_compile_features(kernellake_expression PUBLIC cxx_std_20)

target_link_libraries(kernellake_expression PUBLIC
  kernellake_types
)
