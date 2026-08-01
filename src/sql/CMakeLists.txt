include(ThirdPartySqlParser)

add_library(kernellake_sql STATIC
  parser.cpp
)

target_include_directories(kernellake_sql PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
)

target_compile_features(kernellake_sql PUBLIC cxx_std_20)

target_link_libraries(kernellake_sql PUBLIC
  kernellake_common
PRIVATE
  sqlparser
)
