add_library(kernellake_types STATIC
  schema.cpp
  arrow_adapter.cpp
)

target_include_directories(kernellake_types PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
)

target_compile_features(kernellake_types PUBLIC cxx_std_20)

target_link_libraries(kernellake_types PUBLIC
  kernellake_common
  Arrow::arrow_static
)
