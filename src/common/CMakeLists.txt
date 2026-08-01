add_library(kernellake_common STATIC
  config.cpp
  logging.cpp
  date_util.cpp
)

target_include_directories(kernellake_common PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
)

target_compile_features(kernellake_common PUBLIC cxx_std_20)

target_link_libraries(kernellake_common PUBLIC
  yaml-cpp::yaml-cpp
  spdlog::spdlog
)
