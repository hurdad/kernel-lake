add_library(kernellake_common STATIC
  config.cpp
  logging.cpp
  date_util.cpp
)

target_include_directories(kernellake_common PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
)

target_compile_features(kernellake_common PUBLIC cxx_std_20)

# Arrow's GCS filesystem support (arrow::fs::GcsFileSystem, used by
# config.hpp's GcsSection/GcsOptions) pulls in libarrow_bundled_dependencies.a's
# bundled google-cloud-cpp, which needs symbols in this system's real Abseil
# libraries (absl::debian9::StrCat/StrAppend/FormatTime/etc. -- "debian9" is
# this Ubuntu 26.04 apt package's own inline-namespace name for Abseil, an
# ABI-stability convention, not a version number). A --start-group wrap
# declared *here* does not fix this: Arrow::arrow_static is also linked
# plainly by several other kernellake_* targets, and CMake's link-line
# flattening places the actual libarrow.a/libarrow_bundled_dependencies.a
# reference at whatever position that whole-graph ordering picks -- not
# necessarily inside a group declared on this one target, confirmed by an
# actual link attempt still failing with the group here. The real fix lives
# on each final executable target instead (kernellake_unit_tests,
# kernellake-server's own existing group, etc.) -- see their own comments.
target_link_libraries(kernellake_common PUBLIC
  yaml-cpp::yaml-cpp
  spdlog::spdlog
  Arrow::arrow_static
)
