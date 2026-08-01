# Vendors hyrise/sql-parser (MIT license), a small, permissively licensed
# SQL parser, per the KernelLake spec's instruction to use an existing
# parser rather than hand-write one. The upstream repository has no
# CMakeLists.txt of its own (it ships a plain Makefile and pre-generated
# flex/bison output), so we declare our own `sqlparser` target listing its
# sources directly instead of add_subdirectory-ing it.
#
# Pinned to a specific commit for reproducible builds.
include(FetchContent)

FetchContent_Declare(
  hsql
  GIT_REPOSITORY https://github.com/hyrise/sql-parser.git
  GIT_TAG ccd3f68b50bb2b96ce69afa7b956b3bd826643cc
)

FetchContent_MakeAvailable(hsql)

if(NOT TARGET sqlparser)
  add_library(sqlparser STATIC
    "${hsql_SOURCE_DIR}/src/SQLParser.cpp"
    "${hsql_SOURCE_DIR}/src/SQLParserResult.cpp"
    "${hsql_SOURCE_DIR}/src/parser/bison_parser.cpp"
    "${hsql_SOURCE_DIR}/src/parser/flex_lexer.cpp"
    "${hsql_SOURCE_DIR}/src/sql/CreateStatement.cpp"
    "${hsql_SOURCE_DIR}/src/sql/Expr.cpp"
    "${hsql_SOURCE_DIR}/src/sql/PrepareStatement.cpp"
    "${hsql_SOURCE_DIR}/src/sql/SQLStatement.cpp"
    "${hsql_SOURCE_DIR}/src/sql/statements.cpp"
    "${hsql_SOURCE_DIR}/src/util/sqlhelper.cpp"
  )
  target_include_directories(sqlparser SYSTEM PUBLIC
    "${hsql_SOURCE_DIR}/src"
  )
  # Upstream targets C++17 and builds with -Wall -Werror under its own
  # Makefile; we compile it under our own flags (no -Werror) so upstream
  # warnings never break the KernelLake build.
  target_compile_features(sqlparser PRIVATE cxx_std_17)
endif()
