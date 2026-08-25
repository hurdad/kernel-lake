#pragma once

// Shared test-only fixture pieces for src/cli/ command tests that need a
// minimal real Parquet source and a CPU-backend CliConfig -- used by any
// test file driving a run_*() command function directly (see
// query_command_test.cpp/explain_command_test.cpp, both of which used to
// hand-duplicate this exact setup).
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <cstdint>
#include <string>

#include "kernellake/common/config.hpp"

namespace kernellake::cli {

[[nodiscard]] inline CliConfig cpu_backend_config() {
  CliConfig config = default_cli_config();
  config.engine_config.engine.backend = "cpu";
  return config;
}

// Writes a single-column ("id", INT64, 0..row_count-1) Parquet file to
// `path` -- the minimal source every run_query()/run_explain() test needs
// to bind and execute a real query against.
inline void write_id_column_parquet(const std::string& path, std::int64_t row_count = 3) {
  arrow::Int64Builder id_builder;
  for (std::int64_t i = 0; i < row_count; ++i) {
    ASSERT_TRUE(id_builder.Append(i).ok());
  }
  std::shared_ptr<arrow::Array> id_array;
  ASSERT_TRUE(id_builder.Finish(&id_array).ok());
  const auto schema = arrow::schema({arrow::field("id", arrow::int64(), false)});
  const auto table = arrow::Table::Make(schema, {id_array});
  auto sink = arrow::io::FileOutputStream::Open(path).ValueOrDie();
  const arrow::Status status =
      parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, /*chunk_size=*/row_count);
  ASSERT_TRUE(status.ok()) << status.ToString();
}

}  // namespace kernellake::cli
